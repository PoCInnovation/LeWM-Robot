"""
Campagne d'entraînement sans surveillance (RTX 4090 louée).

Enchaîne, d'après un fichier de campagne (configs/campaign_rtx4090.yaml) :
    02 (si les latents manquent) → 03 une fois (choix + poids de la fusion)
    → sweeps de 04 → LoRA 05 (si démos réelles) → réglage du CEM (08)

    - chaque run a son dossier results/campaign/<name>/<étape>/<run>/ avec
      son log ; il est sauté s'il est terminé (*.done.json / *.json) ;
    - un run 04 interrompu reprend à son dernier epoch (--resume) et son run
      W&B continue (WANDB_RUN_ID déterministe + WANDB_RESUME=allow) ;
    - un run qui échoue (OOM...) est noté et la campagne continue ;
    - le meilleur predictor (val loss la plus basse parmi ceux qui battent
      la baseline « rien ne bouge », ratio < 1) est copié dans
      results/campaign/<name>/best_predictor.pt ; idem LoRA ; le meilleur
      réglage CEM va dans best.json ;
    - une étape 08 peut viser un checkpoint existant (`predictor_ckpt`, sans
      réentraîner) ou chaque run d'une étape 04 (`checkpoints_from`), au
      lieu du meilleur predictor de la campagne.

Durée : --dry-run liste les runs et le nombre d'epochs par taille de modèle ;
pendant la campagne, le temps d'epoch MESURÉ de chaque run terminé sert à
projeter le temps restant.

Usage:
    python scripts/campaign.py --campaign configs/campaign_rtx4090.yaml [--dry-run]
"""

import sys
import json
import os
import shutil
import hashlib
import argparse
import itertools
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml

from src.config import load_config

SCRIPTS = {"02": "02_encode_dataset.py", "03": "03_compare_fusion.py",
           "04": "04_train_predictor.py", "05": "05_train_lora.py",
           "08": "08_eval_planner.py"}


def resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def read_json(path: Path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def fmt(v) -> str:
    return f"{v:g}" if isinstance(v, float) else str(v)


def run_name(params: dict) -> str:
    short = {"lr": "lr", "n_layers": "L", "weight_decay": "wd", "dropout": "do",
             "lora_rank": "r", "horizon": "H", "n_samples": "N", "n_iter": "it",
             "cost": "", "seed": "seed", "rollout_horizon": "RH", "goal_steps": "G"}
    parts = []
    for k, v in params.items():
        if k == "residual":
            parts.append("res" if v else "abs")
        elif k in short:
            parts.append(f"{short[k]}{fmt(v)}")
    return "-".join(parts)


def expand(stage: dict, previous: dict) -> list:
    """Liste des paramètres de chaque run de l'étape."""
    bases = [{}]
    if stage.get("from_best"):
        fb = stage["from_best"]
        ranked = ranked_runs(previous.get(fb["stage"], []))[:fb["top"]]
        bases = [{k: r["params"][k] for k in fb["keep"]} for r in ranked]
    combos = [{}]
    if "grid" in stage:
        keys = list(stage["grid"])
        combos = [dict(zip(keys, values))
                  for values in itertools.product(*stage["grid"].values())]
    if "variants" in stage:
        combos = [dict(c, **v) for c in combos for v in stage["variants"]]
    return [dict(stage.get("fixed", {}), **base, **combo)
            for base in bases for combo in combos]


def ranked_runs(runs: list) -> list:
    """Runs terminés qui battent la baseline identité, meilleure val loss d'abord."""
    ok = [r for r in runs if r.get("result")
          and (r["result"].get("best_ratio_vs_identity") or 99) < 1.0]
    return sorted(ok, key=lambda r: r["result"]["best_val_loss"])


def stream(cmd: list, log_path: Path, env: dict) -> int:
    """Lance cmd, recopie sa sortie à l'écran et dans log_path (en ajout)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as log:
        log.write(f"\n$ {' '.join(cmd)}\n")
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return proc.wait()


def check_wandb(cfg: dict) -> tuple:
    """
    Vérifie avant de lancer que les runs seront visibles en ligne sur W&B.

    Sans clé, src.tracking passe chaque run en offline sans s'arrêter : la
    campagne tournerait des heures sans rien afficher sur le site.

    Returns:
        (ok, message) — ok False : la campagne ne doit pas démarrer.
    """
    from src.tracking import DEFAULT_PROJECT, _has_netrc_entry
    yaml_wandb = cfg.get("wandb") or {}
    project = (os.environ.get("WANDB_PROJECT") or yaml_wandb.get("project")
               or DEFAULT_PROJECT)
    mode = os.environ.get("WANDB_MODE") or yaml_wandb.get("mode")
    if mode in ("offline", "disabled", "dryrun"):
        return True, f"W&B en mode {mode} (choisi explicitement) : rien en ligne"
    try:
        import wandb
    except ImportError:
        return False, "wandb non installé : pip install -r requirements_wm.txt"
    if not (os.environ.get("WANDB_API_KEY") or _has_netrc_entry()):
        return False, ("pas de clé W&B : export WANDB_API_KEY=... (ou wandb login). "
                       "Pour lancer quand même sans suivi en ligne : WANDB_MODE=offline")
    try:
        wandb.login(verify=True, timeout=30)
        entity = wandb.Api(timeout=30).default_entity
    except Exception as exc:
        return False, (f"clé W&B refusée ou serveur injoignable "
                       f"({type(exc).__name__}: {exc})")
    return True, f"W&B en ligne : https://wandb.ai/{entity}/{project}"


class Campaign:
    def __init__(self, path: Path, dry_run: bool):
        with open(path) as f:
            self.spec = yaml.safe_load(f)
        self.dry_run = dry_run
        self.config_path = self.spec["config"]
        self.cfg = load_config(resolve(self.config_path))
        self.dir = resolve("results/campaign") / self.spec["name"]
        self.encoded = self.spec["encoded_data"]
        self.cem_data = self.encoded       # devient real_data si la LoRA a tourné
        self.results: dict = {}           # étape → [{name, params, result}]
        self.failures: list = []
        self.epoch_seconds: dict = {}     # "L<couches>-RH<horizon>" → [s/epoch mesurés]

    # ── utilitaires ──────────────────────────────────────────────
    def env_for(self, stage: str, name: str) -> dict:
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        if "WANDB_RUN_ID" in env:
            del env["WANDB_RUN_ID"]
        key = f"{self.spec['name']}/{stage}/{name}"
        env["WANDB_RUN_ID"] = hashlib.sha1(key.encode()).hexdigest()[:16]
        env["WANDB_RESUME"] = "allow"
        return env

    def py(self, script: str, *args) -> list:
        return [sys.executable, str(ROOT / "scripts" / SCRIPTS[script]),
                "--config", self.config_path, *map(str, args)]

    def wandb_args(self, stage: str, name: str) -> list:
        return ["--wandb-group", f"{self.spec['name']}-{stage}",
                "--wandb-name", f"{stage}-{name}"]

    # ── 02 / 03 ──────────────────────────────────────────────────
    def ensure_encoded(self):
        if resolve(self.encoded).exists():
            return
        print(f"[Campagne] latents absents → 02 ({self.encoded})")
        if self.dry_run:
            return
        code = stream(self.py("02", "--output", self.encoded,
                              *map(str, self.spec.get("encode_args", []))),
                      self.dir / "02_encode.log", dict(os.environ, PYTHONUNBUFFERED="1"))
        if code != 0:
            sys.exit("[Campagne] 02 a échoué — voir 02_encode.log")

    def choose_fusion(self) -> str:
        if self.spec.get("fusion", "auto") != "auto":
            return self.spec["fusion"]
        result = self.dir / "fusion_comparison.json"
        if not result.exists() and not self.dry_run:
            print("[Campagne] 03 : comparaison des fusions (une fois par campagne)")
            code = stream(self.py("03", "--encoded-data", self.encoded,
                                  "--wandb-group", f"{self.spec['name']}-03"),
                          self.dir / "03_fusion.log", dict(os.environ, PYTHONUNBUFFERED="1"))
            src = resolve(self.cfg["paths"].get("fusion_results",
                                                "results/fusion_comparison.json"))
            if code != 0 or not src.exists():
                sys.exit("[Campagne] 03 a échoué — voir 03_fusion.log")
            self.dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, result)
        data = read_json(result)
        if data is None:
            return "<gagnante de 03>"
        return min(data, key=lambda k: data[k]["best_val_mae"])

    # ── étapes ───────────────────────────────────────────────────
    def run_stage(self, stage: dict, fusion: str, best_ckpt: str):
        name_s = stage["name"]
        if stage.get("requires") and not self.spec.get(stage["requires"]):
            print(f"\n[Campagne] {name_s} sautée ({stage['requires']} non défini)")
            return
        if (stage["script"] == "05" and not self.dry_run
                and not resolve(self.spec["real_data"]).exists()):
            sys.exit(f"[Campagne] real_data introuvable : {self.spec['real_data']} — encoder "
                     "les démos réelles d'abord : python scripts/02_encode_dataset.py "
                     f"--config {self.config_path} --dataset-id <démos> --output {self.spec['real_data']}")
        runs = expand(stage, self.results)
        targets = [("", best_ckpt)]
        if stage.get("predictor_ckpt"):
            targets = [("", stage["predictor_ckpt"])]
        elif stage.get("checkpoints_from"):
            src = [r for r in self.results.get(stage["checkpoints_from"], [])
                   if r.get("result") or self.dry_run]
            targets = [(r["name"] + "/", str(Path(r["dir"]) / "predictor.pt")) for r in src]
            if not targets:
                print(f"\n[Campagne] {name_s} sautée (aucun run terminé dans "
                      f"{stage['checkpoints_from']})")
                return
        print(f"\n══════ [Campagne] {name_s} : {len(runs) * len(targets)} runs")
        self.results[name_s] = []
        plans = []
        for prefix, ckpt in targets:
            for params in runs:
                name = prefix + run_name(params)
                run_dir = self.dir / name_s / name
                done, cmd = self.command(stage, params, run_dir, fusion, ckpt, name)
                plans.append((params, name, run_dir, done, cmd))
        for i, (params, name, run_dir, done, cmd) in enumerate(plans):
            entry = {"name": name, "params": params, "dir": str(run_dir)}
            self.results[name_s].append(entry)
            if done.exists():
                entry["result"] = read_json(done)
                self.record_timing(params, entry["result"])
                print(f"  ✓ {name} (déjà fait)")
                continue
            if self.dry_run:
                print(f"  · {name}")
                continue
            if stage["script"] == "04":
                self.print_eta([(q, stage["epochs"])
                                for q, _, _, d, _ in plans[i:] if not d.exists()])
            print(f"\n──── {name_s}/{name}")
            t0 = time.time()
            code = stream(cmd, run_dir / "run.log", self.env_for(name_s, name))
            entry["wall_seconds"] = time.time() - t0
            if code != 0 or not done.exists():
                print(f"[Campagne] ÉCHEC {name_s}/{name} (code {code}) — voir {run_dir}/run.log")
                self.failures.append(f"{name_s}/{name}")
                continue
            entry["result"] = read_json(done)
            self.record_timing(params, entry["result"])
        self.save_state()

    def command(self, stage, params, run_dir, fusion, best_ckpt, name):
        s = stage["script"]
        wb = self.wandb_args(stage["name"], name.replace("/", "-"))
        if s == "04":
            out = run_dir / "predictor.pt"
            return out.with_name("predictor.done.json"), self.py(
                "04", "--encoded-data", self.encoded, "--output", out,
                "--fusion", fusion, "--n-epochs", stage["epochs"], "--resume",
                *self.flags(params), *wb)
        if s == "05":
            out = run_dir / "predictor_real.pt"
            return out.with_name("predictor_real.done.json"), self.py(
                "05", "--real-data", self.spec["real_data"], "--predictor-ckpt", best_ckpt,
                "--output", out, "--n-epochs", stage["epochs"], *self.flags(params), *wb)
        if s == "08":
            out = run_dir / "cem_eval.json"
            return out, self.py(
                "08", "--encoded-data", stage.get("cem_data", self.cem_data),
                "--predictor-ckpt", best_ckpt, "--output", out, *self.flags(params), *wb)
        raise ValueError(f"script inconnu : {s}")

    @staticmethod
    def flags(params: dict) -> list:
        out = []
        for k, v in params.items():
            flag = f"--{k.replace('_', '-')}"
            if isinstance(v, bool):          # drapeau store_true
                out += [flag] if v else []
            else:
                out += [flag, fmt(v)]
        return out

    # ── temps ────────────────────────────────────────────────────
    @staticmethod
    def size_of(params) -> tuple:
        """(couches, horizon de rollout) : ce qui fixe le temps d'epoch de 04."""
        return params["n_layers"], params.get("rollout_horizon", 1)

    def record_timing(self, params, result):
        if result and result.get("epoch_seconds_mean") and "n_layers" in params:
            L, h = self.size_of(params)
            self.epoch_seconds.setdefault(f"L{L}-RH{h}", []).append(
                result["epoch_seconds_mean"])

    def t_epoch(self, params):
        """s/epoch mesuré pour cette taille, sinon extrapolé ∝ couches × horizon."""
        L, h = self.size_of(params)
        if f"L{L}-RH{h}" in self.epoch_seconds:
            v = self.epoch_seconds[f"L{L}-RH{h}"]
            return sum(v) / len(v), True
        if self.epoch_seconds:
            key, v = next(iter(self.epoch_seconds.items()))
            L0, h0 = (int(x[1:] if x[0] == "L" else x[2:]) for x in key.split("-"))
            return sum(v) / len(v) * (L * h) / (L0 * h0), False
        return None, False

    def print_eta(self, pending: list):
        """pending : [(params, epochs)] des runs 04 restant dans l'étape."""
        if not pending:
            return
        total, exact = 0.0, True
        for params, epochs in pending:
            t, measured = self.t_epoch(params)
            if t is None:
                print("[Campagne] temps d'epoch pas encore mesuré (premier run de 04).")
                return
            total += epochs * t
            exact &= measured
        print(f"[Campagne] reste dans l'étape : {sum(e for _, e in pending)} epochs "
              f"≈ {total / 60:.0f} min{'' if exact else ' (en partie extrapolé ∝ couches × horizon)'}"
              " + ~1 min de chargement par run")

    @staticmethod
    def cem_table(stage_name: str, evals: list):
        """Tous les réglages CEM de l'étape (ratio < 1 : mieux que l'action moyenne)."""
        if not evals:
            return
        print(f"[Campagne] {stage_name} — ratio MAE vs action moyenne "
              "(1er pas [IC 95 %] / séquence), ms/plan :")
        for r in evals:
            res = r["result"]
            ci = res.get("mae_first_ratio_ci95")
            ci = f" [{ci[0]:.3f}–{ci[1]:.3f}]" if ci else ""
            print(f"  {r['name']:<48} {res['mae_first_ratio']:.3f}{ci} / "
                  f"{res['mae_seq_ratio']:.3f}  {res['ms_per_plan']:.0f} ms")

    def save_state(self):
        if self.dry_run:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / "state.json", "w") as f:
            json.dump({"results": self.results, "failures": self.failures,
                       "epoch_seconds": self.epoch_seconds}, f, indent=2, default=str)

    def keep_best(self, stages: list, filename: str, ckpt_name: str):
        runs = [r for s in stages for r in self.results.get(s, [])]
        ranked = ranked_runs(runs)
        if not ranked:
            return None
        best = ranked[0]
        src = Path(best["dir"]) / ckpt_name
        dst = self.dir / filename
        if not self.dry_run and src.exists():
            tmp = dst.with_name(dst.name + ".tmp")
            shutil.copy(src, tmp)
            os.replace(tmp, dst)
        print(f"[Campagne] meilleur ({', '.join(stages)}) : {best['name']} — "
              f"val {best['result']['best_val_loss']:.5f}, "
              f"ratio vs identité {best['result']['best_ratio_vs_identity']:.3f} → {dst}")
        return str(dst), best

    def seed_spread(self, stage: dict):
        """
        Dispersion entre seeds de la meilleure config de l'étape source,
        comparée à l'écart avec sa 2ᵉ config : dit si le classement est fiable.
        """
        ranked = ranked_runs(self.results.get(stage["from_best"]["stage"], []))
        seeds = [r for r in self.results.get(stage["name"], []) if r.get("result")]
        if not ranked or not seeds:
            return None
        vals = [r["result"]["best_val_loss"] for r in [ranked[0]] + seeds]
        spread = max(vals) - min(vals)
        out = {"config": ranked[0]["name"], "val_losses": vals,
               "mean": sum(vals) / len(vals), "spread": spread}
        line = (f"[Campagne] seeds de {ranked[0]['name']} : val "
                + " / ".join(f"{v:.5f}" for v in vals)
                + f" (moyenne {out['mean']:.5f}, dispersion {spread:.5f})")
        if len(ranked) > 1:
            gap = ranked[1]["result"]["best_val_loss"] - ranked[0]["result"]["best_val_loss"]
            out.update(second=ranked[1]["name"], gap=gap, significant=gap > spread)
            line += (f"\n[Campagne] écart avec la 2ᵉ config ({ranked[1]['name']}) : {gap:.5f} → "
                     + ("au-delà de la dispersion entre seeds : classement fiable"
                        if gap > spread else
                        "dans la dispersion entre seeds : les deux configs sont équivalentes"))
        print(line)
        return out

    # ── orchestration ────────────────────────────────────────────
    def run(self):
        ok, msg = check_wandb(self.cfg)
        print(f"[Campagne] {msg}")
        if not ok and not self.dry_run:
            sys.exit("[Campagne] arrêt : corriger W&B avant de lancer (voir ci-dessus).")
        if ok and "wandb.ai" in msg:
            print(f"[Campagne] runs groupés par étape : {self.spec['name']}-<étape> "
                  f"(ex. {self.spec['name']}-A_screen), nommés <étape>-<réglages>")
        if not self.dry_run:
            self.dir.mkdir(parents=True, exist_ok=True)
        self.ensure_encoded()
        fusion = self.choose_fusion()
        print(f"[Campagne] fusion : {fusion}")
        best_ckpt, best = None, {}
        for stage in self.spec["stages"]:
            if stage["script"] == "04":
                self.run_stage(stage, fusion, None)
                kept = self.keep_best([s["name"] for s in self.spec["stages"]
                                       if s["script"] == "04"],
                                      "best_predictor.pt", "predictor.pt")
                if kept:
                    best_ckpt, best["predictor"] = kept
                if "seed" in stage.get("grid", {}) and stage.get("from_best"):
                    best["seeds"] = self.seed_spread(stage)
            elif stage["script"] == "05":
                self.run_stage(stage, fusion, best_ckpt)
                kept = self.keep_best([stage["name"]], "best_predictor_real.pt",
                                      "predictor_real.pt")
                if kept:
                    best_ckpt, best["lora"] = kept
                    # modèle adapté aux démos réelles → CEM évalué sur leurs épisodes de val
                    self.cem_data = self.spec["real_data"]
            elif stage["script"] == "08":
                own_ckpt = stage.get("predictor_ckpt") or stage.get("checkpoints_from")
                if best_ckpt is None and not own_ckpt and not self.dry_run:
                    print("[Campagne] pas de predictor qui bat la baseline → CEM sauté")
                    continue
                self.run_stage(stage, fusion, best_ckpt or "<best_predictor.pt>")
                evals = [r for r in self.results.get(stage["name"], []) if r.get("result")]
                self.cem_table(stage["name"], evals)
                if evals:
                    # 1re action : la seule que le MPC exécute avant de replanifier
                    top = min(evals, key=lambda r: (r["result"]["mae_first_ratio"],
                                                    r["result"]["ms_per_plan"]))
                    best["cem"] = top
                    print(f"[Campagne] meilleur CEM : {top['name']} — MAE 1er pas "
                          f"ratio {top['result']['mae_first_ratio']:.3f}, "
                          f"{top['result']['ms_per_plan']:.0f} ms/plan")
        self.save_state()
        if not self.dry_run:
            with open(self.dir / "best.json", "w") as f:
                json.dump(best, f, indent=2, default=str)
        self.report()

    def report(self):
        if self.dry_run:
            print("\n[Campagne] epochs de 04 :")
            for stage in self.spec["stages"]:
                if stage["script"] != "04":
                    continue
                if "from_best" in stage:     # configs pas encore connues : 1 base fictive
                    top = stage["from_best"]["top"]
                    src = stage["from_best"]["stage"]
                    runs = [dict(p, n_layers=p.get("n_layers", f"la taille retenue en {src}"))
                            for p in expand(dict(stage, from_best=None), {})] * top
                else:
                    runs = expand(stage, self.results)
                per_size: dict = {}
                for p in runs:
                    size = self.size_of(p)
                    per_size[size] = per_size.get(size, 0) + stage["epochs"]
                print(f"  {stage['name']} : " + ", ".join(
                    f"{e} epochs à {L}" + (" couches" if isinstance(L, int) else "")
                    + (f" en rollout {h} pas" if h > 1 else "")
                    for (L, h), e in sorted(per_size.items(), key=str)))
            print("Durée de 04 = Σ (epochs × s/epoch mesuré pour la taille et l'horizon) "
                  "+ ~1 min de chargement des latents par run.")
            return
        print(f"\n══════ [Campagne] terminée — {self.dir}")
        if self.failures:
            print(f"[Campagne] runs en échec (relancer la commande pour les retenter) : "
                  f"{', '.join(self.failures)}")


def main():
    # Lancée sous nohup / tee, la sortie n'est pas un terminal : sans ça,
    # Python la bufferise et le log reste vide pendant des minutes.
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", default="configs/campaign_rtx4090.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="Liste les runs et le nombre d'epochs sans rien lancer")
    args = parser.parse_args()
    Campaign(resolve(args.campaign), args.dry_run).run()


if __name__ == "__main__":
    main()
