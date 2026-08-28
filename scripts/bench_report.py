"""
Résumé lisible (summary.md) d'un rapport de benchmark — appelé par bench.sh.

Usage: python scripts/bench_report.py <dossier contenant benchmark.json>
"""
import json
import platform
import sys
from pathlib import Path


def fmt(s):
    if s is None:
        return "n/a"
    if s < 60:
        return f"{s:.0f} s"
    if s < 3600:
        return f"{s / 60:.1f} min"
    return f"{s / 3600:.2f} h"


def main():
    d = Path(sys.argv[1])
    r = json.load(open(d / "benchmark.json"))
    enc_cfg = r["config"]["encoder"]
    L = ["# Benchmark LeWM-Robot — rapport", "",
         f"- GPU : {r['gpu']}",
         f"- Précision : {r['precision']}",
         f"- Encodeur : {enc_cfg.get('family')}-{enc_cfg.get('size')}",
         f"- Latents : {r['latents']['source']} ({r['latents']['n_pairs']} paires, "
         f"{r['latents']['data_device']})",
         f"- Epochs visés : predictor {r['config']['n_epochs']}, "
         f"fusions {r['config']['fusion_epochs']}, LoRA {r['config']['lora_epochs']}",
         f"- Machine : {platform.platform()}",
         "", "## Estimation d'un run complet", "",
         "| Étape | Temps estimé |", "|---|---|"]
    for k, v in r["summary"].items():
        L.append(f"| {'TOTAL' if k == 'total_s' else k} | {fmt(v)} |")

    enc = r.get("encoder") or {}
    if enc:
        L += ["", "## Encodeur (02)", "",
              f"- {enc.get('gpu_img_per_s', 0):.0f} img/s GPU ({enc.get('dtype')}), "
              f"VRAM pic {enc.get('peak_vram_gb', 0):.2f} GB"]
        if "real_pairs_per_s" in enc:
            L.append(f"- pipeline réel (décodage vidéo, {enc['num_workers']} workers) : "
                     f"{enc['real_pairs_per_s']:.1f} paires/s sur {enc['real_pairs']} paires")

    L += ["", "## Predictor (04) — mini-trains", "",
          "| batch | ms/step | éch/s | s/epoch | total | VRAM pic |",
          "|---|---|---|---|---|---|"]
    for bs, v in r["predictor"]["per_batch_size"].items():
        if v.get("oom"):
            L.append(f"| {bs} | OOM | | | | |")
        else:
            L.append(f"| {bs} | {v['s_per_train_step'] * 1000:.1f} | {v['samples_per_s']:.0f} | "
                     f"{fmt(v['s_per_epoch'])} | {fmt(v['estimate_s'])} | {v['peak_vram_gb']:.1f} GB |")
    L += ["", f"Batch le plus rapide : **{r['predictor']['fastest_batch_size']}**", "",
          "## Fusions (03)", "", "| stratégie | ms/step | total | VRAM pic |", "|---|---|---|---|"]
    for n, v in r["fusion"]["per_strategy"].items():
        L.append(f"| {n} | {v['s_per_step'] * 1000:.1f} | {fmt(v['estimate_s'])} | "
                 f"{v['peak_vram_gb']:.2f} GB |")
    lora = r.get("lora") or {}
    if lora:
        L += ["", "## LoRA (05)", "",
              f"- batch {lora['batch_size']} : {lora['s_per_step'] * 1000:.1f} ms/step, "
              f"{fmt(lora['s_per_epoch'])}/epoch, {fmt(lora['estimate_s'])} au total "
              f"({lora['n_real_pairs']} paires réelles)"]
    L += ["", "## CEM (06)", "", "| n_samples | ms/plan | VRAM pic |", "|---|---|---|"]
    for n, v in r["cem"].items():
        L.append(f"| {n} | {v['latency_ms']:.0f} | {v['peak_vram_gb']:.1f} GB |")

    out = d / "summary.md"
    out.write_text("\n".join(L) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()
