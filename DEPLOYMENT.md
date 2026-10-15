# Déploiement sur RTX 5090

Cible : une RTX 5090, architecture Blackwell (`sm_120`), 32 Go de VRAM,
sous Linux, WSL2 ou Windows natif via `run_windows.ps1`. Le code conserve son
fonctionnement CPU et RTX 4090.
Les performances sur 5090 doivent être mesurées sur la machine cible.

## Commande unique depuis le dépôt

```bash
make
```

Cette commande installe les dépendances si nécessaire, lit le token dans
`configs/hf_token.txt`, connecte Hugging Face, vérifie le matériel, lance
le benchmark puis produit l’archive du rapport. `make all` et `make bench`
sont équivalents. Le fichier du token est destiné à être versionné selon
le choix du projet ; il est exclu du contenu du rapport et des logs.
Une variable `HF_TOKEN` non vide prend priorité sur ce fichier.

Le benchmark estime la durée des entraînements ; il ne lance pas les
entraînements complets. Ceux-ci restent accessibles avec `make run`.

## Installation

- Python 3.10–3.12, de préférence 3.12 pour les dépendances LeRobot.
- Driver NVIDIA R570 ou plus récent compatible avec la 5090 et CUDA 12.8.
  Sous WSL2, installer le driver côté Windows. Vérifier `nvidia-smi`.
- `ffmpeg` pour les vidéos ; accès Hugging Face approuvé pour DINOv3.
- RAM et disque adaptés au dataset : en format v2 (frames uniques, bf16, cf.
  `src/latents.py`), le dataset complet (17 478 paires + 1 passe augmentée)
  occupe ~16 Go sur disque et en VRAM. L’encodage écrit dans des tables
  préallouées en VRAM quand elles tiennent ; ailleurs, le .pt est lu en mmap.

```bash
make install
source .venv/bin/activate
hf auth login
make check
```

L’installation fixe le couple officiel **torch 2.10.0 / torchvision 0.25.0**
sur l’index CUDA 12.8. Transformers est en 5.x : lerobot >= 0.5 impose
huggingface-hub >= 1.0, que transformers 4.x refuse à l’import.
Les contraintes sont aussi appliquées à l’installation de LeRobot pour éviter
qu’elle remplace ce couple.
LeRobot est figé en 0.5.1, la version validée : `lerobot>=0.5` installait
la 0.6.1 (octobre 2026), jamais testée avec ce code. Un conflit de dépendances doit être résolu avant
le lancement du pipeline.

Installation manuelle équivalente :

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --upgrade --force-reinstall torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements_wm.txt
python -m pip install 'lerobot==0.5.1' -c requirements_wm.txt
python scripts/00_check_gpu.py --require-cuda
```

Blackwell nécessite PyTorch >= 2.7 construit avec CUDA >= 12.8. Les anciennes
roues `cu124` et `cu126` ne conviennent pas à cette carte. Installer un toolkit
CUDA local plus récent ne change pas le runtime embarqué par une roue PyTorch.
Le projet garde volontairement une version précise disponible sur `cu128`.

## Contrôle matériel

`00_check_gpu.py` ne télécharge aucun modèle. Il vérifie la compatibilité du
runtime, puis exécute une fusion cross-attention, un predictor SDPA, le backward
et un pas AdamW (fused sur CUDA), avec la précision configurée.

```bash
python scripts/00_check_gpu.py --require-cuda
make check  # contrôle CUDA puis chargement et vérification de DINO
```

`--require-cuda` fait échouer le contrôle si aucun GPU n’est utilisable. Sans
cette option, le contrôle peut tourner sur CPU. `bench.sh` et `run_local.sh`
l’exécutent avant leur travail ; un benchmark CPU est explicitement identifié.

## Benchmark et réglages

```bash
HF_TOKEN=hf_xxx bash bench.sh
# ou, sans téléchargement du dataset :
NO_DATASET=1 bash bench.sh
# sans encodeur ni dataset, avec latents synthétiques :
make bench-quick
```

Le benchmark essaie **32, 64, 128 et 256** pour le predictor. Chaque essai
libère ses modèles, gradients et optimiseur ; un manque de VRAM est enregistré
comme OOM puis les essais suivants continuent. Les autres étapes ont leurs
propres tailles de batch et peuvent nécessiter une réduction.

```bash
BATCH_SIZES=32,64,128 N_EPOCHS=100 bash bench.sh
BENCH_ARGS='--encoder-batch 32 --lora-batch 16 --cem-samples 64,200' bash bench.sh
```

Résultats : `results/benchmark.json` et
`results/benchmark_report_<date>.tar.gz` (résumé, logs, configuration, versions).
Vérifier si les latents sont réels ou synthétiques et si l’encodage a été mesuré :
un total sans encodage n’est pas une estimation complète. Le benchmark ne mesure
pas la qualité du modèle ni la réussite d’une tâche robotique.

| Réglage | Valeur initiale | Utilisation |
|---|---|---|
| `hardware.power_limit_percent` | `80` | plafond électrique calculé depuis le TGP NVIDIA par défaut |
| `hardware.power_limit_required` | `true` | interdit le calcul GPU si le plafond ne peut pas être appliqué et vérifié |
| `hardware.gpu_index` | `0` | GPU ciblé par PyTorch et `nvidia-smi` |
| `hardware.precision` | `auto` | bf16 sur GPU compatible, fp32 sur CPU |
| `hardware.tf32` | `true` | TF32 pour les opérations fp32 |
| `hardware.data_device` | `auto` | VRAM si latents + `vram_reserve_gb` tiennent dans la mémoire libre, sinon RAM |
| `hardware.vram_reserve_gb` | `5` | VRAM laissée au training à côté des latents (04 en batch 64 : ~4 Go) |
| `hardware.num_workers` | `auto` | min(8, cœurs CPU − 2), à ajuster selon le décodage |
| `hardware.compile` | `false` | opt-in ; première compilation coûteuse, gain à mesurer |
| Predictor `--batch-size` | `64` | augmenter uniquement après benchmark du modèle choisi |
| LoRA `--batch-size` | `32` | ajuster selon la mémoire restante |
| CEM `--rollout-chunk` | `500` dans le pipeline complet | limiter les candidats traités simultanément |

Les 32 Go ne garantissent pas qu’un dataset entier ou un batch de 256 tienne :
la taille de l’encodeur, la fusion et le nombre de couches changent le besoin.
La fusion ne calcule plus de matrices de poids d’attention inutilisées, ce qui
permet à PyTorch d’utiliser SDPA. Le choix du kernel reste géré par PyTorch.

## RTX 4090 louée (24 Go VRAM, 32 Go RAM) — run final

`configs/rtx4090.yaml` hérite de `default.yaml` et ne change que le matériel :
plafond électrique demandé mais non bloquant (un conteneur loué refuse
souvent `nvidia-smi -pl`), réserve VRAM de 5 Go.

```bash
export HF_TOKEN=hf_...           # accès DINOv3 + dataset
export WANDB_API_KEY=...         # optionnel : suivi W&B (cf. WANDB.md)
SMOKE=1 PY_CONFIG=configs/rtx4090.yaml bash run_local.sh   # ~5 min, à faire d’abord
PY_CONFIG=configs/rtx4090.yaml bash run_local.sh
```

Budget mémoire attendu sur le dataset complet :

| Étape | VRAM | RAM |
|---|---|---|
| 02 encodage | ~16 Go de tables + ~2 Go d’encodeur | workers + file d’images uint8 (~4 Go), pic ~8 Go à la sauvegarde |
| 03 / 04 / 05 | ~16 Go de latents + ~4 Go de training | faible : .pt lu en mmap, copié tenseur par tenseur vers le GPU |

À surveiller dans W&B (ou les logs de 04/05) : `val/ratio_vs_identity`,
la val loss divisée par celle de la baseline « rien ne bouge »
(z_t+1 = z_t). Il doit passer nettement sous 1 ; sinon le predictor n’a pas
appris la dynamique, quelle que soit la valeur absolue de la loss. La
validation est faite sur 8 épisodes entiers jamais vus (split par épisode).

Si `[Device] Latents (...) > VRAM libre` s’affiche, les latents restent en
RAM (mmap) : le run continue, plus lentement. Si torchcodec ne trouve pas
ses libs FFmpeg, le décodage bascule automatiquement sur pyav (plus lent) ;
`apt install ffmpeg` le rend inutile.

## Campagne d'entraînement (sans surveillance)

`scripts/campaign.py` enchaîne la campagne décrite dans
`configs/campaign_rtx4090.yaml` : 02 si les latents manquent, 03 une fois
(fusion + poids), criblage de 04 (lr × sortie standard/résiduelle),
taille et régularisation des 2 meilleures configs, confirmation de la
meilleure avec 2 autres seeds (`--seed`), LoRA si `real_data` est posé, puis
réglage du CEM hors ligne sur les épisodes de validation
(`scripts/08_eval_planner.py`).

```bash
export WANDB_API_KEY=...
python scripts/campaign.py --dry-run      # liste des runs et epochs par taille de modèle
nohup python scripts/campaign.py > logs/campaign.log 2>&1 &
```

- Au démarrage, la campagne vérifie la clé W&B (`WANDB_API_KEY` ou `wandb
  login`) et s'arrête si elle manque ou est refusée : sans elle, chaque run
  passerait en offline sans rien afficher en ligne. Elle affiche l'adresse du
  projet ; les runs sont groupés par étape (`c1-A_screen`, `c1-B_regul`...).
  `WANDB_MODE=offline` pour lancer volontairement sans suivi en ligne.
- Relancer la même commande après une coupure : runs terminés sautés, run
  04 en cours repris à son dernier epoch (`<run>/predictor.last.pt`), même
  run W&B (id déterministe). `name:` dans le fichier = nouvelle campagne.
- Sorties : `results/campaign/<name>/` — un dossier par run (log, checkpoint,
  `*.done.json`), `best_predictor.pt` (val loss la plus basse parmi les runs
  à ratio vs identité < 1), `best.json` (meilleurs predictor / LoRA / CEM,
  dispersion entre seeds), `state.json`. La ligne `[Campagne] écart avec la
  2ᵉ config ... → classement fiable / équivalentes` dit si le choix de la
  meilleure config dépasse le bruit entre seeds.
- Durée : Σ (epochs × s/epoch) + ~1 min de chargement par run. Le s/epoch
  se lit sur la ligne de fin d'epoch de 04 (`[NNs, ... éch/s, VRAM pic ...]`,
  ou `perf/epoch_seconds` dans W&B) ; la campagne l'affiche et projette le
  temps restant de l'étape avant chaque run.

## Pipeline

```bash
make smoke
make run
REAL_DATASET=user/demos_reelles bash run_local.sh
TRAIN_ARGS='--n-epochs 100 --batch-size 128' bash run_local.sh
SKIP_ENCODE=1 SKIP_FUSION=1 FUSION=cross_attn_bd bash run_local.sh
```

Le script utilise `.venv/bin/python` s’il existe, sinon `python3`.
`PYTHON=/chemin/python` permet de choisir un autre environnement.
Le mode smoke réduit les données et le modèle. LoRA n’est exécuté que si
`REAL_DATASET` est fourni.

Sous Windows natif, ouvrir PowerShell **en administrateur** :

```powershell
# Validation courte recommandée avant le run complet
powershell -ExecutionPolicy Bypass -File .\run_windows.ps1 -Smoke

# Pipeline complet
powershell -ExecutionPolicy Bypass -File .\run_windows.ps1
```

Le lanceur Windows crée `.venv\Scripts\python.exe`, installe les dépendances et
exécute les mêmes étapes que `run_local.sh`. À chaque processus GPU, le plafond
est appliqué avant les kernels lourds, vérifié, puis l'ancienne valeur est
restaurée à la sortie. Une RTX 5090 FE de 575 W est ainsi plafonnée à 460 W ; le
calcul utilise le TGP réel annoncé par `nvidia-smi`, donc les modèles partenaires
sont traités selon leur propre limite par défaut.

Sorties :

- `results/encoded/encoded_data.pt` : latents, actions et métadonnées.
- `results/fusion_comparison.{json,png}` : comparaison des fusions.
- `results/checkpoints/predictor_simu.pt` : predictor et fusion.
- `results/checkpoints/predictor_real.pt` : predictor adapté avec LoRA.
- `logs/` : journaux par étape.

## Tests et dépannage

```bash
python -m pip install pytest
python -m pytest tests/ -q
```

Les tests simulent les versions Blackwell sur CPU et vérifient l’équivalence
de la fusion SDPA, les rollouts et LoRA. Les tests CUDA nécessitent un GPU.
Ils ne remplacent pas un smoke test sur une vraie 5090 avec le dataset cible.

| Symptôme | Action |
|---|---|
| CUDA indisponible | vérifier `nvidia-smi`, le driver et l’environnement Python sélectionné |
| limite GPU impossible | lancer PowerShell en administrateur ; vérifier que le modèle et le pilote autorisent `nvidia-smi -pl` |
| `no kernel image` / `sm_120` incompatible | réinstaller le couple CUDA 12.8 ci-dessus, relancer `00_check_gpu.py` |
| OOM | réduire batch/chunk ou choisir `hardware.data_device: cpu` |
| GPU peu occupé à l’encodage | mesurer et ajuster les workers ; vérifier CPU, stockage et décodage vidéo |
| DINOv3 401/403 | vérifier accès au modèle et authentification ; DINOv2 est une alternative configurable |

Références : [versions officielles PyTorch](https://pytorch.org/get-started/previous-versions/),
[support Blackwell depuis PyTorch 2.7](https://pytorch.org/blog/pytorch-2-7/),
[spécifications RTX 5090](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/),
[DINOv3 dans Transformers 4.56](https://huggingface.co/docs/transformers/v4.56.0/en/model_doc/dinov3).
