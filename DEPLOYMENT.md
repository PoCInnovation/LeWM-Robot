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
- RAM et disque adaptés au dataset : les quatre tenseurs fp32 occupent environ
  1,2 Go pour 1 000 paires DINOv3-small. L’encodage accumule les latents en RAM
  puis les concatène : prévoir aussi la mémoire temporaire.

```bash
make install
source .venv/bin/activate
hf auth login
make check
```

L’installation fixe le couple officiel **torch 2.10.0 / torchvision 0.25.0**
sur l’index CUDA 12.8. Transformers est limité à `>=4.56,<5` pour DINOv3.
Les contraintes sont aussi appliquées à l’installation de LeRobot pour éviter
qu’elle remplace ce couple. Un conflit de dépendances doit être résolu avant
le lancement du pipeline.

Installation manuelle équivalente :

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --upgrade --force-reinstall torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements_wm.txt
python -m pip install 'lerobot>=0.5' -c requirements_wm.txt
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
| `hardware.data_device` | `auto` | VRAM si les latents tiennent dans 45 % de la mémoire libre, sinon RAM |
| `hardware.num_workers` | `auto` | min(8, cœurs CPU − 2), à ajuster selon le décodage |
| `hardware.compile` | `false` | opt-in ; première compilation coûteuse, gain à mesurer |
| Predictor `--batch-size` | `64` | augmenter uniquement après benchmark du modèle choisi |
| LoRA `--batch-size` | `32` | ajuster selon la mémoire restante |
| CEM `--rollout-chunk` | `500` dans le pipeline complet | limiter les candidats traités simultanément |

Les 32 Go ne garantissent pas qu’un dataset entier ou un batch de 256 tienne :
la taille de l’encodeur, la fusion et le nombre de couches changent le besoin.
La fusion ne calcule plus de matrices de poids d’attention inutilisées, ce qui
permet à PyTorch d’utiliser SDPA. Le choix du kernel reste géré par PyTorch.

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
