# WM-LeRobot — Architecture DINOv3-WM pour SO-101

Projet de world model action-conditionné pour le robot SO-101, basé sur
DINOv3 (encodeur figé) + predictor custom + CEM planner.

**Cette branche (`feat/adastra-scale`) : adaptation à la machine Adastra
(MI250X, SLURM, offline) uniquement.** La méthodologie d'entraînement est
celle d'origine, inchangée.

## Vision du projet

Apprendre la **dynamique du bras** (pas une tâche spécifique) pour permettre
au robot de résoudre des tâches **zero-shot** au runtime via planification
dans l'espace latent.

## Structure du repo

```
.
├── requirements.txt
├── configs/
│   └── default.yaml           Config centralisée (encoder, dataset, fusion, probe)
├── src/
│   ├── encoders.py            Wrapper DINOv3/DINOv2 (frozen, offline fail-fast)
│   ├── fusion.py              5 stratégies de fusion multi-camera
│   ├── data.py                Loaders LeRobot HF (frames + paires, local/offline)
│   ├── encoded_data.py        Format encodé v2 par épisode (shardable)
│   ├── probes.py              ActionProbe + DynamicsProbe
│   ├── config.py              Loader YAML + reproductibilité
│   ├── predictor.py           World model action-conditionné (LoRA-friendly)
│   ├── lora.py                Injection LoRA + merge
│   ├── losses.py              MSE multi-step + cosine + DANN backup
│   └── planner.py             CEM planner + MPC controller
├── scripts/
│   ├── 00_check_env.py              Validation d'environnement pré-job
│   ├── 01_test_dinov3.py            Sanity check encodeur
│   ├── 02_encode_dataset.py         Pré-encodage par épisode, shardable (job array)
│   ├── 03_compare_fusion.py         Compare les fusions (M1)
│   ├── 04_train_predictor.py        Train predictor (bf16, resume)
│   ├── 05_train_lora.py             LoRA sur démos réelles (bf16, resume)
│   └── 06_inference_demo.py         Démo CEM (+ mode --from-encoded offline)
├── slurm/                     Jobs sbatch Adastra + setup offline
├── run_all.sh                 COMMANDE UNIQUE : setup + chaîne SLURM complète
├── tests/                     Batterie pytest (25 tests, CPU, sans réseau)
├── results/                   Outputs (non versionnés)
├── ADASTRA_SETUP.md           Guide Adastra complet (offline, actions requises)
└── DEPLOYMENT.md              Guide machine GPU générique
```

## Adaptation machine (ce qu'apporte cette branche)

| Adaptation | Où | Motivation machine |
|---|---|---|
| Mode offline fail-fast | encoders.py, data.py, 00_check_env | pas d'internet sur les nœuds de calcul |
| Dataset par chemin local | data.py | lerobot ne résout pas un repo_id Hub sans réseau |
| Fallback vidéo pyav | data.py | torchcodec exige des libs ffmpeg système souvent absentes |
| Encodage shardé par épisode (fp16) | 02, encoded_data.py | job arrays (1 shard/GCD), RAM bornée, fichiers Lustre-friendly, 1 seul encodage par frame |
| bf16 autocast | 04, 05 | MI250X excelle en bf16 |
| Checkpoint last.pt + --resume | 04, 05 | walltime 24 h + --requeue |
| Éval sur latents (--from-encoded) | 06 | éval offline sans Hub ni encodeur |
| Jobs sbatch + chaîne afterok | slurm/, run_all.sh | binding CINES 8×1 GCD, une étape échouée bloque la suite |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# GPU AMD : voir l'en-tête de requirements.txt (index ROCm d'abord)
```

## Pipeline

```bash
# Sur Adastra : UNE commande fait tout (cf. ADASTRA_SETUP.md)
HF_TOKEN=hf_xxx bash run_all.sh

# En manuel :
python scripts/00_check_env.py                 # validation pré-job
python scripts/01_test_dinov3.py               # sanity encodeur
python scripts/02_encode_dataset.py --dataset-id <id|chemin>   # → results/encoded/<name>/
python scripts/03_compare_fusion.py --encoded-data results/encoded/<name>
python scripts/04_train_predictor.py --encoded-data results/encoded/<name>
python scripts/05_train_lora.py --encoded-data results/encoded/<real> \
    --predictor-ckpt results/checkpoints/predictor_sim/best.pt
python scripts/06_inference_demo.py --from-encoded results/encoded/<name>

# Tests
python -m pytest tests/ -q
```

## État actuel

- [x] Pipeline d'origine adapté à Adastra, validé end-to-end en local
      (CPU, offline simulé, dataset duck réel)
- [x] Encodage shardé + merge, format par épisode fp16
- [x] 25 tests (pytest, CPU, sans réseau)
- [ ] Runs GPU réels sur Adastra
- [ ] Correctifs scientifiques du world model (branche `adastra-test` :
      normalisation des actions pour le CEM, fusion gelée, split par épisode,
      baseline identité, predictor résiduel, multi-step)
- [ ] Génération sim data (décision MuJoCo-CPU vs Isaac externe)

## Choix techniques

| Composant | Choix | Justification |
|---|---|---|
| Encodeur vision | DINOv3 ViT-S/B/L/G (figé) | SOTA SSL vision ; DINOv2 en fallback public |
| World model | DINO-WM style | validé ICML 2025, sample-efficient |
| Adaptation sim-real | LoRA rank=8 | régularisation par construction |
| Planner | CEM/MPPI | standard, robuste, sans gradient |
