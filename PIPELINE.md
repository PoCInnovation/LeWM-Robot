# WM-LeRobot — Architecture DINOv3-WM pour SO-101

Projet de world model action-conditionné pour le robot SO-101, basé sur
DINOv3 (encodeur figé) + predictor custom + CEM planner.

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
│   ├── encoded_data.py        Format encodé v2 : épisodes, stats, vues paires/séquences
│   ├── probes.py              ActionProbe + DynamicsProbe
│   ├── config.py              Loader YAML + reproductibilité
│   ├── predictor.py           World model (résiduel : z+delta, LoRA-friendly)
│   ├── lora.py                Injection LoRA ciblée + merge
│   ├── losses.py              MSE multi-step + cosine + DANN backup
│   └── planner.py             CEM planner (bornes par dim, espace normalisé) + MPC
├── scripts/
│   ├── 00_check_env.py              Validation d'environnement pré-job
│   ├── 01_test_dinov3.py            Sanity check encodeur
│   ├── 02_encode_dataset.py         Pré-encodage par épisode, shardable (job array)
│   ├── 03_compare_fusion.py         Compare les fusions, sauve les poids (M1)
│   ├── 04_train_predictor.py        Train predictor (bf16, resume, multi-step)
│   ├── 05_train_lora.py             LoRA sur démos réelles
│   └── 06_inference_demo.py         Démo CEM (mode --from-encoded 100% offline)
├── slurm/                     Jobs sbatch Adastra + setup offline
├── tests/                     Batterie pytest (30 tests, CPU, sans réseau)
├── results/                   Outputs (encodés, checkpoints — non versionnés)
├── ADASTRA_SETUP.md           Guide Adastra complet (offline, actions requises)
└── DEPLOYMENT.md              Guide machine GPU générique
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# GPU AMD : voir l'en-tête de requirements.txt (index ROCm d'abord)
```

## Pipeline

```bash
# 0. Vérifier l'environnement (obligatoire avant tout job cluster)
python scripts/00_check_env.py

# 1. Sanity check encodeur (CPU OK)
python scripts/01_test_dinov3.py

# 2. Pré-encoder le dataset (une frame = un encodage, sortie par épisode)
python scripts/02_encode_dataset.py --dataset-id divisio74/duck_dataset_v3
#    → results/encoded/duck_dataset_v3_dinov3-small/ (+ meta.json avec stats)
#    Cluster : sharder via --shard-index/--num-shards puis --finalize
#    (cf. slurm/encode_array.sbatch)

# 3. Comparer les fusions (sauve les poids de la gagnante)
python scripts/03_compare_fusion.py --encoded-data results/encoded/<name>

# 4. Entraîner le predictor (fusion GELÉE, actions normalisées,
#    baseline identité loggée, bf16, resume)
python scripts/04_train_predictor.py --encoded-data results/encoded/<name> \
    --fusion cross_attn_bd --fusion-ckpt results/fusion/cross_attn_bd.pt \
    --delta 2            # --horizon 4 pour du multi-step autorégressif

# 5. LoRA sur les démos réelles
python scripts/05_train_lora.py --encoded-data results/encoded/<real> \
    --predictor-ckpt results/checkpoints/predictor_sim/best.pt

# 6. Démo CEM (100% offline sur latents)
python scripts/06_inference_demo.py --from-encoded results/encoded/<name>

# Tests
python -m pytest tests/ -q
```

## Garde-fous scientifiques intégrés

| Garde-fou | Où | Pourquoi |
|---|---|---|
| Fusion gelée pendant le training | 04 | fusion entraînable dans le target MSE → collapse possible avec val loss excellente |
| Actions normalisées (z-score + bornes dataset) | 04/05/06, planner | actions brutes en degrés (±100) ; le CEM échantillonne l'espace normalisé et dénormalise en sortie |
| Baseline identité à chaque val | 04/05 | à delta faible, "copier z_t" est un minimum local fort ; si val/id ≥ 1, rien d'appris |
| Split par épisode | 03/04/05 | le split par paires fuit de l'info (frames voisines quasi identiques) |
| Predictor résiduel (z + delta, head init zéro) | predictor.py | la LayerNorm finale imposait un plancher de MSE incompressible vs des targets non normalisés |
| Multi-step autorégressif (--horizon) | 04 | le CEM déroule 10 steps ; l'entraînement doit voir l'erreur composée |

## État actuel

- [x] Pipeline complet validé end-to-end en local (CPU, offline simulé)
- [x] Encodage shardé + merge, format par épisode fp16
- [x] Adaptation Adastra/offline (slurm/, ADASTRA_SETUP.md)
- [x] 30 tests unitaires/fonctionnels (pytest)
- [ ] Runs GPU réels (encodage complet, training long, ablations)
- [ ] Génération sim (décision MuJoCo-CPU vs Isaac externe à prendre)

## Choix techniques

| Composant | Choix | Justification |
|---|---|---|
| Encodeur vision | DINOv3 ViT-S/B/L/G (figé) | SOTA SSL vision, fort sur 3D ; DINOv2 en fallback public |
| World model | DINO-WM style, prédiction résiduelle | validé ICML 2025, sample-efficient |
| Adaptation sim-real | LoRA rank=8 (attention seulement) | régularisation par construction |
| Planner | CEM/MPPI en espace d'actions normalisé | standard, robuste, sans gradient |
