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
│   └── default.yaml           Config centralisée (encoder, dataset, fusion, training)
├── src/
│   ├── encoders.py            Wrapper DINOv3/DINOv2 (frozen)
│   ├── fusion.py              5 stratégies de fusion multi-camera
│   ├── data.py                Loader LeRobot HuggingFace
│   ├── probes.py              ActionProbe + DynamicsProbe + training utils
│   ├── config.py              Loader YAML + utils reproductibilité
│   ├── predictor.py           World model action-conditionné (LoRA-friendly)
│   ├── lora.py                Injection LoRA + merge
│   ├── losses.py              MSE multi-step + cosine + DANN backup
│   └── planner.py             CEM planner + MPC controller
├── scripts/
│   ├── 01_test_dinov3.py            Sanity check DINOv3
│   ├── 02_encode_dataset.py         Pré-encode dataset (sim ou réel)
│   ├── 03_compare_fusion.py         Compare les fusions (M1)
│   ├── 04_train_predictor.py        Train predictor sur Isaac sim (M4)
│   ├── 05_train_lora.py             Train LoRA sur démos réelles (M4)
│   └── 06_inference_demo.py         Démo end-to-end avec CEM planner (M5)
├── results/                   Outputs (encoded data, checkpoints, plots)
├── tests/                     (à venir)
└── DEPLOYMENT.md              Guide pour la machine GPU
```

## Setup

```bash
# Crée un venv
python3 -m venv .venv
source .venv/bin/activate

# Installe les dépendances
pip install -r requirements.txt
```

## Pipeline (Milestone 1 — fusion multi-camera)

```bash
# 1. Vérifier que DINOv3 fonctionne (CPU OK)
python scripts/01_test_dinov3.py

# 2. Encoder tes 40 démos (remplace par ton dataset_id HF)
python scripts/02_encode_dataset.py user/so101_pick_drop_duck --size small

# 3. Comparer les stratégies de fusion
python scripts/03_compare_fusion.py
```

## État actuel

- [x] DINOv3 wrapper (avec fallback DINOv2 non-gated)
- [x] Loader LeRobot HF (compatible lerobot 0.5.2)
- [x] 5 stratégies de fusion implémentées et comparées empiriquement
- [x] Probe d'évaluation (ActionProbe)
- [x] Encodage validé sur duck_dataset_v3 (500 paires, CPU)
- [x] Cross-attention BD gagne (val MAE 8.45 vs ~11-12 pour les autres)
- [x] Predictor world model + LoRA + losses
- [x] CEM planner + MPC controller
- [x] Scripts de training (04, 05) écrits et import-checks OK
- [x] Script d'inférence end-to-end (06)
- [ ] Encodage du dataset complet (à faire sur GPU)
- [ ] Training predictor + LoRA (à faire sur GPU)
- [ ] Setup Isaac Lab (Milestone 2)
- [ ] Génération sim data (Milestone 3)

## Choix techniques

| Composant | Choix | Justification |
|---|---|---|
| Encodeur vision | DINOv3 ViT-S/B/L (figé) | SOTA SSL vision, fort sur 3D |
| World model | DINO-WM style (à coder) | Validé ICML 2025, sample-efficient |
| Sim | Isaac Lab + play data + DR | Génération massive, sim-to-real friendly |
| Adaptation sim-real | LoRA rank=8 | Régularisation par construction |
| Planner | CEM/MPPI | Standard, robuste, sans gradient |

## Notes

- Le code est **CPU-compatible** mais entraînera vite besoin d'un GPU NVIDIA
  pour les étapes lourdes (predictor training, Isaac generation).
- Les milestones 2-5 sont à venir.
