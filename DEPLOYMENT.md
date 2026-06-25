# Guide de déploiement (machine GPU)

Workflow : code développé localement sur CPU → push sur Git → clone sur machine
GPU pour les entraînements lourds.

## Sur la machine GPU (RTX/A100/H100)

### Étape 1 — Setup environnement

```bash
git clone <ton-repo-url>
cd WM+deepselfhiddenmodif

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Vérifier que CUDA est dispo
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0))"
```

### Étape 2 — Login HuggingFace (pour le dataset privé éventuellement)

```bash
huggingface-cli login
# Coller ton token HF
```

### Étape 3 — Configurer le dataset

Édite `configs/default.yaml` ou crée `configs/production.yaml` :

```yaml
encoder:
  size: "large"          # Sur GPU on peut prendre large (300M)

dataset:
  hf_id: "ton_user/so101_pick_drop_duck"

probe:
  n_epochs: 100          # Plus d'epochs sur GPU
  batch_size: 64
```

### Étape 4 — Lancer le pipeline

```bash
# Sanity check
python scripts/01_test_dinov3.py

# Encodage du dataset
python scripts/02_encode_dataset.py --config configs/production.yaml

# Comparaison des fusions
python scripts/03_compare_fusion.py --config configs/production.yaml
```

### Étape 5 — Récupérer les résultats

```bash
# Sur la machine GPU :
git add results/
git commit -m "Results from GPU run"
git push

# Localement :
git pull
```

## Bonnes pratiques

### Logs

Lance avec redirection pour garder une trace :

```bash
python scripts/02_encode_dataset.py 2>&1 | tee logs/encoding_$(date +%Y%m%d_%H%M).log
```

### Sessions persistantes

Utilise `tmux` pour ne pas perdre la session si SSH coupe :

```bash
tmux new -s training
python scripts/02_encode_dataset.py
# Ctrl+b puis d pour détacher
# tmux attach -t training pour revenir
```

### Monitoring GPU

```bash
watch -n 1 nvidia-smi
```

## Pour Milestones suivants

Les milestones 2+ (Isaac Lab, predictor training, etc.) nécessiteront :
- Machine avec GPU NVIDIA (Isaac demande CUDA)
- Au moins 8 GB VRAM (16 GB pour batch_size confortable)
- ~50 GB de disque pour le dataset Isaac généré

Si pas d'accès à une machine NVIDIA : RunPod / Vast.ai propose des A100 à
~0.50 €/h, suffisant pour un projet étudiant.
