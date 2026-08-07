# Setup Adastra (MI250X) — pipeline offline

Guide complet : du nœud de login au premier training. Les nœuds de calcul
n'ont **pas d'accès internet** : tout se pré-télécharge sur le nœud de login,
et les jobs tournent avec `HF_HUB_OFFLINE=1` (posé par `slurm/env.sh`).

## 🚀 COMMANDE UNIQUE (recommandé pour l'opérateur)

Sur le **nœud de login**, depuis la racine du repo :

```bash
HF_TOKEN=hf_xxx bash run_all.sh
```

C'est tout. Le script fait le setup complet (venv ROCm, téléchargement du
modèle et du dataset, validation offline) puis soumet la chaîne SLURM avec
dépendances `afterok` :

```
smoke test (45 min, pipeline miniature)
  └─ afterok → encodage (array 16 GCD)
       └─ afterok → merge + comparaison des fusions
            └─ afterok → training predictor (24 h, resume auto)
```

**Si une étape échoue, les suivantes ne partent jamais** — aucun risque de
brûler l'allocation sur un pipeline cassé. Suivi : `squeue --me` et
`tail -f logs/*.out`.

Variantes utiles :
```bash
# Sans accès DINOv3 approuvé (fallback public, aucun token requis) :
ENCODER_MODEL=facebook/dinov2-base bash run_all.sh

# Voir ce qui serait soumis sans rien soumettre :
DRY_RUN=1 bash run_all.sh

# Relancer la chaîne sans refaire le setup :
SKIP_SETUP=1 bash run_all.sh
```

Le reste de ce document détaille chaque étape pour un usage manuel ou du
débogage.

## ⚠️ ACTIONS UTILISATEUR REQUISES (avant tout)

1. **Accès DINOv3 (gated)** : sur huggingface.co, demander l'accès aux
   modèles `facebook/dinov3-*` (approbation Meta), puis créer un token
   (read) et le garder sous la main pour `huggingface-cli login`.
   *En attendant l'approbation : `encoder.family: "dinov2"` (public) marche
   partout — c'est ce qu'utilise la CI locale.*
2. **Compte Adastra / allocation** : accès eDARI actif, login CINES
   (`ssh <login>@adastra.cines.fr`, mot de passe uniquement — pas de clés).
3. **Choisir l'espace scratch** : repérer ton `$SCRATCHDIR` (Lustre) — tout
   le setup (`WORK_ROOT`) doit y vivre, pas dans le home.
4. **Dataset** : par défaut `divisio74/duck_dataset_v3` (public). Si le
   dataset visé est privé, le token HF doit y donner accès.

## 1. Sur le nœud de login (internet via proxy)

```bash
git clone <repo> && cd LeWM-Robot
huggingface-cli login                      # token HF (cf. actions requises)

export WORK_ROOT=$SCRATCHDIR/lewm-robot
bash slurm/setup_offline.sh \
    divisio74/duck_dataset_v3 \
    facebook/dinov3-vits16-pretrain-lvd1689m
```

Le script : venv + torch ROCm + dépendances → pré-télécharge modèle(s) et
dataset → **valide le tout en mode offline** (`00_check_env.py`). S'il finit
sur "environnement prêt", les jobs passeront.

Si le proxy n'est pas pré-configuré :
```bash
export http_proxy=http://proxy-l-adastra.cines.fr:3128
export https_proxy=$http_proxy
```

## 2. Encodage (job array — 1 shard par GCD)

```bash
export DATASET_DIR=$WORK_ROOT/lerobot_home/divisio74/duck_dataset_v3
sbatch --export=ALL,WORK_ROOT=$WORK_ROOT,DATASET_DIR=$DATASET_DIR \
    --array=0-15 slurm/encode_array.sbatch

# quand l'array est fini (léger, ok sur frontale) :
source slurm/env.sh
python scripts/02_encode_dataset.py \
    --output $WORK_ROOT/encoded/duck_dataset_v3 --finalize
```

Sortie : `$WORK_ROOT/encoded/<dataset>/` — un `ep_XXXXX.pt` par épisode
(fp16) + `meta.json` (stats de normalisation incluses).

## 3. Comparaison des fusions (optionnel, 1 GCD, rapide)

```bash
python scripts/03_compare_fusion.py \
    --encoded-data $WORK_ROOT/encoded/duck_dataset_v3 \
    --output-dir $WORK_ROOT/fusion
# → imprime la stratégie gagnante + sauve ses poids pour 04 --fusion-ckpt
```

## 4. Training predictor (1 GCD, 24 h max, resume automatique)

```bash
sbatch --export=ALL,WORK_ROOT=$WORK_ROOT,\
ENCODED_DIR=$WORK_ROOT/encoded/duck_dataset_v3 \
    slurm/train_predictor.sbatch
```

- `--requeue` + `--resume` : reprend depuis `last.pt` après le mur des 24 h.
- Args par défaut : `--fusion cross_attn_bd --delta 2 --n-epochs 100` ;
  surcharger via `TRAIN_ARGS="..."` dans le `--export`.
- **Surveiller la colonne `identité`** dans les logs : si `val/id >= 1`,
  le modèle ne bat pas "copier z_t" → augmenter `--delta`, `--horizon`.
- Multi-step autorégressif : `TRAIN_ARGS="... --horizon 4"` (recommandé
  avant de faire confiance au CEM, qui déroule 10 steps).

## 5. LoRA sur les démos réelles, puis éval

```bash
sbatch --export=ALL,WORK_ROOT=$WORK_ROOT,\
ENCODED_DIR=$WORK_ROOT/encoded/<demos_reelles>,\
PREDICTOR_CKPT=$WORK_ROOT/checkpoints/predictor_sim/best.pt \
    slurm/train_lora.sbatch

# éval 100% offline sur latents (ni Hub, ni encodeur) :
python scripts/06_inference_demo.py \
    --from-encoded $WORK_ROOT/encoded/duck_dataset_v3 \
    --ckpt $WORK_ROOT/checkpoints/predictor_real/best.pt
```

## Points de vigilance validés en local

| Sujet | Comportement |
|---|---|
| Dataset offline | lerobot ne résout PAS un repo_id Hub en offline même caché → toujours passer un **chemin local** (`DATASET_DIR`) |
| torchcodec | souvent cassé (libs ffmpeg système) → fallback pyav automatique |
| DINOv2 vs v3 | patch 14 vs 16 (256 vs 196 patches) + 4 register tokens en v3 : géré par `encoders.py`, ne rien hardcoder |
| Actions | brutes en degrés (±100) ; training + CEM en espace **normalisé** (stats dans `meta.json` et les checkpoints) |
| QoS | ne jamais passer `--qos` (queues transparentes CINES) |
| bf16 | activé par défaut sur GPU (`--no-bf16` pour désactiver) |

## Batterie de tests avant le run

```bash
python -m pytest tests/ -q                        # 30 tests, CPU, sans réseau
HF_HUB_OFFLINE=1 python scripts/00_check_env.py \
    --dataset-id $DATASET_DIR --expect-offline    # validation environnement
```

Un smoke test complet du pipeline sur 4 épisodes tronqués (~10 min sur
1 GCD) avant le vrai run :

```bash
python scripts/02_encode_dataset.py --dataset-id $DATASET_DIR \
    --output $WORK_ROOT/encoded/_smoke --max-episodes 4 \
    --max-frames-per-episode 40 --num-workers 8
python scripts/04_train_predictor.py --encoded-data $WORK_ROOT/encoded/_smoke \
    --n-epochs 2 --output-dir $WORK_ROOT/checkpoints/_smoke
python scripts/06_inference_demo.py --from-encoded $WORK_ROOT/encoded/_smoke \
    --ckpt $WORK_ROOT/checkpoints/_smoke/best.pt --n-samples 64
```
