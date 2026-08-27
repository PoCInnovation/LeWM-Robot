# Déploiement sur une machine RTX 4090 (ou tout GPU NVIDIA)

Cible : 1 × RTX 4090 (24 GB, Ada, sm_89), CUDA 12.x, Linux natif ou WSL2.
Le pipeline est calibré pour cette carte : bf16 + TF32 partout, latents
pré-encodés hébergés en VRAM, batch sizes par défaut dimensionnés pour 24 GB.
Tout reste fonctionnel sur CPU (fp32, plus lent) pour développer.

## 1. Prérequis système

| Quoi | Détail |
|---|---|
| Driver NVIDIA | ≥ 525 (CUDA 12). `nvidia-smi` doit lister la 4090. |
| WSL2 | driver Windows récent suffit — **ne jamais installer de driver dans WSL**. `nvidia-smi` fonctionne dans WSL si le driver Windows est OK. |
| Python | 3.10 – 3.12 (`python3.12` recommandé). |
| ffmpeg | requis pour le décodage vidéo LeRobot (`sudo apt install ffmpeg`). |
| Disque | ~20 GB libres pour les latents d'un dataset + les checkpoints. |
| Accès DINOv3 | modèles gated : demander l'accès à `facebook/dinov3-*` sur huggingface.co puis `huggingface-cli login`. En attendant : `encoder.family: "dinov2"` (public). |

## 2. Installation

Raccourci : `make install` (venv + torch cu128 + dépendances + lerobot), puis
`huggingface-cli login`. Toutes les cibles : `make help`. À la main :

```bash
git clone <repo> && cd LeWM-Robot
git checkout feat/rtx4090

python3.12 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# torch CUDA D'ABORD (sinon pip peut résoudre un build CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements_wm.txt
pip install git+https://github.com/huggingface/lerobot.git   # ou: pip install "lerobot>=0.5"

huggingface-cli login            # token HF (DINOv3 gated)

# Vérifier CUDA + bf16
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
```

## 3. Lancer le pipeline

### Smoke test d'abord (≈ 2-5 min)

Déroule TOUTE la chaîne en miniature (200 paires, 2 epochs) — à faire sur
chaque nouvelle machine avant un vrai run :

```bash
make smoke          # = SMOKE=1 bash run_local.sh
```

### Chaîne complète

```bash
bash run_local.sh                                   # dataset + encodeur de configs/default.yaml
DATASET_ID=/chemin/local/mon_dataset bash run_local.sh
REAL_DATASET=user/demos_reelles bash run_local.sh   # + étape LoRA
TRAIN_ARGS="--n-epochs 100 --batch-size 128" bash run_local.sh
SKIP_ENCODE=1 SKIP_FUSION=1 FUSION=cross_attn_bd bash run_local.sh
```

Chaque étape écrit son log dans `logs/<étape>_<horodatage>.log` et la
chaîne s'arrête à la première erreur. Sorties :

```
results/encoded/encoded_data.pt          latents fp32 (paires t, t+1) + actions + méta
results/fusion_comparison.{json,png}     comparaison des fusions
results/checkpoints/predictor_simu.pt    predictor + fusion (best val)
results/checkpoints/predictor_real.pt    idem avec LoRA
```

### Étape par étape

```bash
python scripts/01_test_dinov3.py                               # débit bf16 vs fp32, cos-sim
python scripts/02_encode_dataset.py --dataset-id divisio74/duck_dataset_v3
python scripts/03_compare_fusion.py
python scripts/04_train_predictor.py --encoded-data results/encoded/encoded_data.pt \
    --fusion cross_attn_bd --n-epochs 100 --batch-size 64
python scripts/02_encode_dataset.py --dataset-id <demos_reelles> \
    --output results/encoded/real_encoded_data.pt
python scripts/05_train_lora.py
python scripts/06_inference_demo.py --n-samples 2000 --rollout-chunk 500
```

## 4. Ce que fait l'adaptation 4090 (et où la régler)

Tout est piloté par la section `hardware` de `configs/default.yaml`
(surchargeable en CLI sur chaque script) :

| Clé | Défaut | Effet |
|---|---|---|
| `tf32` | `true` | matmuls fp32 exécutés en TF32 (tensor cores) : ~2x, précision suffisante |
| `cudnn_benchmark` | `true` | autotune des kernels (shapes fixes) |
| `precision` | `auto` | autocast **bf16** sur la 4090 (fp32 sur CPU). `--precision fp32` pour comparer |
| `data_device` | `auto` | latents pré-encodés chargés **en VRAM** si ≤ 45 % de la VRAM libre, sinon RAM (pinned). `--data-device cpu` pour forcer |
| `num_workers` | `auto` | workers de décodage vidéo à l'encodage (= cœurs − 2, max 8) |
| `compile` | `false` | `torch.compile` du predictor (`--compile`) : +30-80 % après warmup |

Côté encodeur (`encoder.dtype: auto`) : DINOv3 est chargé en **bf16** avec
attention **SDPA** (flash attention), le preprocessing se fait sur le GPU, les
latents ressortent en fp32. `01_test_dinov3.py` affiche la similarité cosine
bf16/fp32 (attendu > 0.99) et le débit.

Côté planner (`06`, `src/planner.py`) : rollouts CEM en autocast bf16 et
découpés en chunks (`--rollout-chunk`) — on peut monter à 2000-5000 candidats
sur 24 GB.

Note sur `encoded_data.pt` : le fichier contient 4 tenseurs de latents fp32
(≈ 4 × N × 196 × 384 × 4 octets ≈ 1.2 GB pour 1 000 paires en DINOv3-small).
Pour un gros dataset, `data_device: auto` les laissera en RAM et transférera
chaque batch ; la VRAM n'est jamais un blocage, la RAM système peut l'être.

## 5. Estimer le temps d'un run complet : `07_benchmark.py`

Avant de lancer un long training, mesure les vrais temps sur TA machine :
le script lance de courts entraînements chronométrés (mêmes modules et même
boucle que 03/04/05/06, 3 batch sizes pour le predictor) et extrapole à la
taille du dataset et au nombre d'epochs visé.

```bash
make bench                     # commande unique (log dans logs/benchmark_*.log)
make bench N_EPOCHS=100        # variables : N_EPOCHS, LORA_EPOCHS, BATCH_SIZES, DATASET_ID, BENCH_ARGS
make bench-quick               # sans dataset ni encodeur (~1 min)

# équivalent direct :
python scripts/07_benchmark.py --n-epochs 100 --dataset-id divisio74/duck_dataset_v3

# avant même d'avoir encodé (latents synthétiques aux bonnes dimensions) :
python scripts/07_benchmark.py --n-pairs 15000 --n-epochs 100 --skip-encoder

# variantes : --batch-sizes 64,128,256  --n-layers 12  --fusion concat_view
#             --lora-epochs 20 --n-real-pairs 4000  --cem-samples 200,1000,5000
```

Sortie : une ligne par étape (`02 encodage`, `03 fusions`, `04 predictor` avec
le batch le plus rapide et le pic VRAM par batch, `05 LoRA`, latence CEM) et
le **TOTAL**, plus `results/benchmark.json`. `--dataset-id` ajoute la mesure
du pipeline d'encodage réel (décodage vidéo inclus — c'est le goulot en
pratique) ; sans lui l'estimation d'encodage est une borne inférieure GPU.
Compter ~1-3 min pour le benchmark lui-même.

## 6. Ordres de grandeur attendus (DINOv3-small)

| Étape | 4090 | Remarque |
|---|---|---|
| Encodage (4 images/paire) | ~500-1500 img/s | goulot = décodage vidéo CPU ; si le GPU n'est pas à 100 % dans `nvidia-smi`, augmenter `--num-workers` |
| Comparaison fusions (4 × 30 epochs) | ~1-3 min | latents en VRAM |
| Predictor 30 epochs, batch 64 | ~5-15 min | `--compile` pour accélérer |
| LoRA 20 epochs | ~1-3 min | |
| CEM (horizon 10, 200 cand., 3 iter) | ~20-50 ms | `--n-samples 2000` reste < 200 ms |

VRAM : DINOv3-small (dim 384, 392 tokens) tient à `--batch-size 256` ;
DINOv3-base (768) à 128 ; DINOv3-large (1024) autour de 48. Le pic VRAM est
loggé à chaque epoch (clé `history` du checkpoint, `peak_vram_gb`).

## 7. Dépannage

| Symptôme | Cause / solution |
|---|---|
| `torch.cuda.is_available() == False` | build torch CPU (réinstaller depuis l'index `cu128`) ou driver absent ; sous WSL2 : mettre à jour le driver **Windows** |
| `CUDA out of memory` | réduire `--batch-size` ; `--data-device cpu` ; `--rollout-chunk` plus petit en 06 |
| GPU à 20-30 % pendant l'encodage | décodage vidéo trop lent : `--num-workers 8`, vérifier ffmpeg |
| latents bf16 trop éloignés de fp32 (01) | `encoder.dtype: float32` (rare) |
| DINOv3 : 401/403 au téléchargement | accès gated non approuvé : `huggingface-cli login` + demande d'accès Meta, ou `encoder.family: dinov2` |
| `attn_implementation=sdpa non supporté` | message informatif : transformers retombe sur l'attention eager (plus lente, même résultat) |

## 8. Bonnes pratiques

- `tmux new -s wm` puis `bash run_local.sh` : la chaîne survit à la fermeture du terminal.
- `watch -n 1 nvidia-smi` dans un second onglet pour vérifier l'utilisation GPU/VRAM.
- Ne pas versionner `results/` (déjà dans `.gitignore`) ; les checkpoints contiennent la config du predictor et de la fusion pour être rejoués ailleurs.
