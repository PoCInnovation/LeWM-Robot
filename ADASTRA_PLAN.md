# Plan d'adaptation Adastra — exploiter le supercalculateur pour le world model

## Contexte

- **Machine cible** : Adastra (CINES) — partition MI250X.
  - 356 nœuds × 4 cartes AMD Instinct MI250X = 8 GCD par nœud = **2848 GPU logiques** (gfx90a).
  - 64 Go HBM2e **par GCD** (8× le minimum requis par `DEPLOYMENT.md`).
  - Walltime standard : **24 h max** par job.
  - Ordonnanceur : SLURM. Pas d'accès internet sur les nœuds de calcul.
- **Code actuel** : pipeline DINOv3-WM (`src/` + `scripts/01-06`), mono-GPU, pensé
  pour un NVIDIA 8-16 Go. Predictor ~43 M params.

### Pourquoi ça tourne (presque) tel quel sur AMD

Sur les builds **ROCm** de PyTorch, l'API HIP se présente comme CUDA :
`torch.cuda.is_available()` renvoie `True`, `device="cuda"` alloue sur le GPU AMD.
Tous les `"cuda" if torch.cuda.is_available() else "cpu"` du code fonctionnent donc
sans modification. Le projet n'utilise aucune extension CUDA custom (PyTorch pur),
et `scaled_dot_product_attention` est supporté sur gfx90a.

Installation : build ROCm de torch (`--index-url https://download.pytorch.org/whl/rocm6.x`)
ou, de préférence, les modules/containers PyTorch fournis par le CINES.

### Principe directeur

Le modèle est petit : le gain ne viendra **pas** de paralléliser *un* entraînement
sur 2848 GCD, mais de :
1. transformer chaque étape du pipeline en travail massivement parallèle (job arrays) ;
2. dépenser plus de compute **par step** (multi-step, modèles plus gros, ensembles) ;
3. répondre à des questions impossibles à se payer sur une seule carte
   (ablations exhaustives, scaling study).

---

## Levier 1 — Encodage shardé : le goulot d'étranglement devient gratuit

*Job array, ~50-200 GCD. L'adaptation la plus rentable et la plus simple.*

- Ajouter `--shard-index i --num-shards N` à `scripts/02_encode_dataset.py` :
  chaque job encode une tranche d'**épisodes** et écrit `encoded_shard_i.pt`,
  plus un script de merge.
- Lancé en job array SLURM (8 tâches/nœud, 1 GCD chacune), l'encodage complet
  passe de ~2 h CPU à quelques minutes.
- Conséquences stratégiques :
  - On peut se permettre **DINOv3-giant (1.1B)** au lieu de small — 64 Go de HBM
    par GCD l'avalent sans problème.
  - On peut encoder **toutes les variantes augmentées** produites par
    `data_augmentation/augment.py` : le pipeline d'augmentation devient un
    multiplicateur de dataset ×8 réellement exploitable.
- **Prérequis structurant** : sortie par épisode (séquences contiguës, pas des
  paires en vrac) — indispensable pour le multi-step (Levier 3).

## Levier 2 — Entraînement : DDP mono-nœud + bf16, et grossir le modèle

- Ajouter `torchrun` + DistributedDataParallel aux scripts 04/05. Sur ROCm, la
  communication passe par RCCL mais l'API PyTorch est identique
  (`backend="nccl"`). **Un nœud = 8 GCD = l'unité de travail naturelle** pour un
  entraînement.
- Ajouter l'autocast **bf16** (excellent sur MI250X) — quelques lignes.
- Réalisme : à 43 M de paramètres, 8 GCD s'ennuient en pur data-parallel. La
  bonne dépense du compute est *par step* (Levier 3). La machine permet aussi de
  tester des predictors **10-20× plus gros** (12-24 couches, dim 1024-1536) pour
  vérifier si la capacité était le facteur limitant.

## Levier 3 — Multi-step training : le changement le plus important scientifiquement

Le modèle est actuellement entraîné **one-step** mais le CEM le déroule sur
**10 steps** en autorégressif : l'erreur composée n'est jamais vue à
l'entraînement. Le multi-step autorégressif (`multi_step_combined`, déjà écrit
dans `src/losses.py`) coûte T forwards + backprop à travers le rollout —
exactement le genre de coût qu'Adastra rend indolore.

À faire :
- Encodage par séquences (cf. Levier 1).
- Un `SequenceDataset` qui échantillonne des fenêtres de longueur H.
- Un curriculum H = 1 → 8 au fil des epochs.
- Métrique de contrôle systématique : **loss de la baseline identité**
  (copier `z_t`) pour détecter le collapse "modèle identité".

C'est ce qui rendra la planification CEM réellement fiable.

## Levier 4 — Sweeps et ablations exhaustifs (job arrays, 100-500 GCD)

Chaque entraînement tient sur 1 GCD en ~minutes. Au lieu de choisir des
hyperparamètres, balayer le produit cartésien complet en une soumission :

| Dimension | Valeurs à balayer | Question tranchée |
|---|---|---|
| Fusion | les 5 stratégies | `concat_view` vs `cross_attn_bd` (débat M1) |
| `delta_timesteps` / stride | 1, 2, 4, 8 | risque "modèle identité" |
| Horizon multi-step H | 1, 2, 4, 8 | fiabilité du rollout CEM |
| Taille encodeur | small / base / large / giant | qualité du latent |
| Taille predictor | 3 configs | capacité limitante ? |
| Seeds | 3-5 par cellule | variance |

Harnais : un YAML par cellule généré par script, `sbatch --array=0-999`,
résultats en JSON agrégés + baseline identité incluse partout.
En 24 h : une **étude d'ablation complète** au lieu d'une intuition.

## Levier 5 — Ensembles de world models → planification consciente de l'incertitude

- Entraîner N = 8-16 predictors (seeds / sous-ensembles de données différents) —
  parallélisme trivial, un membre par GCD.
- Modifier le CEM pour scorer avec l'ensemble : coût = distance au but +
  **pénalité de désaccord entre modèles** (style PETS).
- Combat directement l'exploitation par le planner des zones où le modèle
  hallucine — la faiblesse classique des world models.
- L'inférence d'ensemble (8-16 × 43 M) tient sans effort dans les 64 Go d'un GCD.

## Levier 6 — CEM : exploiter la VRAM

- `n_samples=200, n_iter=3` était calibré pour une petite carte. Sur un GCD :
  **2000-5000 échantillons**, plus d'itérations, horizon plus long. Le rollout
  est déjà batché dans `src/planner.py` → changement de config, pas de code.
- Éval offline massive : rejouer le planner sur des centaines de paires
  (état, but) tirées du dataset et mesurer le taux de succès en latent,
  en job array.

## Levier 7 — Les données sim sans Isaac

**Isaac Lab ne tournera jamais sur Adastra** (verrouillé NVIDIA/CUDA). Les
Milestones 2-3 (génération des trajectoires sim) doivent être repensés.
Deux options :

1. **MuJoCo sur les CPU d'Adastra** (partition CPU ou cœurs Trento des nœuds
   GPU) : MuJoCo est CPU-first et se parallélise par processus — des milliers de
   trajectoires SO-101 en job array CPU, pendant que les GCD encodent/entraînent.
2. **Isaac sur une machine NVIDIA externe** ponctuelle (une carte modeste
   suffit), Adastra pour tout le reste.

→ Décision de projet à prendre avant d'investir dans le setup.

---

## Contraintes pratiques (à intégrer dès le départ)

- **Checkpoint/resume systématique** : les jobs du sweep s'en fichent, mais tout
  run long doit survivre au mur des 24 h — `sbatch --requeue` + reprise depuis le
  dernier checkpoint.
- **Mode offline HF** : pré-téléchargement des modèles/datasets sur le nœud de
  login (`huggingface-cli download`), `HF_HUB_OFFLINE=1` dans les jobs,
  `HF_HOME` sur le scratch. **Token HF requis : DINOv3 est gated.**
- **Lustre-friendly** : shards de bonne taille (pas de milliers de petits
  fichiers), données chaudes copiées en local nœud en début de job.
- **Binding** : 8 tâches/nœud, 1 GCD (`--gpus-per-task=1`) et ~8 cœurs CPU par
  tâche.
- **Pas de support MPS/Metal à prévoir** : le fallback CPU reste le mode dev local.

## Ordre d'implémentation proposé

1. **Sharding de l'encodage + sortie par séquences** (débloque tout le reste).
2. **bf16 + checkpoint/resume + scripts sbatch de base** (job simple 1 GCD).
3. **Multi-step training** (SequenceDataset + curriculum + baseline identité).
4. **Harnais de sweep** + l'ablation du tableau ci-dessus.
5. **Ensembles + CEM à incertitude.**
6. (Décision projet) **MuJoCo-CPU vs Isaac externe.**

## Rappel — correctifs au world model à faire avant tout gros run

Identifiés lors de l'analyse du code (silencieux dans les courbes de loss) :

1. **Geler la fusion dans `scripts/04_train_predictor.py`** (ou a minima
   `detach()` le target `z_t1`) : la fusion entraînable est dans le chemin du
   target de la MSE → risque de collapse de représentation avec un val loss
   excellent. Choisir la fusion en M1, la geler en 04 comme elle l'est déjà en 05.
2. **`delta_timesteps` > 1** (ou stride) + baseline identité dans les logs de
   val : à delta=1, "copier l'entrée" est un minimum local très fort.
3. **Injecter la proprioception dans le predictor** (token supplémentaire,
   ~10 lignes) : un latent d'image seule n'encode pas la vitesse des joints ;
   le proprio est déjà chargé et sauvegardé par le script 02 mais jamais consommé.
4. Mineurs : clamper la moyenne CEM aux bornes d'action ; `ffn_dim` par défaut
   incohérent avec son commentaire (2048 ≠ 4×768) ; `merge_lora` jamais appelé
   (inférence ~30 % plus lente que nécessaire).
