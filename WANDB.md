# Weights & Biases

Le suivi d'expériences est branché dans les trois scripts d'entraînement :

- `scripts/03_compare_fusion.py`
- `scripts/04_train_predictor.py`
- `scripts/05_train_lora.py`

Optionnel : si `wandb` n'est pas installé, le training tourne quand même et
affiche une ligne pour dire que le suivi est inactif.

## Installation

```bash
pip install wandb
wandb login
```

À faire une fois par machine. Le projet `lewm-robot-wm` est créé au premier run.

Alternative sans `wandb login` : exporter `WANDB_API_KEY`.

## Lancer

```bash
python scripts/04_train_predictor.py
```

C'est tout. L'URL du run s'affiche au démarrage et à la fin.

Pour désactiver :

```bash
python scripts/04_train_predictor.py --no-wandb
```

Sur une machine sans réseau, ou si tu n'as pas envie de créer un compte :

```bash
python scripts/04_train_predictor.py --wandb-mode offline
```

Les runs offline sont écrits dans `results/wandb/`. Pour les envoyer plus tard :

```bash
wandb sync results/wandb/offline-run-xxxxx
```

Si aucun credential n'est trouvé, le mode offline est choisi automatiquement,
pour éviter qu'un run lancé sans surveillance se bloque sur le prompt de login.

## Options

Disponibles sur les trois scripts :

```
--no-wandb               désactive le suivi
--wandb-project NOM      projet (défaut : lewm-robot-wm)
--wandb-name NOM         nom du run
--wandb-tags a,b,c       tags
--wandb-group NOM        groupe, pour comparer des runs liés
--wandb-mode MODE        online | offline | disabled
--wandb-log-artifacts    téléverse les checkpoints
--log-every N            fréquence du log intra-epoch (défaut 50)
```

Le projet et le mode peuvent aussi venir de `configs/default.yaml` (section
`wandb:`) ou des variables `WANDB_PROJECT` / `WANDB_MODE`. L'ordre de priorité
est : ligne de commande, puis variables d'environnement, puis le YAML.

## Ce qui est enregistré

Config du run : les arguments de la ligne de commande, le contenu du YAML, le
GPU, la VRAM, la version de CUDA, le commit git.

Pendant l'entraînement :

```
train/loss              train/grad_norm_mean    perf/epoch_seconds
val/loss                train/grad_norm_max     perf/samples_per_s
train/lr                                        perf/gpu_peak_gb
```

Plus `train/batch_loss` et `train/grad_norm` tous les `--log-every` batches, pour
avoir un retour avant la fin de l'epoch.

À la fin : `best_val_loss`, `total_hours`, et pour le LoRA `sim_to_real_gap`
(l'écart entre la val loss sim et la val loss réelle).

`train/grad_norm_mean` monte avant que la loss ne décroche, utile comme alerte.
`perf/samples_per_s` sert à estimer la durée d'un run complet à partir d'un run
court, donc le nombre d'heures de GPU à louer.

## Les checkpoints

Pas téléversés par défaut : ils partent à chaque amélioration de la val loss,
ce qui ralentit l'entraînement sur une connexion lente.

```bash
python scripts/04_train_predictor.py --wandb-log-artifacts
```

## Comparer plusieurs runs

Mettre le même `--wandb-group` sur des runs liés, ils seront regroupés dans l'UI.

`03_compare_fusion.py` le fait tout seul : une stratégie de fusion = un run,
tous dans le groupe `fusion-comparison`.

## Balayer des hyperparamètres

Le plus simple, sans rien installer de plus :

```bash
for lr in 1e-5 3e-5 1e-4 3e-4 1e-3; do
  python scripts/04_train_predictor.py --lr $lr \
    --wandb-group sweep-lr --wandb-name "lr-$lr"
done
```

Les runs sont comparables dans l'UI. Pour un vrai sweep W&B (plusieurs
paramètres croisés, agents en parallèle, early stopping), il faut un YAML de
sweep, qui n'existe pas encore dans le repo.

À savoir : dans `04_train_predictor.py`, seuls `--lr`, `--batch-size`,
`--n-epochs` et `--fusion` sont réglables. `n_layers`, `weight_decay` et le
clipping sont codés en dur. `--rollout-horizon` existe mais n'est pas utilisé,
le training fait une prédiction à un pas.

## Si ça ne marche pas

**`[W&B] Suivi inactif : wandb non installé`**
`pip install wandb`.

**`[W&B] Pas de credentials trouvés — passage en mode offline`**
`wandb login`, ou garde l'offline et synchronise après.

**`[W&B] Suivi inactif : init échouée (...)`**
Réseau ou serveur. L'entraînement continue sans logging, les checkpoints sont
écrits normalement. Passer en `--wandb-mode offline` si ça persiste.

**Des runs apparaissent mais sans métriques**
Vérifier que le training dépasse la première epoch. Les métriques par epoch ne
partent qu'à la fin de chaque epoch.

## Tests

```bash
pytest
```

28 tests sur `src/tracking.py`, sans GPU et sans `wandb` installé. Cas couverts :
paquet absent, credentials manquants, serveur injoignable, exception pendant un
`log()`.
