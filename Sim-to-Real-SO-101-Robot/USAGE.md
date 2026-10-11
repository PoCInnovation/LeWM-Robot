# Collecte de dataset en simulation — SO-101

Extension du workshop NVIDIA Sim-to-Real SO-101 : téléopération clavier,
collecte **automatique** de datasets LeRobot en simulation (Isaac Lab / Isaac Sim),
et système de scènes modulaire.

Le dataset produit est au format **LeRobot v3.0** (mêmes features que les datasets
robot réel : `action` / `observation.state` (6 joints), caméras `wrist` + `front`),
donc directement fusionnable avec un dataset réel.

---

## 1. Setup

Prérequis : environnement conda avec **Isaac Sim 5.1 + Isaac Lab + lerobot**
(ici l'env `leisaac`). Installer le package de ce repo en editable :

```bash
conda activate leisaac
cd <repo>
pip install -e source/sim_to_real_so101 --no-deps
```

Cela installe les commandes CLI : `lerobot_agent`, `lerobot_push_dataset`,
`list_envs`, `random_agent`, `zero_agent`, `lerobot_eval`, `isaac_probe`,
`validate_scenes`.

### Vérifier la machine (à faire une fois)

```bash
python -m sim_to_real_so101.scripts.isaac_probe
```

Écrit `outputs/isaac_probe.json` + un résumé lisible : versions, GPU, encodeurs
ffmpeg, espace disque, cadence de contrôle, butées articulaires, et des
échantillons de cinématique. C'est ce fichier qui permet de développer et tester
le reste du projet sur une machine **sans GPU** (voir [tests/README.md](tests/README.md)).

### Vérifier une scène sans lancer Isaac

```bash
validate_scenes                 # toutes les scènes
validate_scenes cube_to_box     # une seule
```

> Sous Windows, si `conda activate` n'est pas configuré pour le shell, appeler
> directement l'exe : `& <conda>/envs/leisaac/Scripts/lerobot_agent.exe ...`

---

## 2. Lancer

### Tâches disponibles

| Tâche | Contenu |
|---|---|
| `Lerobot-So101-Teleop-Base` | Bras seul, sans décor ni caméras (debug) |
| `Lerobot-So101-Teleop-Task` | Bras + tapis + studio (sol/murs) + 2 caméras (`wrist`, `front`) |
| `Lerobot-So101-Teleop-Vials-To-Rack` | Tâche fioles → rack (workshop d'origine) |

### Trois modes de contrôle

> ⚠️ **`--keyboard` ou `--auto` est obligatoire** (en simulation). Sans l'un des
> deux, le script tente de se connecter à un vrai bras leader USB et échoue
> (`Could not connect on port '/dev/ttyACM0'`).

> ⚠️ Le mode **`--auto` ne fonctionne que pour la scène `cube_to_box`** : la
> policy scriptée est réglée pour saisir l'objet `Cube` et viser `BoxFloor`.
> Pour une autre scène, utiliser `--keyboard`, ou adapter la policy
> (`utils/scripted_policy.py`) aux noms d'objets de cette scène.

```bash
# 1) Téléop clavier (contrôle cartésien de la pince)
lerobot_agent --task Lerobot-So101-Teleop-Task --scene cube_to_box --keyboard

# 2) Policy scriptée automatique (pick-and-place tout seul)
lerobot_agent --task Lerobot-So101-Teleop-Task --scene cube_to_box --auto --num_episodes 5

# 3) Collecte automatique AVEC enregistrement du dataset
lerobot_agent --task Lerobot-So101-Teleop-Task --scene cube_to_box --auto \
  --num_episodes 200 --headless \
  --repo_id <user>/cube_dataset --repo_root ./datasets/cube_dataset \
  --task_name "Pick up the cube and place it in the box" --robot_type so_follower
```

En mode `--auto`, la policy enchaîne les épisodes : succès → épisode enregistré,
échec → jeté, puis reset avec une nouvelle position (cube + boîte randomisés).
Elle s'arrête après `--num_episodes` **succès**. Sans `--keyboard` ni `--auto`,
le script tente de se connecter à un vrai bras leader USB.

---

## 3. Tous les flags

| Flag | Défaut | Rôle |
|---|---|---|
| `--task` | `None` | Nom de la tâche (voir tableau ci-dessus) |
| `--scene` | `None` | Scène custom à injecter (fichier de `scenes/`) |
| `--keyboard` | off | Contrôle clavier au lieu du bras physique (**`--keyboard` ou `--auto` obligatoire en sim**) |
| `--auto` | off | Policy scriptée pick-and-place (**scène `cube_to_box` uniquement**) |
| `--num_episodes` | `None` | (`--auto`) s'arrête après N succès |
| `--repo_id` | `None` | Identifiant du dataset (label / cible du push) |
| `--repo_root` | `None` | Dossier local où écrire le dataset |
| `--task_name` | `None` | Instruction textuelle stockée dans le dataset |
| `--robot_type` | `so101_follower` | `robot_type` des métadonnées (mettre `so_follower` pour matcher un dataset réel) |
| `--seed` | `101` | Graine de l'environnement |
| `--save_mp4` | off | Sauve aussi depth + segmentation en mp4 |
| `--depth` | off | Sauve la depth en mp4 |
| `--instance_id_seg` | off | Sauve la segmentation en mp4 |
| `--num_envs` | `None` | Nombre d'environnements |
| `--port` / `--robot_id` | env vars | Port série / id du bras physique (mode réel) |
| `--disable_fabric` | off | Désactive Fabric (I/O USD) |
| `--headless` | off | Sans fenêtre (plus rapide, idéal collecte en masse) |
| `--device` | `cuda:0` | Device de calcul |

> L'enregistrement s'active dès que `--repo_id`, `--repo_root` et `--task_name`
> sont tous fournis.

---

## 4. Contrôles clavier

La fenêtre Isaac Sim doit avoir le focus. Libellés **AZERTY**.

| Touche | Action |
|---|---|
| `↑` / `↓` | Avancer / reculer la pince |
| `←` / `→` | Gauche / droite |
| `Z` / `S` | Monter / descendre |
| `T` / `G` | Incliner la pince (haut / bas) |
| `Q` / `D` | Rotation de la pince (roll) |
| `A` / `E` | Ouvrir / fermer la pince |
| `H` | Retour automatique à la position de départ |
| `R` | Reset du monde |
| `P` | Démarrer / arrêter l'enregistrement |
| `C` | Annuler l'enregistrement en cours |

---

## 5. Ajouter une nouvelle scène

Une scène = un fichier `.py` dans `source/sim_to_real_so101/scenes/` définissant
un dict `SCENE`. Elle s'utilise ensuite via `--scene <nom_du_fichier>`.

```python
# scenes/ma_scene.py
SCENE = {
    "objects": [
        {
            "name": "Cube",                    # identifiant unique
            "type": "cuboid",                  # cuboid | sphere | cylinder | usd
            "size": (0.025, 0.025, 0.025),     # cuboid : x/y/z (m)
            # "radius": 0.02,                  # sphere / cylinder
            # "height": 0.05,                  # cylinder
            # "usd_path": "asset.usd",         # type usd (relatif au dossier scenes/)
            "color": (0.9, 0.15, 0.15),        # RGB 0-1 (primitives)
            "mass": 0.02,                      # kg
            "static": False,                   # True = décor fixe (murs, support)
            "pos": (0.22, -0.09, 0.06),        # position initiale (repère robot)
            "rot": (1.0, 0.0, 0.0, 0.0),       # quaternion w,x,y,z (optionnel)
            "pos_range": {"x": (-0.06, 0.06), "y": (-0.05, 0.05)},  # randomisation reset
            "group": "box",                    # (optionnel) appartenance à un groupe
        },
    ],
    # Un groupe est déplacé d'un seul bloc (même décalage aléatoire) à chaque reset,
    # pratique pour un objet multi-pièces (une boîte en 5 murs, par ex.).
    "groups": {
        "box": {"pos_range": {"x": (-0.05, 0.05), "y": (-0.03, 0.04)}},
    },
}
```

Repères utiles : robot à l'origine, tapis centré vers `x = 0.22`, surface
`z ≈ 0.035`. Zone atteignable confortable : `x ∈ [0.15, 0.30]`, `y ∈ [-0.15, 0.15]`.
Utiliser la tâche `Teleop-Task` (pas `Base`, qui n'a pas de sol).

Voir aussi `source/sim_to_real_so101/scenes/README.md`.

La policy scriptée `--auto` est réglée pour la scène `cube_to_box` (elle lit
`Cube` comme objet à saisir et `BoxFloor` comme cible). Pour une autre scène,
adapter la policy ou les noms d'objets.

---

## 6. Où sont stockés les datasets

Localement dans le dossier passé à `--repo_root` (par convention `./datasets/<nom>`).
Format LeRobot v3.0 :

Relancer la même commande sur le même `--repo_root` **ajoute** des épisodes au
dataset existant.

## 7. Architecture

```
source/sim_to_real_so101/
├── scripts/
│   ├── lerobot_agent.py          # point d'entrée : téléop / --auto / enregistrement
│   └── lerobot_push_dataset.py   # push d'un dataset vers le Hub
├── tasks/
│   ├── so101_env_cfg.py          # env de base (robot, actions, observations)
│   ├── task_env_cfg.py           # env "Task" : studio (sol/murs), lumière, caméras
│   └── vials_to_rack_env_cfg.py  # tâche fioles → rack (workshop d'origine)
├── utils/
│   ├── keyboard_ee_control.py    # téléop clavier cartésienne (IK)
│   ├── scripted_policy.py        # policy pick-and-place scriptée (--auto)
│   ├── scene_loader.py           # injection des scènes + randomisation de groupe
│   ├── keyboard.py               # touches globales (R / P / C)
│   ├── lerobot_interface.py      # conversions sim <-> réel
│   └── lerobot_recorder.py       # enregistrement asynchrone au format LeRobot
├── scenes/
│   ├── README.md                 # format des scènes
│   └── cube_to_box.py            # scène cube → boîte
├── assets/                       # USD du robot SO-101
└── mdp/                          # termes MDP (observations, resets, randomisation)
```

### Flux en mode `--auto` + enregistrement

1. `scripted_policy` produit les actions frame par frame (approche → prise →
   transport → dépose → retour home), en lisant les positions réelles des objets.
2. `lerobot_agent` applique l'action, enregistre chaque frame (état + action +
   images des 2 caméras) via `lerobot_recorder`.
3. Fin d'épisode → détection de succès (cube dans la boîte) → sauvegarde ou rejet
   → reset (randomisation cube + boîte) → épisode suivant.
4. Après `--num_episodes` succès : flush des sauvegardes en attente puis arrêt.

### Réglages clés de la policy scriptée

En haut de `utils/scripted_policy.py` (valeurs calibrées pour la pince SO-101) :

- `FINGER_LEN = 0.060` — distance corps → bout des doigts
- `GRASP_OFFSET = 0.015` — hauteur de prise au-dessus du cube
- `GRASP_LATERAL = 0.005` — décalage latéral (une seule mâchoire bouge)
- `CARRY_OFFSET = 0.14` — hauteur de transport (dégage les murs de la boîte)

En mode `--auto`, ces trois valeurs sont ajustables **en direct** : `O`/`L`
(hauteur), `I`/`K` (portée), `J`/`H` (latéral), puis `R` pour réessayer.
