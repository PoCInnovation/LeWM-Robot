# Simulation SO-101 — état & journal de travail

Document de suivi **session par session** de la partie simulation (Isaac Sim /
Isaac Lab) du projet LeWM-Robot.

- **Ce fichier est la source de vérité** sur ce qui existe et ce qui reste à faire.
- À chaque feature ajoutée : mettre à jour la section [État actuel](#2-état-actuel)
  **et** ajouter une entrée dans le [Journal](#5-journal-des-sessions).
- **Chantier en cours : [ROADMAP_DATASET.md](ROADMAP_DATASET.md)** — dataset de 5000 épisodes.
- Doc utilisateur (comment lancer, flags, touches) : [USAGE.md](USAGE.md).
- Format des scènes : [source/sim_to_real_so101/scenes/README.md](source/sim_to_real_so101/scenes/README.md).

Dernière mise à jour : **2026-08-04** — branche `leisaac`.

---

## 1. Contexte

Le but de la simulation n'est **pas** d'entraîner une policy de tâche, mais de
produire en masse des données de **dynamique du bras** pour le world model
DINOv3-WM (voir [PIPELINE.md](../PIPELINE.md) à la racine). Le dataset sim doit
donc être :

1. au **format LeRobot v3.0 identique au dataset réel** (mêmes features, mêmes
   noms de caméras `wrist` / `front`, même `robot_type`) pour être fusionnable ;
2. **divers** (scènes, objets, éclairages, points de vue) pour que l'encodeur
   figé + le predictor apprennent la dynamique et pas la scène ;
3. **riche en dynamique** — toutes les dimensions d'action doivent bouger.

Base de code : workshop NVIDIA `Sim-to-Real-SO-101-Workshop`, étendu maison
(système de scènes, téléop clavier cartésienne, policy scriptée, collecte auto).

---

## 2. État actuel

### 2.0 Machine de collecte (mesuré le 2026-08-06)

Relevé par `scripts/isaac_probe.py` sur la machine du binôme.

| | |
|---|---|
| OS | Windows 11 (26200), lancé via Git Bash |
| CPU / RAM | AMD Ryzen 7 5800X (8c/16t) — 32 Go |
| GPU | **RTX 3080, 10 Go VRAM**, driver 580.88, CUDA 12.8 |
| Disque libre | **134,5 Go** — largement suffisant (8-13 Go attendus) |
| ffmpeg | 9.0 — **libsvtav1, libaom-av1, librav1e, libx264, libx265** tous présents |
| Isaac Sim / Isaac Lab | 5.1.0.0 / 0.54.2 (isaaclab_tasks 0.11.12) |
| lerobot / torch / numpy | 0.4.3 / 2.7.0+cu128 / 1.26.0 |
| Python | 3.11.9 |

⚠️ La RTX 3080 (Ampere) est **hors des GPU testés par le workshop** (RTX 6000 Pro
Blackwell, 5090, RTX 6000 Ada). 10 Go de VRAM devraient suffire pour 1 env avec
2 caméras 480×640, mais c'est à surveiller au smoke test.

⚠️ **lerobot 0.4.3** — à confirmer : écrit-il bien du LeRobot **v3.0** ?
`datasets/cube_dataset` est en v3.0, donc probablement oui, mais le dataset réel
cible est aussi en v3.0 et un décalage de format casserait la fusion.

**Confirmé par la sonde :** `control_hz = 30.0` (decimation 4 × dt 1/120,
render_interval 4) — la décision D2 est bien appliquée sur la vraie machine.
Les 6 tâches gym sont enregistrées. Les 2 caméras sont bien en 480×640,
focale 13,5 mm.

### 2.1 Stack

| Élément | Valeur |
|---|---|
| Simulateur | Isaac Sim 5.1 + Isaac Lab |
| Env conda | `leisaac` |
| Package | `sim_to_real_so101` (editable : `pip install -e source/sim_to_real_so101 --no-deps`) |
| Entrées CLI | `lerobot_agent`, `lerobot_eval`, `lerobot_push_dataset`, `list_envs`, `random_agent`, `zero_agent` |

### 2.2 Robot

- USD `assets/usd/SO-ARM101-USD.usd`, `fix_root_link=True`, base à `(-0.05, 0, 0)` yaw 90°.
- 6 joints : `Rotation`, `Pitch`, `Elbow`, `Wrist_Pitch`, `Wrist_Roll`, `Jaw`.
- 7 corps, dans cet ordre (**mesuré**) : `base`, `shoulder`, `upper_arm`,
  `lower_arm`, `wrist`, `gripper`, `jaw`. Le corps `wrist` **existe** — donc
  l'extrapolation `FINGER_LEN` de la policy scriptée est bien active (c'était
  une inconnue : son absence l'aurait désactivée sans le moindre warning).
- ⚠️ `find_bodies()` traite son argument comme une **regex et lève** quand rien
  ne correspond — il ne renvoie pas de liste vide. Tout code qui teste des noms
  de corps optionnels doit filtrer sur `robot.data.body_names`.
- Actionneurs implicites — stiffness `55 / 30 / 25 / 12 / 7 / 4`,
  damping `0.7 / 0.8 / 0.7 / 0.5 / 0.5 / 0.3`, `effort_limit_sim=30` partout.
- Action = `JointPositionActionCfg`, `scale=1`, `use_default_offset=False`
  → **cibles articulaires absolues en radians**.
- Pose home : `(-0.2736, -0.6109, -0.0745, 1.5148, -1.6034, -0.1465)`.
- Variantes : `S0101_NO_CAMERA_CFG`, `S0101_CONTACT_GRASP_CFG` (capteurs de contact).

### 2.3 Tâches enregistrées (`tasks/__init__.py`)

| Gym ID | Config | Contenu |
|---|---|---|
| `Lerobot-So101-Teleop-Base` | `SO101TeleopEnvCfg` | Bras seul, **pas de sol ni caméras** (debug) |
| `Lerobot-So101-Teleop-Task` | `SO101TaskEnvCfg` | **Env de travail** : sol + murs + dome light + tapis + 2 caméras |
| `Lerobot-So101-Teleop-Vials-To-Rack` | `VialsToRackEnvCfg` | Tâche fioles → rack (workshop, non utilisée ici) |
| `…-Vials-To-Rack-DR` | `VialsToRackDREnvCfg` | + HDRI, couleur robot, tapis |
| `…-Vials-To-Rack-Eval` / `-DR-Eval` | `VialsToRackEval*EnvCfg` | + terminaisons succès/timeout |

### 2.4 Paramètres de simulation

| Paramètre | Valeur | Note |
|---|---|---|
| `sim.dt` | `1/120` | pas physique (inchangé) |
| `decimation` | **`4`** | → **1 step de contrôle = 1/30 s (30 Hz)**, aligné sur le `fps=30` du dataset et sur le robot réel (D2, phase 2) |
| `episode_length_s` | `5` | pas de terminaison sur `Teleop-Task` (`terminations=None`) |
| `num_envs` | `1` (forcé dans `__post_init__`) | pas de collecte parallèle |
| `render.rendering_mode` | `"quality"` | |
| `render.enable_translucency` | `False` | |

### 2.5 Observations

- Groupe `policy` (`concatenate_terms=False`, corruption activée) :
  `joint_pos_obs`, `joint_pos_rel`, `ee_frame_state` (pos + quat du gripper dans
  le repère base, via `FrameTransformer` base → gripper).
- Groupe `visual` (corruption désactivée) : `rgb_ego`, `rgb_external_D455`.

### 2.6 Caméras

| Nom scène | Nom dataset | Montage | Optique |
|---|---|---|---|
| `camera_ego` | `wrist` | `Robot/gripper/gripper_cam`, offset `(-0.005, 0.06, -0.062)`, euler `(-45, 0, 0)` | pinhole, f=13.5 mm, f_stop=100, focus 0.05 m |
| `camera_external_D455` | `front` | `external_cam`, eye `(0.22, -0.50, 0.18)` → target `(0.22, 0, 0.07)`, quaternion look-at OpenGL | pinhole, f=13.5 mm, focus = distance eye↔target |

Résolution : **480 × 640**, `data_types=["rgb"]`. Le renommage `ego→wrist` /
`external_D455→front` se fait dans `lerobot_agent.py` (`CAMERA_RENAME`).

### 2.7 Décor `Teleop-Task`

- `Floor` : cuboid 6×6×0.1 m, kinematic, gris `(0.75, 0.75, 0.78)`, top à `z ≈ 0.0257`.
- `WallBack` (x=-0.35) et `WallSide` (y=0.45), cuboids 3 m de haut, gris clair.
- `RoomLight` : `DomeLight` intensité 2000, blanc.
- `Mat` : `assets/usd/mat.usda` à `(0.22, 0, 0.032)`, yaw 90°.
- `sky_light` HDRI : **présent mais commenté** (24 HDRI `.exr` dispo dans `assets/hdri/`).

### 2.8 Domain randomization active sur `Teleop-Task`

| Terme | Plage |
|---|---|
| `reset_lightbox_light_exposure` | exposure dome light `(-3.0, 1.0)` |
| `reset_mat_rotation` | yaw tapis `±0.1 rad` (~±6°) |
| `reset_camera_ego_fov` | focale ego `12 → 15 mm` |
| `reset_camera_external_pose` | pos `±2 cm` (x,y), `±1 cm` (z) ; rot `±0.05 rad` |
| `reset_set_robot_visual_material` | **verrouillé sur `orange`** (palette dispo : orange / teal / white / black) |
| `reset_robot_position` | offset `(0, 0)` → pose home déterministe |

Non randomisés aujourd'hui : HDRI/skybox, textures et couleurs du sol/murs/tapis,
couleur et taille des objets de scène, masse/friction, bruit sur les actions.

### 2.9 Système de scènes (maison)

`utils/scene_loader.py` — injecte des objets dans **n'importe quelle** tâche
avant `gym.make`, via `--scene <nom>` :

- `load_scene_spec()` importe `scenes/<nom>.py` et lit son dict `SCENE`.
- `_spawn_cfg()` construit le spawner : `cuboid` | `sphere` | `cylinder` | `usd`.
- Par objet : `name`, `size`/`radius`/`height`, `color`, `mass`, `static`
  (kinematic), `pos`, `rot`, `pos_range` (→ `reset_root_state_uniform`), `group`.
- `groups` + `randomize_group_offset()` : un décalage aléatoire **commun** à
  plusieurs objets (utile pour une boîte en plusieurs murs).
- Un `reset_scene_to_default` est ajouté en tête des events pour que la
  randomisation reparte de l'état initial.

**Scènes existantes : 1** — `cube_to_box`
(cube rouge 2.5 cm / 20 g, `pos_range` ±6 cm en x, ±5 cm en y ;
boîte marron 10×10 cm faite de 5 cuboids statiques, groupe décalé de ±5/±3-4 cm).

### 2.10 Contrôle

| Module | Rôle |
|---|---|
| `utils/keyboard.py` | Touches globales `R` (reset) / `P` (start-stop rec) / `C` (annuler) |
| `utils/keyboard_ee_control.py` | Téléop **cartésienne** : IK damped-least-squares sur `(Rotation, Pitch, Elbow)`, leash 6 cm, servo `Wrist_Pitch` qui maintient l'inclinaison de la pince, clavier AZERTY |
| `utils/keyboard_arm_control.py` | Téléop articulaire (legacy) |
| `utils/scripted_policy.py` | **Policy scriptée pick-and-place** (`--auto`) |

`ScriptedPickPlace` — machine à états 9 étapes :
`open_gripper → move_above_pick → descend → grasp → lift → move_above_place →
lower_place → release → home`.

- Même IK que le clavier ; le **point de contrôle** est le corps `jaw` poussé de
  `FINGER_LEN` le long de l'axe poignet→gripper, plus un décalage latéral
  `GRASP_LATERAL` (une seule mâchoire bouge sur le SO-101).
- Cibles lues **directement dans la sim** (`root_pos_w` du cube et de `BoxFloor`),
  donc la policy suit la randomisation de reset.
- Constantes calibrées : `FINGER_LEN=0.060`, `GRASP_OFFSET=0.015`,
  `GRASP_LATERAL=0.005`, `APPROACH_OFFSET=0.08`, `CARRY_OFFSET=0.14`,
  `PLACE_DROP_OFFSET=0.04`, `Z_MIN=0.03`, `JAW_OPEN=0.6`, `JAW_CLOSED=-0.35`,
  `GRASP_TILT=1.55`.
- Tuning **à chaud** : `O`/`L` hauteur de prise, `I`/`K` portée doigts,
  `J`/`H` latéral, puis `R` pour réessayer.
- ⚠️ **Les offsets ci-dessus ne sont valides qu'à inclinaison de pince
  constante.** Ils ont été calibrés à `GRASP_TILT = 1.55` et fonctionnent bien
  pour le pick & place du cube tant que le servo de poignet tient cette
  inclinaison. `GRASP_OFFSET` est une hauteur **verticale monde** et
  `GRASP_LATERAL` un décalage **horizontal** : les deux dépendent de
  l'orientation de la pince. Dès qu'on incline, le point de préhension calculé
  ne correspond plus aux vrais doigts. C'est bloquant pour les modes A, C et E,
  qui veulent faire tourner la pince → il faut un point de contrôle exprimé
  dans le **repère de la pince**, invariant par rotation.
- Vitesses et durées exprimées **par seconde** depuis la phase 2
  (`POS_SPEED_RATE`, `ANG_SPEED_RATE`, `DQ_MAX_RATE`, `GRIPPER_HOLD_S`,
  `STATE_TIMEOUT_S`), converties en par-frame via `env.step_dt`. La policy se
  comporte donc à l'identique quelle que soit la valeur de `decimation`.
- ⚠️ Le servo de poignet est **incrémental**, pas absolu : `_pitch_sum` est
  re-calé à partir du résultat clampé à chaque frame. Ce n'est pas redondant,
  c'est l'anti-emballement. Testé : le retirer fait passer la saturation de
  `Wrist_Pitch` de 7 % à 85 % de l'épisode sur tous les placements.
- Sécurités : `POS_TOL=0.012`, `STATE_TIMEOUT_S ≈ 6.7 s` → `status="failed"`.
- `is_success()` : cube dans un rayon XY de 6 cm autour de la boîte et `z < 0.09`.
- **Codée en dur pour `Cube` / `BoxFloor`** → ne marche que sur `cube_to_box`.

### 2.11 Enregistrement dataset

`utils/lerobot_recorder.py` + `utils/lerobot_interface.py` :

- Buffers CPU pré-alloués, capacité `40 × fps = 1200 frames`, dépassement =
  frames silencieusement ignorées.
- Thread asynchrone : à chaque `stop_recording`, l'épisode part dans une queue,
  puis `dataset.save_episode()` + `finalize()` + ré-ouverture du dataset.
- Conversion sim ↔ réel (`LeRobotSO101Interface`) : radians ↔ échelle moteur
  LeRobot `-100..100` (gripper `0..100`), via `SO101_USD_MAPPING`
  (`shoulder_pan ±110°`, `shoulder_lift ±100°`, `elbow_flex -100..90°`,
  `wrist_flex ±95°`, `wrist_roll ±160°`, `gripper -10..100°`).
- `action` enregistrée = **cible** envoyée au step ; `observation.state` =
  position articulaire **mesurée**. Même sémantique que le robot réel.
- Boucle `--auto` + enregistrement : succès → épisode sauvé, échec → jeté, reset
  avec nouvelle randomisation, arrêt après `--num_episodes` succès.
- Extras optionnels hors LeRobot : `--save_mp4`, `--depth`, `--instance_id_seg`.

### 2.12 Dataset produit — `datasets/cube_dataset`

| Champ | Valeur |
|---|---|
| Format | LeRobot **v3.0** |
| `robot_type` | `so_follower` |
| Épisodes / frames | **200 / 69 864** (~349 frames par épisode) |
| Taille disque | 539 MB |
| `fps` déclaré | 30 |
| Features | `action` (6), `observation.state` (6), `observation.images.wrist`, `observation.images.front` |
| Vidéos | 480×640, codec **AV1**, yuv420p |
| Instruction | `"Pick up the cube and place it in the box"` |

Statistiques d'action (échelle moteur LeRobot) :

| Joint | min | max | mean | std |
|---|---|---|---|---|
| shoulder_pan | -29.46 | 32.66 | -0.78 | 16.20 |
| shoulder_lift | -67.45 | 30.50 | -17.83 | 16.33 |
| elbow_flex | -41.26 | 89.49 | 22.47 | 21.71 |
| wrist_flex | -0.71 | **100.00** | 55.43 | 25.91 |
| wrist_roll | -57.42 | -57.42 | -57.42 | **0.00** |
| gripper | -0.00 | 40.34 | 15.15 | 18.49 |

---

## 3. Points d'attention identifiés (2026-08-04)

Trouvés pendant l'analyse de reprise. Classés par impact sur le world model.

| # | Problème | Détail | Impact |
|---|---|---|---|
| 1 | ~~**Cadence 60 Hz écrite comme 30 fps**~~ | ✅ **Corrigé en phase 2.** `decimation` passé de 2 à 4 ⇒ contrôle à 30 Hz. Toutes les constantes de mouvement sont désormais exprimées par seconde et converties via `env.step_dt`, donc le comportement est identique en temps simulé quelle que soit la cadence (vérifié : écart ≤ 4 % et ≤ 4 mm entre 30 et 60 Hz). Effet de bord : ~2× moins de frames et ~2× moins de rendus. | — |
| 2 | **`wrist_roll` constant** | `std = 0.0` sur les 200 épisodes : la policy scriptée fige `_roll_target` à la valeur home. | 🔴 Fort — une dimension d'action morte : le WM ne peut rien apprendre du roll. |
| 3 | **`wrist_flex` effleure sa butée** | max exactement `100.0` = butée 95°, le joint touche donc bien le stop. **Dimensionné sur la géométrie réelle (2026-08-06)** : il n'y passe que **8 % de l'épisode**. Deux hypothèses successives ont été testées et **infirmées** : le servo de poignet (mesure : le retirer aggrave), puis une quasi-singularité en bout d'allonge (c'était un artefact de mes longueurs de segments devinées — avec la vraie chaîne, les 10 placements réussissent au lieu de 3). | 🟡 Faible — cosmétique, pas de perte de données. |
| 4 | **DR quasi absente sur `cube_to_box`** | Pas de HDRI, pas de randomisation de couleur/texture du cube, de la boîte, du sol ni du tapis ; robot figé en orange ; pas de bruit sur les actions ; pas de variation masse/friction. | 🔴 Fort — l'encodeur figé verra 200 fois la même scène. |
| 5 | **Diversité de scène nulle** | 1 scène, 1 objet manipulable, 1 tâche, 1 instruction texte. | 🔴 Fort — c'est précisément le chantier du jour. |
| 6 | ~~`--save_mp4` seul est cassé~~ | ✅ **Corrigé en phase 2.** L'épisode est maintenant commité **avant** tout export vidéo, l'export est isolé dans son propre `try/except`, et depth/segmentation ne sont lus que s'ils existent. Les épisodes perdus par le thread async sont désormais **comptés** (`num_failed_episodes`) et tracés, au lieu d'être avalés. | — |
| 7 | `--auto` non générique | `ScriptedPickPlace` lit `Cube` / `BoxFloor` en dur ; les offsets sont calibrés pour un cube de 2.5 cm. | 🟠 Moyen — bloque toute nouvelle scène en collecte auto. |
| 8 | `num_envs` forcé à 1 | `SO101TeleopEnvCfg.__post_init__` écrase `--num_envs`. | 🟡 Faible — collecte séquentielle uniquement. |
| 9 | Reproductibilité par épisode | `--seed` fixe l'env au lancement ; les tirages de reset viennent du RNG global. Un run entier est reproductible, un épisode isolé non. | 🟡 Faible |
| 10 | Capacité buffer silencieuse | Au-delà de 1200 frames, les frames sont ignorées avec un simple `print`. | 🟡 Faible — marge OK (349 frames actuels). |

---

## 4. Backlog

Non priorisé — à piocher session après session.

**Génération de scènes**
- [ ] Randomisation intra-scène : couleur / taille / masse des objets au reset.
- [ ] Génération procédurale : N objets distracteurs placés aléatoirement.
- [ ] Nouvelles scènes (autres objets, autres cibles, autres géométries).
- [ ] Scènes à base d'assets USD réalistes plutôt que des primitives.
- [ ] Randomisation matériaux/textures du sol, des murs et du tapis.
- [ ] Activer la skybox HDRI (24 `.exr` déjà présents) sur `Teleop-Task`.

**Qualité des données**
- [ ] Corriger l'échelle de temps (cf. point 1) : soit `decimation=4`, soit `fps=60`.
- [ ] Faire bouger `wrist_roll` (cf. point 2).
- [ ] Éviter la saturation `wrist_flex` (cf. point 3).
- [ ] Trajectoires non-optimales / bruitées pour couvrir l'espace d'état.
- [ ] "Play data" : mouvements exploratoires sans tâche, pour la dynamique pure.

**Outillage**
- [ ] Rendre `ScriptedPickPlace` paramétrable par la scène (noms + offsets dans le `SCENE`).
- [ ] Script d'inspection de dataset (durée, taux de succès, couverture par joint).
- [ ] Débloquer `num_envs > 1` pour la collecte.
- [ ] Corriger `--save_mp4` (cf. point 6).

---

## 5. Journal des sessions

Format d'une entrée :

```
### AAAA-MM-JJ — titre
**Fait :** …
**Fichiers :** …
**Notes / à retenir :** …
```

### 2026-07-29 — Base de simulation (commit `01774b8`)

**Fait :** import du workshop NVIDIA Sim-to-Real-SO-101 puis extensions maison —
env `Teleop-Task` (sol, murs, dome light, tapis, 2 caméras), système de scènes
modulaire `--scene`, téléop clavier cartésienne (IK DLS), policy scriptée
pick-and-place `--auto` avec tuning à chaud, enregistrement LeRobot v3.0
asynchrone, boucle de collecte automatique succès/échec.
Collecte de `datasets/cube_dataset` : **200 épisodes / 69 864 frames**.

**Fichiers :** `tasks/task_env_cfg.py`, `utils/scene_loader.py`,
`utils/scripted_policy.py`, `utils/keyboard_ee_control.py`,
`utils/lerobot_recorder.py`, `scenes/cube_to_box.py`, `USAGE.md`.

**Notes :** tout est committé sur `leisaac`. `datasets/` est gitignoré.

---

### 2026-08-04 — Reprise : analyse + mise en place du suivi

**Fait :** analyse complète du code de simulation, création de ce document
(inventaire de l'existant, 10 points d'attention, backlog).

**Fichiers :** `SIMULATION.md` (nouveau).

**Notes / à retenir :** deux problèmes bloquants pour le world model ont été
identifiés avant d'ajouter quoi que ce soit — l'échelle de temps 60 Hz / 30 fps
(point 1) et l'absence de diversité visuelle (points 4 et 5). À traiter avant de
relancer une grosse collecte, sinon les données seront à refaire.

**Suite de la session :** cadrage du chantier « dataset 5000 épisodes » →
[ROADMAP_DATASET.md](ROADMAP_DATASET.md). Contrainte nouvelle : plus d'accès GPU
côté dev (vacances), chaque test passe par un aller-retour avec un binôme. La
feuille de route est donc construite autour de « tout ce qui est testable sans
Isaac Sim doit l'être », et d'une commande unique côté binôme.

Décisions arbitrées :

- **Répartition** : 30 % exploration aléatoire, 25 % interaction objets,
  30 % reaching, 5 % pick & place, 10 % tâches difficiles (ces dernières en fin
  de chantier, les échecs étant conservés).
- **Échelle de temps** : passage à `decimation=4` (contrôle 30 Hz). Vérifié que
  le dataset réel `edslvtre/duck_dataset_augmented` est bien à `fps=30`,
  LeRobot v3.0, `so_follower`, 2 vidéos 480×640 AV1 — même format que la sim.
  Conséquence : `cube_dataset` (200 ép.) est à l'ancienne échelle et sera refait.
- **Rétention** : la logique de rejet des échecs est **supprimée** pour tous les
  modes. Seul reste un filtre d'épisodes *invalides* (trop courts, NaN, IK
  divergente, bras immobile, butée permanente). Un side-car
  `meta/collection_meta.json` trace mode / seed / succès par épisode.
- **Assemblage** : un seul dataset construit par greffes successives
  (A → B → C → D → E), un `task` distinct par mode.

**Calibration obtenue (2026-08-06)** — le 2e passage de la sonde a livré les
200 échantillons. La chaîne cinématique du SO-101 est désormais **résolue
exactement** (résidu 0,0001 mm) à partir des jacobiennes et des positions
mesurées, via `tests/fit_chain.py`. Conséquences immédiates :

- le substitut n'est plus approximatif, il est **calibré** ;
- les butées réelles correspondent **exactement** à `SO101_USD_MAPPING` ;
- le pick & place réussit **10/10** placements (contre 3/10 sur la géométrie
  devinée) en **151-172 frames, soit 5,0-5,7 s** — dans la cible de 3-6 s ;
- 217 tests verts, plus aucun skip.

**Phase 2 livrée** — corrections bloquantes : contrôle à 30 Hz (`decimation=4`)
avec toutes les constantes de mouvement exprimées par seconde et converties via
`env.step_dt` ; `--save_mp4` ne perd plus l'épisode ; les échecs du thread de
sauvegarde sont comptés. Une hypothèse sur la saturation du poignet a été
implémentée, mesurée, **infirmée** et annulée — la vraie cause est une
quasi-singularité en bout d'allonge, à confirmer sur le vrai bras. 180 tests
verts. Détail dans [ROADMAP_DATASET.md §3](ROADMAP_DATASET.md).

**Phase 1 livrée** — filet de test sans GPU : 159 tests verts sur CPU, sans
Isaac Sim. Détail dans [tests/README.md](tests/README.md). Nouveaux modules :
`scripts/isaac_probe.py` (sonde machine réelle, 1 commande),
`utils/scene_validation.py` (→ `validate_scenes`),
`utils/episode_validation.py` (filtre de validité), plus le harness de test
(`tests/`). Deux défauts trouvés à cette occasion, dont un corrigé — voir
[ROADMAP_DATASET.md §2](ROADMAP_DATASET.md).
