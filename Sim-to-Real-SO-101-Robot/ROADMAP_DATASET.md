# Feuille de route — dataset simulé 5000 épisodes

Objectif : produire un dataset LeRobot v3.0 de **~5000 épisodes** pour
l'entraînement du world model DINOv3-WM, en travaillant **sans accès GPU**.

État de la simu avant ce chantier : [SIMULATION.md](SIMULATION.md).

---

## 0. Contraintes qui dictent toute la suite

| Contrainte | Conséquence sur la méthode |
|---|---|
| **Pas de GPU NVIDIA côté dev** | Rien ne peut être validé dans Isaac Sim en local. Tout ce qui *peut* être testé hors Isaac **doit** l'être, sinon on brûle un aller-retour. |
| **Chaque test = un aller-retour humain** | Les runs de validation doivent être **groupés** : une commande, un maximum d'infos en retour. Pas de « refais un run pour voir ». |
| **Le pote doit juste lancer une commande** | Une seule entrée CLI, un fichier de plan YAML, reprise automatique après crash, aucun réglage manuel. |
| **5000 épisodes ≈ 10-20 h de calcul** | Ça se lance une fois. Toute erreur de format découverte à la fin = tout à refaire. Les décisions de format se verrouillent **avant** la collecte. |

**Règle de travail :** aucune modif n'est envoyée au pote sans être couverte par
un test qui tourne sur CPU chez moi.

---

## 1. Décisions à verrouiller avant de coder

Ces points changent le contenu des 5000 épisodes. Une erreur ici = tout refaire.

### D1 — Répartition finale ✅ arbitrée (2026-08-04)

| Mode | Part | Épisodes | Contenu | Ordre |
|---|---|---|---|---|
| A — Random exploration | 30 % | 1500 | Bras seul, décor sans objets | |
| B — Interaction objets | 25 % | 1250 | Pousser / renverser / balayer des objets simples | |
| C — Reaching | 30 % | 1500 | Pince de A vers B, avec orientation | |
| D — Pick & place | 5 % | 250 | `cube_to_box`, script existant retuné | |
| E — Tâches difficiles | 10 % | 500 | Tâches que le robot **ne réussit pas forcément** | **en dernier** |

**Note sur E :** les échecs sont de la donnée *utile* pour un world model —
glissement, chute, contact raté, collision. C'est justement ce que A/B/C/D ne
produisent pas. Conséquence technique : voir C1 ci-dessous.

### C1 — Suppression du rejet des échecs ✅ arbitré (2026-08-04)

La boucle `--auto` actuelle **jette** tout épisode qui ne satisfait pas
`is_success()`, et ne compte que les succès pour `--num_episodes`.
→ **Cette logique est supprimée pour tous les modes.** Un échec est de la donnée
de dynamique de contact, pas du déchet.

`--num_episodes` compte désormais les épisodes **enregistrés**.

**Mais ce n'est pas « tout garder aveuglément ».** Il faut distinguer :

| | Définition | Décision |
|---|---|---|
| **Échec** | Le robot a agi, l'objectif n'est pas atteint (l'objet glisse, la pile s'effondre, la prise rate) | ✅ **conservé** — c'est la donnée qu'on veut |
| **Épisode invalide** | L'épisode ne contient pas de dynamique exploitable | ❌ **rejeté** |

Critères d'invalidité (le seul filtre qui reste) :

- moins de N frames (timeout immédiat, crash de la machine à états) ;
- NaN / inf dans les actions ou l'état ;
- IK divergente (déplacement articulaire aberrant entre deux frames) ;
- bras immobile sur tout l'épisode (variance d'action ≈ 0) ;
- un ou plusieurs joints collés en butée sur > X % de l'épisode.

Ce filtre est **testable hors GPU** (phase 1) et protège les 5000 épisodes d'une
pollution silencieuse.

### C2 — Traçabilité des épisodes

Comme on ne jette plus les échecs, il faut pouvoir les retrouver a posteriori.
Le schéma LeRobot v3.0 n'a pas de champ libre ⇒ **fichier side-car**
`meta/collection_meta.json` écrit par l'orchestrateur :
`episode_index → {mode, scene, seed, success, longueur, motif de validité}`.

Aucune modification du schéma LeRobot, et l'entraînement peut filtrer par mode
ou par succès sans re-parcourir les vidéos.

### D2 — Échelle de temps (bloquant, cf. point 1 de SIMULATION.md)

**Ce qu'on fait aujourd'hui :** physique à 120 Hz, `decimation=2` ⇒ un
`env.step()` = 1/60 s, donc consigne articulaire à **60 Hz**. Le recorder écrit
**une frame par step** et estampille le dataset `fps=30`. LeRobot croit donc que
deux frames sont espacées de 1/30 s alors qu'elles le sont de 1/60 s.

Vérifié sur `cube_dataset` : épisode le plus long = 406 frames, `timestamp` max
= 13,53 s, alors qu'il n'a duré que **6,8 s de temps simulé**. Les vidéos sont en
ralenti 2×.

**Pourquoi c'est bloquant :** le predictor apprend `z_{t+1} = f(z_t, a_t)`. Si un
pas vaut 1/60 s en sim et 1/30 s en réel, **la même action produit deux fois
moins de mouvement en sim**. Ce n'est pas un domain gap à apprendre, c'est une
erreur d'unité que le LoRA absorberait en pure perte.

> ✅ **Prérequis vérifié (2026-08-04)** sur
> [`edslvtre/duck_dataset_augmented`](https://huggingface.co/datasets/edslvtre/duck_dataset_augmented)
> (duck_dataset_v4, 40 réels + 320 augmentés) : `fps = 30`, LeRobot v3.0,
> `robot_type = so_follower`, 2 vidéos 480×640 AV1 à 30 fps, mêmes 6 joints.
> 360 épisodes / 152 311 frames, soit ~423 frames (~14 s) par épisode.
> **La cible est donc bien 30 Hz.**

| Option | Effet | Coût |
|---|---|---|
| **`decimation=4`** → contrôle à 30 Hz ✅ | Aligne sim, réel et label. Épisodes ~175 frames. Le rendu ne tourne plus qu'une fois par pas de contrôle ⇒ **collecte ~2× plus rapide** (le rendu des 2 caméras domine). Dataset 2× plus léger. Plus fidèle au robot réel, qui tourne à 30 Hz. | Retuning de la policy scriptée : `POS_SPEED`, `ANG_SPEED`, `DQ_MAX` ×2 ; `GRIPPER_FRAMES`, `STATE_TIMEOUT`, `HOME_MAX_FRAMES` ÷2. |
| `fps=60` dans le recorder | Timestamps corrects, contrôle fin conservé. | Dataset 2× plus lourd, collecte 2× plus longue, et sous-échantillonnage obligatoire pour fusionner avec le réel. |

**Retenu : `decimation=4` + retuning.** La convergence en boucle fermée se
valide dans le harness offline (phase 1) ; le smoke test AR#2 tranche sur le
taux de succès réel du mode D. Comme les générateurs A/B/C sont écrits de zéro,
le retuning ne pèse en pratique que sur le mode D.

**Conséquence :** les 200 épisodes de `cube_dataset` sont à l'ancienne échelle et
ne sont **pas mélangeables** avec les nouveaux. Ils sont refaits dans le cadre
des 250 épisodes du mode D.

### D3 — Format de sortie : un dataset unique, construit par greffes ✅

**Collecte séquentielle, mode par mode, dans le même dossier de sortie.**
A tourne en premier ; quand il finit, B se greffe dessus ; puis C, D et enfin E.

C'est déjà le comportement du recorder : `init_dataset()` rouvre un dataset
existant au lieu d'en créer un, et `save_episode()` ajoute à la suite
(cf. USAGE.md §6). L'orchestrateur (phase 6) ne fait qu'enchaîner les blocs.

Chaque mode écrit un `task` distinct (ex. `"random exploration"`,
`"push and disturb the objects"`, `"reach the target pose"`,
`"pick up the cube and place it in the box"`, `"hard manipulation attempt"`)
⇒ `task_index` sert de filtre par mode, **sans aucun changement de schéma**.

Avantages : pas de merge à écrire, pas de renumérotation d'index, reprise après
crash triviale (on compte ce qui est déjà là), et le dataset est exploitable
même si on s'arrête après le mode C.

### D4 — Codec vidéo

`cube_dataset` **et** le dataset réel sont tous les deux en **AV1** — on garde
donc AV1 par défaut, c'est cohérent.

Réserve : l'encodage AV1 est lent et pourrait dominer le temps de collecte sur
10 000 vidéos. À mesurer au run de calibrage. Si c'est le goulot, bascule h264
possible — ça ne change rien à l'entraînement (les frames sont décodées), juste
la cohérence de forme avec le réel.

### D5 — Résolution

480×640 × 2 caméras. À conserver (cohérent avec le réel).

Ordre de grandeur : `cube_dataset` fait 539 MB pour 200 épisodes de ~349 frames,
soit ~2,7 MB/épisode. Avec `decimation=4` les épisodes tombent à ~175 frames
⇒ **≈ 8-13 GB pour 5000 épisodes**. À confirmer au run de calibrage, et à
vérifier sur le disque du binôme **avant** de lancer (`--selftest`).

---

## 2. Phase 1 — Filet de test sans GPU ✅ **faite (2026-08-04)**

**C'est la phase la plus importante.** Elle conditionne le nombre d'allers-retours.

> **État : livrée.** 159 tests passent sur CPU, sans Isaac Sim.
> Voir [tests/README.md](tests/README.md) pour le mode d'emploi.
>
> | Livrable | Emplacement |
> |---|---|
> | Sonde Isaac (selftest + dump), **1 commande** | `scripts/isaac_probe.py` |
> | Stubs isaaclab / omni / carb / pxr / lerobot | `tests/isaac_stubs.py` |
> | Substitut cinématique + `FakeEnv` | `tests/so101_surrogate.py`, `tests/fake_env.py` |
> | Validateur de scènes (prod, sans Isaac) | `utils/scene_validation.py` → `validate_scenes` |
> | Filtre de validité d'épisode (prod) | `utils/episode_validation.py` |
> | Bascule auto sur la vraie géométrie | `tests/calibration.py`, `tests/test_calibration.py` |
>
> **Défaut trouvé et corrigé pendant la phase :** la première version du filtre
> de validité rejetait **tous** les épisodes de pick & place. La policy fait
> passer la pince de `JAW_OPEN` à `JAW_CLOSED` en une frame — un pas mesuré de
> 0,77 rad — que le contrôle de divergence lisait comme une IK qui explose. Le
> contrôle porte désormais sur les positions **mesurées** (la physique ne
> téléporte pas un joint), les dimensions commandées en consigne étant exclues
> quand seules les commandes sont disponibles.
>
> **Défaut trouvé, non corrigé :** sur certains placements du cube, le servo de
> poignet coince `Wrist_Pitch` contre sa butée et l'IK ne s'en sort plus —
> l'épisode rampe 425 frames avec 3 dimensions figées. À traiter en phase 2
> (c'est la forme extrême du défaut #3).

### 2.1 Aller-retour #1 — dump de calibration ✅ **obtenu (2e passage)**

**Résultat final :** la chaîne cinématique du SO-101 est **résolue exactement**
— résidu **0,0001 mm** sur 200 configurations — par `tests/fit_chain.py`, qui
lit les axes dans les jacobiennes en forme close puis résout les longueurs de
segments par moindres carrés. Le substitut n'est plus approximatif.

Ce que la calibration a changé, mesuré :

| | Géométrie devinée | Géométrie calibrée |
|---|---|---|
| Erreur FK vs vrai bras | 263 mm | **0,0001 mm** |
| Pick & place réussi | 3/10 placements | **10/10** |
| Butée `Wrist_Pitch` | 74-100 % de l'épisode | **8 %** |
| Durée d'épisode | 7-11 s | **5,0-5,7 s** |
| Suite de tests | 10 skips | **217 verts, 0 skip** |

Deux « défauts » que j'avais remontés étaient en fait des artefacts de ma
géométrie devinée : le coincement du poignet et la quasi-singularité en bout
d'allonge. Ils n'existent pas sur le vrai bras.

Butées articulaires : **identiques à `SO101_USD_MAPPING`** sur les 6 joints —
la table du repo était bien les vraies limites de l'USD, pas une approximation.

<details>
<summary>1er passage (échec de ma sonde) — pour mémoire</summary>

La moitié « machine » était complète, la moitié « robot » a échoué :
`find_bodies()` traite son argument comme une regex et **lève** quand rien ne
correspond, au lieu de renvoyer une liste vide. La sonde testait des candidats
(`moving_jaw`, `Jaw`, `Wrist`) absents de ce bras.

**La même hypothèse erronée existait dans `scripted_policy.py`** : ses chemins
de repli étaient du **code mort**. Corrigé.

La cause profonde était que `scripts/isaac_probe.py` construisait l'app
Omniverse au niveau module, donc était **impossible à importer hors Isaac** — la
logique est maintenant dans `utils/probe_report.py`, couverte par 15 tests.
</details>

**Cause — un bug de la sonde.** `find_bodies()` traite son argument comme une
regex et **lève** quand rien ne correspond ; il ne renvoie pas de liste vide.
La sonde testait des noms candidats (`moving_jaw`, `Jaw`, `Wrist`) qui
n'existent pas sur ce bras → les sections `robot` et `fk_samples` sont mortes.
Résultat : ni butées articulaires, ni pose de repos, ni échantillons de
cinématique. **Un second passage est nécessaire.**

**Ce qu'on a quand même appris**, et qui ne sera pas à refaire :

- `control_hz = 30.0` → **D2 confirmé sur la vraie machine** ;
- l'ordre des 7 corps, dont le fait que **`wrist` existe** ;
- ffmpeg dispose de tous les encodeurs AV1 ;
- 134,5 Go de disque libre ;
- caméras en 480×640, focale 13,5 mm ;
- les 6 tâches gym s'enregistrent.

**Corrections apportées** pour que le 2e passage aboutisse :

1. La sonde filtre désormais les candidats sur `robot.data.body_names` au lieu
   de les sonder un par un.
2. La **même hypothèse erronée existait dans `scripted_policy.py`** : ses
   chemins de repli (`jaw` ou `wrist` absents) étaient du **code mort**, ils
   auraient levé au lieu de se replier. Corrigé de la même façon.
3. La logique de la sonde a été sortie dans `utils/probe_report.py`. Le script
   `scripts/isaac_probe.py` n'est plus qu'un lanceur : il ne pouvait pas être
   importé hors Isaac (AppLauncher au niveau module), **c'est pour ça que le bug
   est passé**. La logique est maintenant couverte par 15 tests hors GPU.
4. Le substitut reproduit désormais la sémantique réelle de `find_bodies`
   (il lève), ce qui rend les tests de repli réellement significatifs.

### 2.1bis Contenu du dump de calibration

**Commande à envoyer au binôme** (dans l'env Isaac, une seule fois) :

```bash
python -m sim_to_real_so101.scripts.isaac_probe
```

Elle produit `outputs/isaac_probe.json` + un résumé lisible dans le terminal.
Le script est écrit et vérifié syntaxiquement ; chaque section est isolée, donc
un échec partiel n'empêche pas le reste d'être écrit.

Il dumpe :

- ordre exact des joints et des bodies de l'articulation ;
- `soft_joint_pos_limits` réelles des 6 joints ;
- `default_joint_pos` ;
- pour ~200 configurations articulaires tirées au hasard : positions monde des
  bodies `gripper`, `jaw`, `wrist` **et** la jacobienne.

Avec ça je peux construire en local un **modèle cinématique de substitution**
du SO-101 et tester les contrôleurs en boucle fermée sans Isaac.

> C'est le meilleur rapport info/aller-retour de tout le projet : 2 minutes pour
> lui, et ça débloque tous les tests suivants.

### 2.2 Couche de stubs Isaac

Un module de test qui simule `isaaclab`, `isaacsim`, `omni`, `carb`, `pxr`
(faux `AppLauncher`, faux `configclass`, faux `SceneEntityCfg`…), pour que
`scene_loader`, `scripted_policy`, `keyboard_ee_control`, `lerobot_interface` et
les nouveaux générateurs de trajectoires soient **importables et exécutables
sous pytest** sur CPU.

### 2.3 Environnement factice

Un `FakeEnv` qui expose exactement la surface utilisée par les contrôleurs :
`scene[name].data.root_pos_w`, `scene.env_origins`, `robot.data.joint_pos`,
`robot.data.body_pos_w`, `robot.root_physx_view.get_jacobians()`,
`find_joints`, `find_bodies`, `soft_joint_pos_limits` — alimenté par le modèle
cinématique de 2.1.

### 2.4 Ce qu'on teste alors **sans GPU**

- Chaque machine à états **termine** (pas de timeout, pas de boucle infinie) sur
  des centaines de configurations tirées au hasard.
- Le **filtre de validité** (C1) rejette bien les épisodes dégénérés et ne rejette
  **jamais** un simple échec.
- Les actions produites **restent dans les limites articulaires** et ne saturent
  pas (cf. point 3 de SIMULATION.md).
- Les 6 dimensions d'action **bougent** — assertion directe sur `std > seuil`,
  ce qui interdit la régression `wrist_roll` figé.
- Les cibles échantillonnées sont **dans l'espace de travail** atteignable.
- Les specs de scènes sont valides : noms uniques, pas d'objet sous le sol, pas
  d'interpénétration à l'état initial, `pos_range` qui ne sort pas du tapis.
- Les conversions radians ↔ échelle moteur LeRobot sont **inversibles**.
- L'assemblage du dataset produit le bon schéma de features.
- Le plan YAML est cohérent (somme des parts = 100 %, modes connus, scènes existantes).

**Livrable :** `pytest` vert en local, sans Isaac Sim installé.

---

## 3. Phase 2 — Corrections bloquantes ✅ **faite (2026-08-04)**

À faire avant tout nouveau mode, sinon les 5000 épisodes héritent des défauts.
**180 tests verts sur CPU.**

1. ✅ **D2 — échelle de temps.** `decimation` 2 → 4, soit un contrôle à 30 Hz.
   Le retuning n'a **pas** été fait à coups de constantes doublées : toutes les
   vitesses et durées des deux contrôleurs sont désormais exprimées **par
   seconde** et converties via `env.step_dt`. Le comportement est donc correct à
   n'importe quelle cadence, par construction plutôt que par réglage.
   Vérifié en boucle fermée : même issue, écart de durée ≤ 4 % et écart de
   position finale ≤ 4 mm entre 30 Hz et 60 Hz, et moitié moins de frames.
2. ⏭️ **`wrist_roll`** — reste mort dans le pick & place ; sera résolu par la
   conception des modes A et C. Le test de non-régression est en place
   (`xfail(strict=True)`), il basculera tout seul quand ce sera fait.
3. ✅ **Saturation `wrist_flex` — dimensionnée, et bien plus bénigne que craint.**
   Deux hypothèses ont été testées et **toutes deux infirmées par la mesure** :
   d'abord « le servo de poignet écrase sa consigne au clamp » (le retirer fait
   passer la saturation de 7 % à 85 %, donc le re-calage est bien son
   anti-emballement) ; puis une **quasi-singularité en bout d'allonge**, qui
   n'était qu'un artefact de mes longueurs de segments devinées. Avec la chaîne
   calibrée, `Wrist_Pitch` ne passe que **8 % de l'épisode** contre sa butée, et
   les 10 placements réussissent au lieu de 3. Rien à corriger.
4. ✅ **`--save_mp4`** — l'épisode est maintenant commité **avant** tout export
   vidéo, l'export est isolé dans son propre `try/except`, et depth/segmentation
   ne sont lus que s'ils existent.
5. ✅ **Robustesse du thread de sauvegarde** — les épisodes perdus sont comptés
   (`num_failed_episodes`) et tracés avec leur traceback, au lieu d'être avalés.

**Reporté à la phase 3 :** le point de contrôle de la pince n'est valide qu'à
**inclinaison constante** (`GRASP_TILT`). `GRASP_OFFSET` est une hauteur
verticale **monde** et `GRASP_LATERAL` une projection **horizontale monde** :
les deux cessent d'être justes dès que la pince tourne. C'est bloquant pour les
modes A, C et E. La correction — un point de contrôle exprimé dans le repère de
la pince — demande la géométrie réelle du gripper, donc elle attend le dump de
calibration plutôt que d'être devinée au risque de casser un pick & place qui
marche.

---

## 4. Phase 3 — Refonte du générateur 🔄 **3/4 fait (2026-08-06)**

**257 tests verts, mode D non régressé (10/10, 5,0-5,7 s).**

### ✅ 4.1 — Point de préhension invariant par rotation

Le point de contrôle était reconstruit chaque frame à partir de termes en repère
**monde**, dont une projection sur le plan horizontal monde. Mesuré sur le bras
calibré, il dérivait de **34 mm quand `Wrist_Roll` tourne** et **4,7 mm quand
l'inclinaison change** — plus large que le cube de 25 mm.

Il ne tenait que parce que la policy scriptée ne tourne jamais le poignet. Les
modes A, C et E le font par conception, donc le point est désormais un **offset
constant dans le repère de la pince** : rigidement attaché aux doigts, donc juste
sous n'importe quelle rotation. Calibré une fois pour reproduire exactement
l'ancien calcul à la pose de référence.

Testé : rigidité vérifiée sur 100 configurations tirées dans tout l'espace
articulaire (dérive < 0,01 mm), plus un test qui **mesure l'ancienne dérive**
pour que personne ne revienne en arrière en croyant les deux équivalents.

### ✅ 4.2 — Base de contrôleur commune

`ArmController` (dans `utils/arm_control.py`) porte la résolution des joints,
les clamps, le servo de poignet et l'IK amortie. Les deux contrôleurs en
héritent : `scripted_policy.py` passe de 368 à 310 lignes,
`keyboard_ee_control.py` de 253 à 217. L'extraction n'était pas spéculative — le
code était déjà dupliqué entre les deux.

### ✅ 4.3 — Policy paramétrable par la scène

La scène déclare une section `task` (`pick`, `place`, `place_at`, et toute la
géométrie de préhension). `--auto` n'est plus lié à `cube_to_box`. Le validateur
**résout les noms d'objets contre la scène** : une faute de frappe est attrapée
en une seconde au lieu de plusieurs minutes de démarrage Isaac. Une clé inconnue
lève au lieu d'être ignorée silencieusement.

### ⏭️ 4.4 — Spec de scène étendue (randomisation par épisode)

**Pas encore fait, et son profil de risque diffère des trois autres** :
randomiser couleur / taille / masse au reset demande d'écrire des termes
d'événement Isaac qui manipulent des prims USD (comme `randomize_robot_color`).
La construction des configs reste testable hors GPU, mais **pas leur effet
runtime**. À grouper avec le smoke test plutôt qu'à valider à l'aveugle.


## 5. Phase 4 — Les modes

### Mode A — Random exploration (1500 ép., 30 %)

- **Scène :** décor `Teleop-Task` seul (tapis, sol, murs) — pas de `--scene`,
  donc ni cube ni boîte.
- **Trajectoire :** marche aléatoire **lissée** de la pince dans l'espace de
  travail (waypoints tirés + interpolation, pas de sauts), **plus** rotation
  aléatoire du poignet (`Wrist_Roll`, `Wrist_Pitch`) et ouverture/fermeture
  aléatoire de la pince. C'est ce mode qui remplit la dimension `wrist_roll`.
- **Durée :** longueur fixe par épisode (à caler, ~200-300 frames).
- **Validité :** tous les épisodes sont valides sauf divergence IK ou butée
  prolongée. Pas de critère de succès.
- **Enjeu :** c'est la donnée la plus utile au world model (dynamique pure,
  aucune corrélation avec une tâche).

### Mode B — Interaction objets (1250 ép., 25 %)

**C'est le mode qui porte toute la dynamique de contact du dataset.** Le but
n'est pas de réussir quelque chose, c'est de produire des interactions physiques
variées : un objet qui glisse de la pince, une pile qui s'effondre, un objet
poussé hors du tapis, deux objets qui s'entrechoquent. C'est exactement ce que
les modes A, C et D ne produisent jamais.

- **Objets :** primitives d'abord (cube, sphère, cylindre couché « canette »,
  cuboid plat « livre », cylindre fin « crayon »), tailles/masses/couleurs
  randomisées. Pas de dépendance à des assets USD externes → testable et
  reproductible chez le binôme sans téléchargement.
- **Primitives d'interaction :** pousser latéralement, renverser, balayer
  plusieurs objets d'un coup, tapoter, traîner, saisir puis relâcher trop tôt,
  pousser un objet vers le bord du tapis. La préhension n'est pas l'objectif.
- **Scène :** 1 à 3 objets, positions randomisées, sans interpénétration
  initiale ; empilements possibles pour provoquer des effondrements.
- **Rétention :** tout est gardé (cf. C1). L'issue de l'interaction — objet
  déplacé, tombé, sorti du tapis, jamais touché — est enregistrée comme
  métadonnée dans le side-car (C2), pas comme filtre.
- **Mesure :** déplacement de chaque objet entre le début et la fin. Simple et
  robuste, pas besoin de capteur de contact.

### Mode C — Reaching (1500 ép., 30 %)

- **Scène :** décor seul, éventuellement un marqueur visuel statique au point B.
- **Trajectoire :** pose de départ A → pose cible B, avec **orientation** de la
  pince imposée (roll + pitch) et état de pince imposé. Via-points pour des
  chemins non triviaux, vitesses variables.
- **Succès :** pince dans une tolérance de position **et** d'orientation de B.
- **Enjeu :** couvre l'espace de travail bien plus uniformément que le pick &
  place, et fait travailler la rotation.

### Mode D — Pick & place (250 ép., 5 %)

Réutilisation du script existant, avec les corrections de la phase 2 et la scène
paramétrable de 4.2. Rétention : tout est gardé (cf. C1), le succès part dans le
side-car.

Sert surtout de **test de non-régression** : si le taux de succès s'effondre
après le passage à `decimation=4`, c'est que le retuning est raté. C'est le seul
mode dont on connaît la performance de référence (200 épisodes déjà collectés).

### Mode E — Tâches difficiles (500 ép., 10 %) — **à faire en dernier**

⚠️ **Périmètre à préciser.** Le mode B couvre désormais toute la dynamique de
contact « subie » (glissement, effondrement, objet éjecté). Ma lecture de ce qui
reste pour E, à confirmer au moment de l'écrire :

> **B = interaction sans but** (on perturbe, on regarde ce que fait la physique).
> **E = tentative de tâche avec un but, souvent ratée** (le robot *essaie*
> quelque chose de difficile et échoue de façon intéressante).

Pistes :

- saisir un objet trop gros / trop lourd / trop lisse pour la pince ;
- empiler un objet sur un autre ;
- insérer un objet dans un réceptacle étroit ;
- attraper un objet qui roule (sphère, cylindre couché) ;
- viser des cibles en limite d'espace de travail, atteignables seulement en partie ;
- enchaîner deux manipulations dans un environnement encombré.

**Pourquoi en dernier :** mode le plus coûteux à concevoir, le plus dépendant du
retour visuel du binôme, et le seul dont on peut réduire le volume sans
compromettre le reste si le temps manque. Si son périmètre reste flou, la
solution de repli est de basculer ses 500 épisodes sur B.

---

## 6. Phase 5 — Domain randomization

Sans ça, l'encodeur figé voit 5000 fois le même décor (points 4 et 5 de SIMULATION.md).

- Activer la **skybox HDRI** (24 `.exr` déjà présents, code déjà écrit mais
  commenté dans `task_env_cfg.py`) sur `Teleop-Task`.
- Débloquer la **palette de couleurs du robot** (orange / teal / white / black).
- Randomiser **couleurs et matériaux** du sol, des murs et du tapis.
- Élargir les plages existantes : exposition, yaw du tapis, pose caméra externe.
- Randomiser **masse et friction** des objets (mode B).
- Optionnel : léger bruit sur les actions, pour éviter des trajectoires trop lisses.

Chaque terme de DR est ajouté avec une plage **conservatrice** au départ — on
peut toujours élargir, pas rattraper des données irréalistes.

---

## 7. Phase 6 — L'orchestrateur « une commande »

### 7.1 Plan déclaratif

Un YAML unique décrit toute la campagne : modes, parts, scènes, seeds, longueurs
d'épisode, chemin de sortie. Il est **validé hors Isaac** (phase 1).

### 7.2 Commande unique

```
collect_dataset --plan configs/dataset_5000.yaml
```

Ce que ça fait :

1. Valide le plan et l'espace disque avant de lancer quoi que ce soit.
2. Enchaîne les sous-runs (un par mode/scène — le changement de scène impose de
   recréer l'env, donc un processus par bloc).
3. **Écrit tout dans le même dataset**, en mode ajout.
4. **Reprend où ça s'est arrêté** : compte les épisodes déjà présents par
   `task_index` et ne collecte que le reste. Un crash à l'épisode 3000 ne coûte
   rien.
5. Logue en continu : épisodes/heure, taux de succès par mode, ETA.
6. Écrit un rapport final (`collection_report.json`).

### 7.3 Mode smoke test

```
collect_dataset --plan configs/dataset_5000.yaml --smoke
```

~3 épisodes par mode, en headless, avec sortie mp4 et rapport de diagnostic.
**C'est la commande que le pote lance à chaque itération.**

### 7.4 Parallélisation (si nécessaire)

À décider **après** avoir mesuré le débit réel. Option la plus simple : N
processus, N dossiers de sortie, un merge à la fin. À ne faire que si le débit
mesuré rend la collecte séquentielle trop longue.

---

## 8. Phase 7 — Protocole d'aller-retour avec le pote

Pour rendre chaque échange rentable :

- **Une seule commande** à lancer, jamais de réglage manuel.
- **Un template de retour** figé : le log complet, `meta/info.json`,
  `meta/stats.json`, `collection_report.json`, et 2-3 mp4 (un par mode).
- Les questions sont **groupées** : jamais « relance pour voir », toujours un
  lot de vérifications dans le même run.
- Un `--selftest` qui vérifie l'environnement du pote (versions Isaac Lab /
  lerobot, GPU, ffmpeg, espace disque) et sort un rapport — à lancer **une fois**
  au tout début, pour éliminer les surprises d'environnement.

Allers-retours anticipés :

| # | Objet | Ce qu'on en tire |
|---|---|---|
| 1 | Dump de calibration + `--selftest` | Débloque tous mes tests locaux |
| 2 | Smoke test après phases 2-3 | Valide échelle de temps + non-régression pick & place (mode D) |
| 3 | Smoke test après phase 4 | Valide les modes A, B, C visuellement |
| 4 | Smoke test après phase 5 | Valide le rendu DR |
| 5 | Run de calibrage (~100 ép.) | Mesure débit réel, taille disque, taux de succès → dimensionne la campagne |
| 6 | **Collecte 4500 (modes A-D)** | Peut tourner pendant qu'on conçoit le mode E |
| 7 | Smoke test mode E | Valide les tâches difficiles |
| 8 | **Collecte 500 (mode E)** | Complète le dataset |

Objectif : **8 allers-retours**, pas 20. Les modes A-D sont indépendants de E :
la grosse collecte peut démarrer avant que E soit écrit.

---

## 9. Phase 8 — QA du dataset final

Un script d'audit à faire tourner sur le dataset produit (par le pote, ou par
moi si le dataset est rapatrié) :

- comptage par mode, contrôle de la répartition réelle vs cible ;
- **écart-type par dimension d'action** — aucune dimension morte ;
- taux de saturation aux butées ;
- distributions des durées d'épisode ;
- couverture spatiale de la pince (histogramme 3D de l'espace de travail) ;
- intégrité : vidéos décodables, frames alignées avec les parquets,
  pas d'épisode vide ou tronqué ;
- planche de vignettes pour un contrôle visuel rapide de la diversité.

---

## 10. Risques

| Risque | Parade |
|---|---|
| Le retuning d'échelle de temps casse le pick & place | Mode D sert de non-régression ; smoke test avant toute collecte massive |
| L'encodage AV1 domine le temps de collecte | Mesuré au run de calibrage ; bascule h264 possible |
| Espace disque insuffisant chez le pote | Vérifié par `--selftest` **avant** de lancer |
| Crash à mi-collecte | Reprise automatique par comptage des épisodes existants |
| Un mode produit des données inexploitables, découvert à la fin | Audit (phase 9) lancé aussi sur le run de calibrage de 100 épisodes |
| Mon modèle cinématique de substitution diverge du vrai robot | Il sert à tester la **logique**, pas la physique ; le smoke test tranche |
| Dérive de dépendances (Isaac Lab / lerobot) chez le pote | Versions figées et vérifiées par `--selftest` |

---

## 11. Ordre d'exécution

```
D1 ✅  D2 ✅  D3..D5 (format) ──► Phase 1 (tests sans GPU) ──► AR#1 dump calibration
                                                                    │
                                      Phase 2 (corrections) ◄───────┘
                                               │
                                      Phase 3 (refonte) ──► AR#2 smoke (non-régression D)
                                               │
                                      Phase 4 (modes A, B, C, D) ──► AR#3 smoke
                                               │
                                      Phase 5 (DR) ──► AR#4 smoke
                                               │
                                      Phase 6 (orchestrateur) ──► AR#5 calibrage 100 ép.
                                               │
                                               ├──► AR#6 collecte 4500 (A-D) ──┐
                                               │                                │
                                      Phase 4bis (mode E) ──► AR#7 smoke        │
                                               │                                │
                                               └──► AR#8 collecte 500 (E) ──────┴──► Phase 8 (audit)
```

**Point de départ concret :** confirmer que le dataset réel est bien à 30 fps
(prérequis de D2), puis j'attaque la phase 1 — c'est la seule qui ne demande
rien à personne.
