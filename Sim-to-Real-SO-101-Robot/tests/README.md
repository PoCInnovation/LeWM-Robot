# Tests — exécution sans GPU

Cette suite fait tourner le **vrai code de simulation** (contrôleurs, policy
scriptée, conversions, chargeur de scènes) sur une machine **sans NVIDIA et sans
Isaac Sim**. Objectif : ne jamais envoyer au binôme une modif qui aurait pu être
prise en défaut en local.

```bash
# depuis Sim-to-Real-SO-101-Robot/
.venv/bin/python -m pytest            # toute la suite (~35 s)
.venv/bin/python -m pytest -v tests/test_scripted_policy.py
```

Le venv local est ignoré par git. Pour le recréer :

```bash
python3 -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv/bin/pip install numpy pytest pyyaml
```

---

## Comment ça marche

| Fichier | Rôle |
|---|---|
| `conftest.py` | Installe les stubs **avant** que pytest importe les tests |
| `isaac_stubs.py` | Faux modules `isaaclab` / `isaacsim` / `omni` / `carb` / `pxr` / `lerobot` |
| `so101_surrogate.py` | Modèle cinématique CPU du SO-101 : FK, jacobienne, modèle d'actionneur |
| `fake_env.py` | `FakeEnv` minimal + `run_controller()` pour dérouler une boucle fermée |
| `calibration.py` | Charge le dump de la machine réelle, quand il existe |

**`torch` est réel, pas stubbé.** Tous les résultats numériques sur lesquels
portent les assertions sont calculés par la même bibliothèque que le simulateur.

Les stubs créent leurs membres à la demande : un attribut qui commence par une
majuscule devient une classe de config qui enregistre ses kwargs, un attribut en
minuscule devient une fonction. Un nouvel import passe donc en général sans
toucher à `isaac_stubs.py`.

---

## Ce que la suite prouve — et ce qu'elle ne prouve pas

✅ **Prouvé sans GPU**

- La jacobienne du substitut est la dérivée exacte de sa propre FK
  (différences finies, `test_surrogate.py`) — c'est ce qui rend valides tous les
  tests de contrôle bâtis dessus.
- Les machines à états **terminent toujours**, sur 10 placements d'objets et sur
  des géométries de bras **randomisées** : rien n'est calé sur un bras précis.
- Aucune commande ne sort des butées articulaires, aucun NaN, aucun infini.
- Les 8 touches de déplacement bougent la pince **dans la bonne direction**
  (l'IK ferme bien la boucle).
- Les conversions radians ↔ échelle moteur sont inversibles dans les deux sens.
- Le validateur de scènes attrape chacune de ses ~20 règles (chaque règle a un
  test qui la déclenche).
- Le filtre de validité garde les **échecs** et rejette les **dégénérés**.

❌ **Non prouvé — c'est le rôle du smoke test sur machine réelle**

- Le rendu, les caméras, l'encodage vidéo.
- La physique de contact (préhension, frottements, collisions).
- Les **offsets géométriques de préhension** (`FINGER_LEN`, `GRASP_OFFSET`…) :
  ils dépendent des longueurs de segments réelles, que le substitut approxime.
- Le débit réel de collecte.

---

## Le substitut cinématique — calibré

Depuis le 2026-08-06, la géométrie n'est plus devinée : elle est **résolue à
partir du dump de calibration** (200 configurations mesurées sur la machine
réelle), avec un résidu de **0,0001 mm**.

```bash
.venv/bin/python tests/fit_chain.py     # regenere DEFAULT_CHAIN depuis un dump
```

La méthode exploite le fait que la partie angulaire de la colonne `i` de la
jacobienne **est** l'axe du joint `i` en repère monde : les axes sortent en
forme close, puis les longueurs de segments par moindres carrés (driver `gelsd`
— le système est rang-déficient, trois joints partageant un axe).

Deux garde-fous restent en place malgré la calibration :

1. Le modèle est **auto-cohérent** : sa jacobienne est vérifiée numériquement
   contre sa propre FK.
2. Les contrôleurs sont aussi déroulés sur des **géométries randomisées**
   (`randomized_kinematics`), donc rien ne peut être secrètement calé sur un
   seul bras.

Seul l'axe du `Jaw` n'est pas mesurable : les origines des corps sont sur leur
propre axe de joint, donc tourner la pince ne déplace pas l'origine du corps
`jaw` et l'axe ne laisse aucune trace dans les positions. Il est mis parallèle
aux axes de tangage, et rien de ce qui utilise le substitut ne le lit.

---

## Quand le dump de calibration arrive

Faire tourner **une fois** sur la machine GPU :

```bash
python -m sim_to_real_so101.scripts.isaac_probe
```

Récupérer `outputs/isaac_probe.json`, le déposer à l'un de ces emplacements :

```
outputs/isaac_probe.json
tests/calibration/isaac_probe.json
```

(ou pointer `SO101_PROBE_JSON` dessus). Les 10 tests de `test_calibration.py`,
aujourd'hui *skipped*, s'activent alors tout seuls et vérifient les hypothèses
du harness contre le vrai robot : ordre des joints, noms des corps, convention
d'indexation de la jacobienne, butées réelles, fréquence de contrôle, résolution
des caméras. Ils affichent aussi l'écart entre la géométrie approximative et le
vrai bras.

```bash
.venv/bin/python -m pytest tests/test_calibration.py -v
```

---

## Défauts connus encodés dans la suite

| Défaut | Encodage |
|---|---|
| #2 — `wrist_roll` figé | `test_wrist_roll_is_actuated` en `xfail(strict=True)` : il **échouera** dès que la phase 2 corrigera le tir, ce qui forcera à retirer le marqueur |
| #3 — saturation `wrist_flex` | Remonté en *warning* par le filtre de validité, jamais en rejet |
| #10 — capacité du buffer | Un épisode qui atteint 1200 frames est marqué « possiblement tronqué » |

---

## Validateur de scènes

Utilisable seul, sans Isaac et sans venv de test :

```bash
validate_scenes                     # toutes les scènes
validate_scenes cube_to_box         # une seule
validate_scenes --strict            # les warnings deviennent bloquants
```
