# Scènes personnalisées

Chaque fichier `.py` de ce dossier définit une scène : des objets ajoutés
par-dessus n'importe quelle tâche du repo au moment du lancement.

## Utilisation

```bash
lerobot_agent --task Lerobot-So101-Teleop-Task --scene cube_to_box --keyboard
```

`--scene <nom>` charge `scenes/<nom>.py`. Utiliser de préférence la tâche
`Lerobot-So101-Teleop-Task` (tapis + lightbox + caméras) : la tâche `Base`
n'a pas de sol, les objets tomberaient dans le vide.

## Format d'une scène

Le fichier doit définir un dict `SCENE` avec une liste `objects` :

```python
SCENE = {
    "objects": [
        {
            "name": "Cube",              # nom unique (identifiant USD valide)
            "type": "cuboid",            # cuboid | sphere | cylinder | usd
            "size": (0.025, 0.025, 0.025),  # cuboid: dimensions x/y/z en mètres
            # "radius": 0.02,            # sphere / cylinder
            # "height": 0.05,            # cylinder
            # "usd_path": "mon_asset.usd",  # type usd (relatif à ce dossier)
            "color": (0.9, 0.15, 0.15),  # RGB 0-1 (primitives uniquement)
            "mass": 0.02,                # kg (défaut 0.05)
            "static": False,             # True = objet fixe/décor (murs, support)
            "pos": (0.22, -0.08, 0.06),  # position initiale (m, repère robot)
            "rot": (1.0, 0.0, 0.0, 0.0), # quaternion w,x,y,z (optionnel)
            # Randomisation de pose à chaque reset (optionnel) :
            "pos_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "yaw": (-1.5, 1.5)},
        },
    ],
}
```

## Repères utiles

- Le robot est à l'origine, le tapis est centré vers `x = 0.22`, `z ≈ 0.035`
  (surface). La zone de travail confortable du bras est `x ∈ [0.15, 0.30]`,
  `y ∈ [-0.15, 0.15]`.
- Touche `R` en cours de simulation : remet la scène à son état initial
  (avec re-tirage des `pos_range`).
