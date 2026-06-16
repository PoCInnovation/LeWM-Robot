# Data augmentation - LeWM-Robot

A partir d'un dataset `augment.py` genere un nouveau
dataset contenant les episodes originaux plus un episode augmenté par technique choisie.

Toutes les commandes se lancent depuis le dossier qui contient le dataset source

## Installation

```bash
pip install -r data_augmentation/requirements.txt
sudo apt install ffmpeg
```

`ffmpeg` est un programme systeme (pas un paquet pip), il s'installe a part.

## Utilisation

On choisit les techniques avec des flags, ou `--all` pour toutes. Par defaut le script
lit `duck_dataset` et ecrit `duck_dataset_augmented`.

```bash
# Toutes les techniques sur le dataset par defaut
python data_augmentation/augment.py --all

# Seulement certaines techniques (crop + flou)
python data_augmentation/augment.py --crop --blur

# Choisir le dataset source et le dataset de sortie
python data_augmentation/augment.py --all --src mon_dataset --dst mon_dataset_augmented

# Tester rapidement sur 2 episodes seulement avant un run complet
python data_augmentation/augment.py --all --n-orig 2 --dst test_augmente
```

## Options

| Option | Defaut | Role |
|---|---|---|
| `--src DIR` | `duck_dataset` | dataset source |
| `--dst DIR` | `duck_dataset_augmented` | dataset de sortie (ecrase s'il existe) |
| `--n-orig N` | `0` | nombre d'episodes a traiter (0 = tous) |
| `--seed N` | `42` | graine aleatoire |
| `--all` | | applique toutes les techniques |


## Techniques

| Flag | Effet |
|---|---|
| `--crop` | recadrage + zoom |
| `--blur` | flou gaussien |
| `--color` | luminosite / contraste / saturation |
| `--speed-slow` | ralenti |
| `--speed-fast` | accelere |
| `--frame-drop` | supprime quelques frames |
| `--temporal-crop` | sous-fenetre temporelle |
| `--motor-noise` | bruit sur les signaux moteurs |

Les techniques visuelles (`crop`, `blur`, `color`) appliquent le meme reglage a toutes
les frames d'un episode : pas de tremblement ni de clignotement.