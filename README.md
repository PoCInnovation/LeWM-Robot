# LeWM-Robot

World model action-conditionné pour le bras SO-101 : encodeur DINOv3 figé,
fusion multi-caméra, predictor transformer, LoRA sim→réel et planner CEM.
Cible matérielle : **RTX 5090 (Blackwell, 32 Go)**, avec PyTorch 2.10.0
+ torchvision 0.25.0 compilés pour CUDA 12.8. Validation sur GPU à effectuer
sur la machine cible ; le pipeline reste utilisable sur CPU et RTX 4090.

## Une seule commande : estimer le temps d'entraînement sur la 5090

```bash
git clone git@github.com:PoCInnovation/LeWM-Robot.git && cd LeWM-Robot
git checkout feat/rtx4090
make
```

`make` lance `bench.sh`, qui crée le venv, installe torch CUDA + dépendances, vérifie le GPU,
lance quatre mini-entraînements chronométrés (mêmes modules que le vrai
pipeline) et affiche l'estimation du temps de chaque étape + le total.

Le token est lu automatiquement depuis `configs/hf_token.txt`, inclus dans
le dépôt conformément au choix du projet. Aucune commande `hf auth login`
n’est nécessaire. `HF_TOKEN` permet de le remplacer pour un lancement ;
`HF_TOKEN_FILE` permet de choisir un autre fichier. Le token n’est pas inclus
dans les logs ni dans l’archive du rapport.

**À la fin, renvoyez-nous le fichier `results/benchmark_report_<date>.tar.gz`**
(résumé, `benchmark.json`, log complet, `nvidia-smi`, versions des libs).

Prérequis : driver NVIDIA R570 ou plus récent compatible RTX 5090 (`nvidia-smi` fonctionne), Python ≥ 3.10,
`ffmpeg` (`sudo apt install ffmpeg`), ~10 GB de disque, et le token HF
configuré (nécessaire pour DINOv3). La durée dépend du réseau, du dataset et des paramètres du benchmark.
L’installation est réutilisée aux lancements suivants.

### Windows natif (sans WSL2)

Ouvrir **PowerShell en administrateur** dans le dépôt, puis commencer par le
smoke test :

```powershell
powershell -ExecutionPolicy Bypass -File .\run_windows.ps1 -Smoke
```

Après validation, retirer `-Smoke` pour la chaîne complète. Le lanceur crée le
venv Windows, installe les dépendances et appelle directement les scripts
Python. `ffmpeg`, Python 3.10–3.12 et un pilote NVIDIA récent restent requis.

Tous les scripts GPU appliquent avant le calcul une limite de puissance égale à
**80 % du TGP NVIDIA par défaut**, vérifient qu'elle est active et restaurent la
limite précédente à la sortie. Le lancement est refusé si `nvidia-smi` ne peut
pas garantir ce plafond ; ne pas désactiver ce contrôle pour contourner une
erreur de droits.

Variables : `N_EPOCHS=100 bash bench.sh`, `BATCH_SIZES=64,128,256`,
`DATASET_ID=user/dataset`, `NO_DATASET=1` (pas de téléchargement).

## Ensuite

```bash
make check      # kernels CUDA réels puis chargement DINO
make smoke      # pipeline miniature de bout en bout
make run        # pipeline complet : encodage → fusions → predictor → démo CEM
make help       # toutes les cibles
```

Documentation : [DEPLOYMENT.md](./DEPLOYMENT.md) (machine 5090, réglages,
dépannage) et [PIPELINE.md](./PIPELINE.md) (architecture, scripts, état).

## Get involved

You're invited to join this project ! Check out the [contributing guide](./CONTRIBUTING.md).

If you're interested in how the project is organized at a higher level, please contact the current project manager.

## Our PoC team ❤️

Developers
| [<img src="https://github.com/MrZalTy.png?size=85" width=85><br><sub>[Developer's name]</sub>](https://github.com/MrZalTy) | [<img src="https://github.com/MrZalTy.png?size=85" width=85><br><sub>[Developer's name]</sub>](https://github.com/MrZalTy) | [<img src="https://github.com/MrZalTy.png?size=85" width=85><br><sub>[Developer's name]</sub>](https://github.com/MrZalTy)
| :---: | :---: | :---: |

Manager
| [<img src="https://github.com/adrienfort.png?size=85" width=85><br><sub>[Manager's name]</sub>](https://github.com/adrienfort)
| :---: |

<h2 align=center>
Organization
</h2>

<p align='center'>
    <a href="https://www.linkedin.com/company/pocinnovation/mycompany/">
        <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn logo">
    </a>
    <a href="https://www.instagram.com/pocinnovation/">
        <img src="https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white" alt="Instagram logo"
>
    </a>
    <a href="https://twitter.com/PoCInnovation">
        <img src="https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white" alt="Twitter logo">
    </a>
    <a href="https://discord.com/invite/Yqq2ADGDS7">
        <img src="https://img.shields.io/badge/Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white" alt="Discord logo">
    </a>
</p>
<p align=center>
    <a href="https://www.poc-innovation.fr/">
        <img src="https://img.shields.io/badge/WebSite-1a2b6d?style=for-the-badge&logo=GitHub Sponsors&logoColor=white" alt="Website logo">
    </a>
</p>

> 🚀 Don't hesitate to follow us on our different networks, and put a star 🌟 on `PoC's` repositories

> Made with ❤️ by PoC
