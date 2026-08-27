# LeWM-Robot

World model action-conditionné pour le bras SO-101 : encodeur DINOv3 figé,
fusion multi-caméra, predictor transformer, LoRA sim→réel et planner CEM.
Cette branche (`feat/rtx4090`) est prête pour une machine **RTX 4090**.

## Une seule commande : estimer le temps d'entraînement sur la 4090

```bash
git clone git@github.com:PoCInnovation/LeWM-Robot.git && cd LeWM-Robot
git checkout feat/rtx4090
HF_TOKEN=hf_xxx bash bench.sh        # ou simplement : bash bench.sh
```

`bench.sh` crée le venv, installe torch CUDA + dépendances, vérifie le GPU,
lance 2-3 mini-entraînements chronométrés (mêmes modules que le vrai
pipeline) et affiche l'estimation du temps de chaque étape + le total, avec
`results/benchmark.json` et un log dans `logs/`. Relancer est instantané
(installation sautée). `HF_TOKEN` est facultatif : DINOv3 est gated, sans
token l'étape encodeur est sautée et tout le reste est mesuré.

Variables : `N_EPOCHS=100 bash bench.sh`, `BATCH_SIZES=64,128,256`,
`DATASET_ID=user/dataset`, `NO_DATASET=1` (pas de téléchargement).

## Ensuite

```bash
make smoke      # pipeline miniature de bout en bout (~2-5 min)
make run        # pipeline complet : encodage → fusions → predictor → démo CEM
make help       # toutes les cibles
```

Documentation : [DEPLOYMENT.md](./DEPLOYMENT.md) (machine 4090, réglages,
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