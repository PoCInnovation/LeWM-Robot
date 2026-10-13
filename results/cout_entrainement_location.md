# Coût de l'entraînement LeWM-Robot en location de GPU

> Périmètre : **uniquement la location de machine** (heures GPU facturées + stockage
> loué). Ni achat de matériel, ni électricité, ni temps humain.
> Prix : grilles publiques RunPod / Vast.ai de septembre 2026, en **dollars US**
> (monnaie de facturation). Les prix du marché Vast.ai bougent tous les jours.

---

## En bref

| | Code actuel | Code optimisé (cf. §5) |
|---|---|---|
| Un run complet (00 → 04 + 06) sur RTX 4090 | ~55 min | ~20 min |
| Campagne complète « robot fonctionnel » (§2) | **~12 h de 4090** | **~8,5 h de 4090** |
| Coût de la campagne sur 4090 louée (0,34–0,40 $/h) | **~4–5 $** | **~3–3,5 $** |
| Budget total conseillé (campagne + tests robot + stockage) | ~12–17 $ | **~8–12 $** |
| RAM exigée sur la machine louée | **≥ 128 Go** | **32 Go minimum, 64 Go conseillé** (pic ~25-30 Go) |

**Recommandation : RTX 4090 (ou 5090 si le prix est bas) sur Vast.ai « datacenter
verified » ou RunPod Community, avec le code optimisé.** Un GPU plus cher (A100,
H100) finit plus vite mais ne donne **pas un meilleur modèle**. Même code, même
seed, même résultat. Il coûte 3 à 6 fois plus pour gagner quelques heures.

---

## 1. Point de départ : ce que mesurent (et ne mesurent pas) les résultats

Seul benchmark réel disponible : `results/benchmark_report_20260918_184401/`
(RTX 5090, Windows 11, limite de puissance 80 %). Il annonce **11,7 min** pour un
run complet. Ce chiffre **sous-estime le coût réel d'environ ×3** :

1. **Latents synthétiques** : 15 000 lignes, 1,2 Go, entièrement en VRAM.
   Le vrai dataset (`divisio74/duck_dataset_v3` : 40 épisodes, 17 518 frames,
   17 478 paires, 2 caméras AV1 640×480) donne **34 956 lignes** avec la passe
   d'augmentation activée par défaut, soit un fichier **`.pt` de 42 Go** en fp32.
   Il ne tient pas en VRAM : chaque batch est lu en RAM puis copié vers le GPU.
2. **Décodage vidéo non mesuré** : `ImportError` lerobot sur la machine Windows.
   L'erreur réelle est masquée par `src/data.py`, sans doute torchcodec/FFmpeg.
   Un vrai run sur cette machine s'arrêterait à l'étape 02. **Louer une machine
   Linux règle ce problème.**
3. **Augmentation ignorée**, alors qu'elle double le volume d'entraînement.

### Durée d'un run complet réel sur RTX 4090 (code actuel)

Estimation. Le benchmark 5090 est converti avec un facteur ×1,4 (4090 vs 5090).
Le décodage AV1 a été mesuré en local (4,6 ms/frame en accès aléatoire, 1 cœur).
Le coût des copies RAM → GPU est ajouté.

| Étape | Durée estimée |
|---|---|
| 00-01 vérifications | ~2 min |
| 02 encodage (passe normale + passe augmentée) + sauvegarde 42 Go | ~6 min (4-12) |
| 03 comparaison des 4 fusions × 30 epochs | ~24 min (16-35) |
| 04 predictor, 30 epochs, batch 64 | ~19 min (15-23) |
| 06 démo CEM | ~1-2 min |
| **Total** | **~55 min (40-80)** |
| 05 LoRA, 20 epochs (si données réelles séparées) | +~18 min |

---

## 2. Campagne complète pour un robot fonctionnel

Un seul run ne suffit pas. Il faut encoder, choisir la fusion, régler le predictor
et le LoRA, puis régler le planner CEM. Hypothèse de campagne, en heures de RTX 4090 :

| Phase | Runs | Code actuel | Code optimisé |
|---|---|---|---|
| Installation, smoke test, debug (GPU loué mais peu utilisé) | – | 1,0 h | 1,0 h |
| 02 encodage (dont un ré-encodage éventuel) | 2 | 0,2 h | 0,2 h |
| 03 comparaison des fusions (une seule fois) | 1 | 0,4 h | 0,1 h |
| 04 predictor : 10 runs × 30 epochs + 2 runs × 100 epochs | 12 | 5,2 h | 3,3 h |
| 05 LoRA sim → réel | 4 | 1,2 h | 0,8 h |
| 06 réglage du CEM hors ligne (horizon, candidats, itérations) | – | 1,5 h | 1,5 h |
| Marge +25 % (runs ratés, machine inactive) | – | 2,4 h | 1,7 h |
| **Total** | | **~12 h** | **~8,5 h** |

Pour les autres GPU, seule la partie calcul est mise à l'échelle.
L'installation (1 h + marge) reste fixe.

---

## 3. Ce qu'il faut exiger de la machine louée

| Critère | Code actuel | Code optimisé |
|---|---|---|
| RAM système | **≥ 128 Go** (pic ~84 Go dans 02 et 04, plus jusqu'à ~30 Go d'images en attente dans le DataLoader de 02) | **32 Go minimum** (avec 4-6 workers), **64 Go conseillé** : pic ~25-30 Go dans 02, ~19 Go dans 03/04/05 (détail au §9) |
| VRAM | ≥ 16 Go (24 Go conseillé) | 24 Go (latents entièrement en VRAM) |
| Disque | ≥ 80 Go | ≥ 40 Go |
| CPU | ≥ 8 cœurs (le décodage vidéo AV1 limite l'étape 02) | idem |
| OS | **Linux** (évite le problème torchcodec/FFmpeg de Windows) | idem |

Beaucoup d'offres 4090 grand public ont 32 à 64 Go de RAM. **Avec le code actuel,
filtrer les offres par RAM.** C'est la première raison d'appliquer les
optimisations du §5.

---

## 4. Classement par tranche de prix

Le coût est celui de la **campagne complète du §2** (hors tests robot et stockage).
La colonne « Même résultat ? » indique si le modèle obtenu est équivalent.
Tous les GPU listés calculent en bf16, sauf le T4.

### Gratuit : 0 $

| Option | GPU | Durée totale | Coût total | Même résultat ? | Remarques |
|---|---|---|---|---|---|
| Kaggle Notebooks | T4 16 Go (×2) | ~45 h (code optimisé **obligatoire**) | 0 $ | Oui, en forçant `precision: fp32` | Quota ~30 h GPU/semaine, donc ~2 semaines. Sessions de 9-12 h, 29 Go de RAM. Pas de bf16 matériel sur T4 : en fp32, compter ×5 à ×8 par rapport à une 4090. Utilisable pour du debug ; pénible pour une vraie campagne. |

### Moins de 5 $ : les moins chers, résultat identique

| Option | $/h | Coût total — code actuel (durée) | Coût total — code optimisé (durée) | Même résultat ? | Remarques |
|---|---|---|---|---|---|
| RTX 4090 spot (Vast.ai, interruptible) | ~0,11 | ~1,3 $ (12 h) | **~0,95 $** (8,5 h) | Oui | La machine peut être reprise à tout moment. Les scripts ne reprennent pas un run interrompu, mais un run dure ≤ 1 h : on perd au pire un run. Garder les latents sur un stockage persistant. |
| RTX 5090 spot (Vast.ai) | ~0,21 | ~1,9 $ (9,2 h) | ~1,4 $ (6,8 h) | Oui | Mêmes réserves que la 4090 spot ; offres plus rares. |
| RTX 3090 (Vast.ai) | 0,12–0,25 | 2,2–4,6 $ (18 h) | 1,6–3,3 $ (13 h) | Oui (Ampere, bf16 OK) | Environ 1,6 fois plus lent qu'une 4090. Peu d'hôtes avec 128 Go de RAM. |
| **RTX 4090 RunPod Community / Vast.ai** | **0,34–0,40** | **4,0–4,8 $** (12 h) | **2,9–3,4 $** (8,5 h) | Oui | **Le meilleur compromis prix / fiabilité.** Vast « datacenter verified » (0,40 $) est plus stable que les hôtes non vérifiés. |

### De 5 à 15 $ : plus rapide ou plus confortable, résultat identique

| Option | $/h | Coût total — code actuel (durée) | Coût total — code optimisé (durée) | Même résultat ? | Remarques |
|---|---|---|---|---|---|
| RTX 5090 Vast.ai (prix médian) | ~0,65 | ~6,0 $ (9,2 h) | ~4,4 $ (6,8 h) | Oui | ~25 % plus rapide qu'une 4090. Même architecture que la machine de référence. |
| RTX 4090 RunPod Secure | 0,69 | ~8,2 $ | ~5,9 $ | Oui | Datacenter RunPod avec SLA : moins de pannes d'hôte. |
| RTX 5090 RunPod Community | 0,99 | ~9,1 $ | ~6,7 $ | Oui | |
| L40S 48 Go RunPod Community | 1,09 | ~11,8 $ (10,8 h) | ~8,6 $ (7,9 h) | Oui | 48 Go de VRAM : latents bf16 en VRAM sans effort. Pas plus rapide qu'une 4090. |
| A100 80 Go RunPod Community | 1,59 | ~13,0 $ (8,2 h) | ~9,5 $ (6,0 h) | Oui | 80 Go : même les 42 Go de latents fp32 tiennent en VRAM (`data_device: cuda`). Les hôtes ont en général beaucoup de RAM. |

### De 15 à 35 $ : plus cher, sans gain de qualité

| Option | $/h | Coût total — code actuel (durée) | Coût total — code optimisé (durée) | Même résultat ? | Remarques |
|---|---|---|---|---|---|
| RTX 5090 RunPod Secure | 1,58 | ~14,5 $ | ~10,7 $ | Oui | |
| L40S RunPod Secure | 1,75 | ~18,9 $ | ~13,8 $ | Oui | |
| H100 PCIe 80 Go RunPod | 2,89 | ~19,1 $ (6,6 h) | ~14,2 $ (4,9 h) | Oui | Environ 2× la vitesse de calcul d'une 4090. Mais l'encodage (limité par le CPU) et l'installation ne vont pas plus vite. Modèles trop petits pour exploiter un H100. |
| A100 RunPod Secure | 2,72 | ~22,3 $ | ~16,3 $ | Oui | |
| H100 SXM RunPod | 3,49 | ~23,0 $ | ~17,1 $ | Oui | |
| H100 RunPod Secure | 4,79 | ~31,6 $ | ~23,5 $ | Oui | |

> Au-delà (clusters multi-GPU, grands clouds à la demande) : aucun intérêt pour ce
> projet. Le predictor fait ~10 M de paramètres et tient sur un seul GPU grand public.

**Lecture du tableau :** le résultat ne dépend pas du GPU choisi. Payer plus
cher achète du temps d'attente en moins, pas un meilleur robot. Le seul vrai levier
« moins cher, même résultat » est **l'optimisation du code (§5)** combinée à une
**4090/5090 bon marché**.

---

## 5. Réduire la facture sans changer le résultat

Ces optimisations ne changent pas le modèle entraîné. Elles réduisent la mémoire,
les copies et les calculs refaits inutilement.

| Optimisation | Effet | Impact sur le résultat |
|---|---|---|
| Stocker les latents en **bf16** au lieu de fp32 | 42 Go deviennent 21 Go | Négligeable : l'encodeur calcule déjà en bf16 |
| Stocker **chaque frame une seule fois** (la frame t+1 d'une paire = la frame t de la suivante) | Passe normale : 10,5 Go deviennent 5,3 Go ; ~16 Go au total, qui **tiennent dans les 24 Go d'une 4090** | Aucun |
| Latents **en VRAM** (`data_device: cuda`) | Supprime la copie RAM → GPU (~20 ms par batch) : 04 environ 1,4× plus rapide, 03 environ 3× | Aucun |
| Pré-allouer les tenseurs dans 02 et libérer `data` dans 04 | Plus de double copie en RAM (−42 Go de pic) | Aucun |
| DataLoader de 02 : renvoyer les images en **uint8** au lieu de float32, `prefetch_factor=2` au lieu de 4 | Images en attente : de ~30 Go (8 workers × 4 batches × 944 Mo) à ~3,8 Go. L'encodeur accepte déjà l'uint8 et convertit sur le GPU | Aucun |
| **Ne comparer les fusions qu'une fois** (`-SkipFusion` ensuite) | −24 min par run | Aucun (la fusion gagnante est déjà connue) |
| Batch 256 pour l'étape 03 | 03 environ 5× plus rapide (actuellement limité par le coût de lancement des kernels) | À vérifier : le batch change un peu la dynamique d'optimisation du probe |
| `torch.compile` du predictor | Gain de 1,2 à 1,5× attendu, à mesurer | Aucun |

---

## 6. Autres coûts de location à prévoir

| Poste | Hypothèse | Coût |
|---|---|---|
| Stockage persistant (latents, dataset, venv, checkpoints) | ~60 Go (code actuel) ou ~30 Go (optimisé), pendant 1 à 2 mois | ≈ 2–8 $ au total (≈ 0,07 $/Go/mois chez RunPod, à vérifier) |
| **Tests sur le vrai robot avec inférence sur la machine louée** | ~10 h de sessions sur 4090/5090 | ≈ 3,5–6,5 $ |

**Budget total conseillé (4090, code optimisé) :** ~3 $ (campagne) + ~4 $ (tests
robot) + ~2–4 $ (stockage) = **~8–12 $**. En doublant par prudence le nombre
d'itérations : **~20–25 $**.

---

## 7. Limites : ce que l'argent ne règle pas

- **La latence du planner CEM est le vrai verrou pour un robot fonctionnel.** On
  mesure ~4,1 s par plan sur 5090, soit ~6 s sur 4090, avec 1 000 candidats. Le
  temps croît proportionnellement au nombre de candidats. Un H100 ne ferait gagner
  qu'un facteur 2 environ, alors qu'il faut descendre sous ~200 ms (facteur 20 à 30).
  Il faut des changements d'algorithme : moins de candidats ou d'itérations, moins
  de patches (par exemple 196 → 49 par caméra), CUDA graphs. C'est du temps de
  développement, pas du budget GPU.
- **Données** : 40 démonstrations. Si le modèle ne généralise pas, il faudra
  collecter plus de démos ; les heures GPU augmenteront à peu près en proportion
  du nombre de frames.
- **Estimations, pas mesures.** Aucun run n'a été fait sur 4090 : compter ±30 %.
  Pour fiabiliser, lancer
  `scripts/07_benchmark.py --dataset-id divisio74/duck_dataset_v3` avec de vrais
  latents, pendant la première heure de location (~0,40 $).
- Les prix Vast.ai sont ceux d'un marché : ils changent en permanence. Revérifier
  au moment de louer.

---

## Sources

- Benchmark de référence : `results/benchmark_report_20260918_184401/` (summary.md, benchmark.log)
- Métadonnées du dataset : `https://huggingface.co/datasets/divisio74/duck_dataset_v3` (meta/info.json)
- [RunPod — tarifs](https://www.runpod.io/pricing)
- [GPUPerHour — RunPod, sept. 2026](https://gpuperhour.com/providers/runpod)
- [Vast.ai — tarifs](https://vast.ai/pricing) · [Vast.ai — RTX 5090](https://vast.ai/pricing/gpu/RTX-5090)
- [getdeploying — RTX 4090](https://getdeploying.com/gpus/nvidia-rtx-4090) · [RTX 3090](https://getdeploying.com/gpus/nvidia-rtx-3090) · [RTX 5090](https://getdeploying.com/gpus/nvidia-rtx-5090)
- [Spheron — RunPod vs Vast.ai 2026](https://www.spheron.network/blog/gpu-cloud-pricing-comparison-runpod-vs-vastai-2026/)

---

## 8. Récapitulatif final : tous les coûts, durées et meilleure solution

### 8.1 Toutes les options (entraînement complet du §2)

- **$/h** : prix de location à l'heure.
- **Durée d'un run** : un run complet 00 → 04 + 06. En code optimisé, sans refaire la comparaison des fusions.
- **Temps machine** : nombre d'heures de location pour **toute** la campagne d'entraînement.
- **Coût total** : $/h × temps machine, pour toute la campagne (hors tests robot et stockage).

| # | Option | $/h | Durée d'un run (actuel → optimisé) | Temps machine (actuel → optimisé) | **Coût total (actuel → optimisé)** |
|---|---|---|---|---|---|
| 1 | Kaggle T4 (gratuit) | 0 | – → ~2 h | – → ~45 h (~2 semaines de quota) | **0 $** |
| 2 | RTX 4090 spot (Vast.ai) | 0,11 | 55 min → 20 min | 11,9 h → 8,6 h | **1,3 $ → 0,95 $** |
| 3 | RTX 5090 spot (Vast.ai) | 0,21 | 41 min → 15 min | 9,2 h → 6,8 h | **1,9 $ → 1,4 $** |
| 4 | RTX 3090 (Vast.ai) | 0,12–0,25 | 88 min → 32 min | 18,3 h → 13,0 h | **2,2–4,6 $ → 1,6–3,3 $** |
| 5 | **RTX 4090 RunPod Community** | **0,34** | 55 min → 20 min | 11,9 h → 8,6 h | **4,0 $ → 2,9 $** |
| 6 | **RTX 4090 Vast.ai datacenter verified** | **0,40** | 55 min → 20 min | 11,9 h → 8,6 h | **4,8 $ → 3,4 $** |
| 7 | RTX 5090 Vast.ai (prix médian) | 0,65 | 41 min → 15 min | 9,2 h → 6,8 h | **6,0 $ → 4,4 $** |
| 8 | RTX 4090 RunPod Secure | 0,69 | 55 min → 20 min | 11,9 h → 8,6 h | **8,2 $ → 5,9 $** |
| 9 | RTX 5090 RunPod Community | 0,99 | 41 min → 15 min | 9,2 h → 6,8 h | **9,1 $ → 6,7 $** |
| 10 | L40S 48 Go RunPod Community | 1,09 | 50 min → 18 min | 10,8 h → 7,9 h | **11,8 $ → 8,6 $** |
| 11 | A100 80 Go RunPod Community | 1,59 | 36 min → 13 min | 8,2 h → 6,0 h | **13,0 $ → 9,5 $** |
| 12 | RTX 5090 RunPod Secure | 1,58 | 41 min → 15 min | 9,2 h → 6,8 h | **14,5 $ → 10,7 $** |
| 13 | L40S RunPod Secure | 1,75 | 50 min → 18 min | 10,8 h → 7,9 h | **18,9 $ → 13,8 $** |
| 14 | H100 PCIe RunPod | 2,89 | 28 min → 10 min | 6,6 h → 4,9 h | **19,1 $ → 14,2 $** |
| 15 | A100 RunPod Secure | 2,72 | 36 min → 13 min | 8,2 h → 6,0 h | **22,3 $ → 16,3 $** |
| 16 | H100 SXM RunPod | 3,49 | 28 min → 10 min | 6,6 h → 4,9 h | **23,0 $ → 17,1 $** |
| 17 | H100 RunPod Secure | 4,79 | 28 min → 10 min | 6,6 h → 4,9 h | **31,6 $ → 23,5 $** |

Toutes ces options donnent **le même modèle**. Seuls le prix et le temps d'attente changent.

### 8.2 Les trois options remarquables

| | Option | Coût entraînement | Temps machine | Pourquoi pas celle-là ? |
|---|---|---|---|---|
| La moins chère | RTX 4090 spot (Vast.ai), code optimisé | **0,95 $** | 8,6 h | Interruptible à tout moment. Des runs perdus et des sessions de test robot coupées net. Gagne ~2 $ seulement par rapport à la meilleure option. |
| La plus rapide | H100 PCIe (RunPod), code optimisé | **14,2 $** | 4,9 h | 4 fois plus cher que la meilleure option pour gagner ~3,7 h, sans meilleur résultat. |
| **La meilleure** | **RTX 4090 RunPod Community (0,34 $/h) ou Vast.ai verified (0,40 $/h), code optimisé** | **2,9–3,4 $** | **8,6 h** | Meilleur rapport prix / fiabilité / temps. |

### 8.3 Meilleure solution : budget et durée tout compris

**RTX 4090 louée (RunPod Community ou Vast.ai datacenter verified), sous Linux,
avec le code optimisé du §5.**

| Poste | Temps machine | Coût total |
|---|---|---|
| Campagne d'entraînement complète (§2) | 8,6 h | 2,9–3,4 $ |
| Tests sur le vrai robot (inférence sur la 4090 louée) | ~10 h | 3,4–4,0 $ |
| Stockage persistant (~30 Go, 1 à 2 mois) | – | ~2–4 $ |
| **Total** | **~18,6 h de location** | **~8–12 $** |
| Total prudent (2 fois plus d'itérations que prévu) | ~37 h | ~20–25 $ |

**Durée :**

- **Temps machine** : ~8,6 h pour l'entraînement, ~18,6 h avec les tests robot.
- **Un run complet** : ~20 min une fois le code optimisé (~55 min avec le code actuel).
- **Durée réelle du projet** : environ **2 à 3 jours de travail pour l'entraînement**
  (analyse et réglages entre les runs), puis **2 à 3 jours de tests sur le robot**.
  Soit **~1 à 1,5 semaine**. La machine est arrêtée entre les sessions : seules les
  heures d'utilisation sont facturées, le stockage l'est en continu.

> Sans les optimisations du code, la même solution coûte ~12–17 $ au total. Il faut
> alors une machine avec **≥ 128 Go de RAM**, ce qui réduit le nombre d'offres 4090
> disponibles.

---

## 9. RAM nécessaire avec le code optimisé

Hypothèses : latents en bf16, chaque frame stockée une seule fois, tenseurs
pré-alloués, DataLoader en uint8 avec `prefetch_factor=2`, latents chargés en VRAM
pour l'entraînement.

**Taille des latents : ~15,8 Go**
- Passe normale : 17 518 frames × 2 caméras × 196 patches × 384 dims × 2 octets = 5,3 Go.
- Passe augmentée : 17 478 paires × 4 images × 196 × 384 × 2 octets = 10,5 Go.
- Ces ~15,8 Go tiennent en VRAM sur une 4090 : 15,8 Go + ~5 Go pour l'entraînement ≈ 21 Go sur 24.

| Étape | Ce qui occupe la RAM | Pic de RAM |
|---|---|---|
| 02 encodage | Latents pré-alloués 15,8 Go + images en attente ~3,8 Go + 8 workers ~6 Go + processus principal ~3 Go | **~29 Go** (≈ 25 Go avec 4 workers) |
| 03 / 04 / 05 entraînement | Chargement du `.pt` 15,8 Go + ~3 Go, puis transfert en VRAM et libération de la copie RAM | **~19 Go** au chargement, puis ~4 Go (≈ 5 Go avec `torch.load(mmap=True)`) |
| 06 démo CEM | Modèle + quelques frames | ~4 Go |
| Système (OS, cache) | – | +2-4 Go |

**Conclusion : 32 Go de RAM minimum** (en passant `--num-workers 4` à l'étape 02),
**64 Go conseillé** pour être tranquille. Si seul le passage en uint8 du DataLoader
n'est pas fait, la file d'images peut à elle seule monter à ~30 Go : il faut alors
64 Go.
