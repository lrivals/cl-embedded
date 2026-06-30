# Présentation séminaire ONERA — Texte parlé (script)

> Texte d'accompagnement, partie par partie, aligné sur [`01_structure.md`](01_structure.md).
> Public technique embarqué : chaque notion de Continual Learning (CL) est **introduite avant d'être
> utilisée**. Les chiffres à prononcer sont indiqués en clair (ne pas improviser). Ton expert mais
> pédagogique sur le CL — le public connaît le MCU, pas forcément l'apprentissage continu.

---

## Slide 1 — Titre + contexte

« Bonjour à tous. Je m'appelle Léonard Rivals, je suis en stage de M2 entre l'ISAE-SUPAERO, l'ENAC
et l'entreprise Edge Spectrum. Mon sujet, c'est l'apprentissage incrémental pour des capteurs
intelligents à ressources limitées. En une phrase : **comment faire apprendre un modèle en continu,
directement sur un microcontrôleur, sans qu'il oublie ce qu'il savait** — et avec quelques kilo-octets
de RAM seulement. Je vais vous présenter la démarche, mais surtout l'implémentation embarquée et les
résultats sur carte réelle, qui sont le cœur de mon travail. »

## Slide 2 — Problématique & objectif : le triple gap

« Le contexte applicatif, c'est la **maintenance prédictive industrielle** : détecter une panne de
roulement ou de pompe à partir de vibrations. L'idée est de mettre l'intelligence *sur le capteur
lui-même* — un *capteur intelligent* — plutôt que de tout remonter au cloud, parce qu'en usine la
connexion n'est pas garantie et que les données arrivent en flux continu.

Le problème, c'est que les conditions changent dans le temps : usure, température, nouveaux régimes.
On parle de **drift de distribution**. Un modèle figé se dégrade ; il faut donc qu'il **apprenne en
continu**, c'est ce qu'on appelle le *Continual Learning*, ou CL. Mon objectif de stage est de porter
trois méthodes de CL de Python vers du C sur microcontrôleur. Et scientifiquement, je me positionne sur
un **triple gap** que je vais utiliser comme grille de lecture tout au long : Gap 1, valider sur des
**données industrielles réelles** ; Gap 2, tenir sous **100 Ko de RAM avec des mesures précises** ;
Gap 3, **quantifier en INT8 pendant l'apprentissage**. Voici le fil rouge du stage : une phase 1 sur PC,
une phase 2 sur la carte. »

## Slide 3 — État de l'art CL condensé

« Avant d'entrer dans le code, posons rapidement ce qu'est le CL et pourquoi c'est difficile — trois
défis. Premier : le **drift de distribution**, dont je viens de parler ; il peut être graduel, abrupt
ou récurrent. Deuxième, le plus connu : l'**oubli catastrophique**. Quand un réseau de neurones apprend
une nouvelle tâche B, il écrase les poids utiles à la tâche A et peut perdre presque toute sa performance
sur A — c'est ce que montre la courbe rouge à l'écran. Troisième : le **compromis stabilité-plasticité**.
Plus on protège le passé (stabilité), moins on apprend le futur (plasticité) ; on cherche un bon point
sur ce front de Pareto.

La littérature propose trois familles de solutions. La **régularisation** pénalise la modification des
poids importants — c'est EWC. Le **rejeu** rejoue d'anciens exemples ; mais les meilleures méthodes,
comme QLR-CL, demandent près de 1,3 Mo de RAM : c'est **disqualifié** d'emblée sur ma carte qui n'a que
256 Ko. Enfin l'**architecture** alloue des sous-réseaux ou des couches dédiées — c'est TinyOL et HDC.
J'ai donc retenu la régularisation et l'architecture, qui sont compatibles avec la contrainte mémoire. »

## Slide 4 — Les 3+1 modèles retenus

« Concrètement, quatre modèles. **EWC** — *Elastic Weight Consolidation* — c'est un petit MLP qui, à
chaque nouvelle tâche, ajoute une pénalité dite *de Fisher* : la matrice de Fisher mesure quels poids
étaient importants pour les tâches passées, et on les fige doucement. Un hyperparamètre λ règle le curseur
stabilité/plasticité, vous le voyez sur la figure de droite. **HDC**, le calcul hyperdimensionnel, est
radicalement différent : pas de neurones, pas de rétropropagation ; on encode chaque exemple en un grand
vecteur binaire et on *accumule* des prototypes — par construction, il n'y a quasiment pas d'oubli.
**TinyOL** gèle un auto-encodeur pré-entraîné dans la Flash et n'entraîne en ligne qu'une petite tête de
sortie, la tête *OtO* ; le gros du réseau ne bouge plus, donc il n'oublie pas. Enfin **Mahalanobis**, ma
baseline non supervisée : il mesure la distance d'un point à la distribution normale, et je le mets à jour
de façon incrémentale avec l'algorithme de Welford, sans tout recalculer. Le point commun des quatre :
tenir dans 256 Ko, tourner en FP32, et apprendre en ligne. »

## Slide 5 — Données industrielles

« Pour que ce soit crédible — c'est mon Gap 1 — j'ai travaillé sur **six jeux de données industriels
réels** : des pompes, du monitoring d'équipements, et surtout des roulements : **CWRU**, **Pronostia/FEMTO**,
**CMAPSS** de la NASA pour la durée de vie résiduelle, et **Paderborn**. À partir des signaux bruts
d'accéléromètre, j'extrais des descripteurs classiques en surveillance vibratoire : RMS pour l'énergie,
kurtosis et facteur de crête pour les chocs, skewness, etc. Et je construis des **scénarios de CL** : par
exemple en *domain-incremental*, le modèle voit successivement une pompe, puis une turbine, puis un
compresseur — chaque changement est une nouvelle tâche, et c'est là qu'on teste l'oubli. »

## Slide 6 — Résultats CL sur PC

« Sur PC, avant le portage, je compare les algorithmes sur ces scénarios. La figure de gauche est le
**front de Pareto accuracy contre oubli** : on voit qui retient bien tout en restant précis. Le résultat
marquant pour le Gap 1 : sur les roulements **réels** de Pronostia, EWC atteint une accuracy moyenne de
**0,982** avec un oubli **quasi nul** — le *forgetting* moyen est à **0,000** et le transfert arrière est
même légèrement positif, +0,005. Autrement dit, sur des données industrielles réelles, le CL fonctionne.

À droite, **deux cartes de chaleur**. La première croise **modèle × dataset** sur les jeux industriels
réels — c'est ma preuve du Gap 1 : les modèles supervisés tiennent leur rang, EWC et TinyOL restent entre
**0,93 et 1,0** sur CWRU, monitoring et Pronostia, tandis que la baseline non supervisée Mahalanobis, elle,
**décroche sur CWRU** (autour de 0,16) — c'est normal, elle ne voit pas les labels. La seconde heatmap est
la **version PC seule** — j'en ai retiré la variante EWC INT8 pour ne garder que les FP32 comparables : elle
me sert de **référence**, la performance que la carte devra reproduire une fois portée. La comparaison fine
carte contre PC, je la fais sur la slide 10. Tout ceci est reproductible : seed fixé, snapshot de config.

J'ajoute deux cartes de la condition **optimale par modèle** (`best`) : à gauche l'accuracy, à droite le
**F1 de la classe `faulty`**. Et c'est là le point méthodologique : **l'accuracy ment**. Sur de la
détection de panne déséquilibrée, un modèle peut afficher une bonne accuracy en ne prédisant presque jamais
la classe rare — c'est exactement ce qui s'était passé au Sprint 26, accuracy flatteuse mais F1 à **0,243**.
La heatmap F1 démasque ça : Mahalanobis sur CWRU reste autour de **0,25** de F1 quand son accuracy paraissait
acceptable. **C'est le F1 qu'il faut lire**, pas l'accuracy. »

---

## Slide 6bis — Impact du nombre de features (board)

« Une question revenait : pourquoi **5 features** sur la carte, et qu'est-ce que ça coûte ? J'ai donc
comparé, **directement sur la NUCLEO**, trois conditions : `5feat` la référence figée, `all` les
dimensions natives du dataset, et `best` un sous-ensemble choisi **par modèle** via permutation importance,
avec le nombre de features k optimisé sur le F1 de validation. Le haut, c'est le F1 ; le bas, l'accuracy.

Le verdict est nuancé : le **5-feat est un bon compromis** — pour EWC, il atteint déjà l'essentiel de la
performance pour une RAM contenue. Là où passer aux features natives **paie**, c'est sur les datasets à
forte dimension : EWC sur cmapss monte de **0,38 à 0,62** de F1 board en passant de 5 à 21 features. À
l'inverse, c'est sans effet quand le dataset est déjà sous les 5 features, comme monitoring qui est natif
**4 features**. Deux notes honnêtes : la carte reste **câblée à 5 features** — c'est la footnote ; et j'ai
**corrigé un artefact** — HDC sur monitoring affichait 0,113 à cause d'un zéro-padding 4→5 dégénéré ; en
features natives, la vraie valeur board est **0,867**.

Maintenant, le vrai défi du stage : faire tenir ça sur le microcontrôleur. »

---

## Slide 7 — CŒUR : Portage C sur NUCLEO-F439ZI

« Voici la cible : une **NUCLEO-F439ZI**, un Cortex-M4 à 180 MHz, **256 Ko de SRAM**, avec une unité
flottante matérielle mais — point important — **pas de NPU**. Donc le *forward pass* **et** la
rétropropagation s'exécutent tous les deux sur le CPU, en FP32. J'ai réécrit les modèles en C. Côté
architecture, le schéma à l'écran montre un orchestrateur central, `pipeline.c`, qui reçoit une trame,
choisit le modèle et le fait tourner ; autour, une tête par modèle : `ewc_head.c`, `tinyol.c`,
`mahalanobis.c`, `hdc.c`. Contrainte de conception forte, qui parlera à ce public : **allocation
totalement statique**, dans la section `.bss`, **zéro `malloc`** ; toutes les tailles de buffers et de
couches sont des `#define`. Pas d'allocation dynamique, donc pas de fragmentation et un budget mémoire
connu à la compilation. La figure de droite donne la répartition Flash contre SRAM par modèle :
Mahalanobis est minuscule, EWC autour de 9 Ko, TinyOL garde son backbone en Flash. »

## Slide 8 — CŒUR : Protocole UART PC↔carte

« Pour alimenter la carte, un script Python, `sensor_stream.py`, rejoue le flux capteur sur le port
série — l'USART3, à **115 200 bauds**. Chaque échantillon suit un **cycle en six étapes**. Un : réception
par interruption avec un buffer circulaire en **DMA**, pour ne jamais bloquer le CPU. Deux : décodage des
*float32* et vérification d'intégrité par **CRC8**. Trois : routage vers le bon modèle selon un octet de
commande — 0x01 Mahalanobis, 0x02 EWC, 0x03 TinyOL. Quatre, le cœur du CL : l'inférence, **et** si le flag
`UPDATE` est armé, la **mise à jour des poids sur la carte** plus le calcul des métriques en ligne. Cinq :
renvoi de la réponse en DMA. Six : retour visuel par LED — verte si OK, rouge sur erreur CRC, une LED
clignotante en watchdog. Détail qui montre la progression : la trame de réponse a évolué de la v2 à la
**v3**, qui embarque désormais les **métriques de CL calculées directement par le microcontrôleur** —
accuracy glissante, AUROC, oubli estimé. Le MCU ne fait plus que prédire, il s'auto-évalue. »

## Slide 9 — CŒUR : Profiling & Gap 2

« Comment je mesure ? Avec le **DWT CYCCNT**, le compteur de cycles matériel du Cortex-M : précision au
cycle d'horloge, donc des latences fiables, pas des estimations. C'est ce qui me permet de répondre
sérieusement au Gap 2. Les chiffres : côté RAM, Mahalanobis tient en ~200 octets, EWC en ~9,7 Ko, TinyOL
en ~600 octets de SRAM plus 5,8 Ko de backbone en Flash ; et même le **firmware complet multi-modèle**,
qui embarque tout, ne pèse que **104,6 Ko sur 256, soit 40 %**. Côté latence, on est largement sous le
**budget de 100 ms** par inférence-plus-mise-à-jour : de **5 microsecondes** pour Mahalanobis à
**657 microsecondes** pour la paire la plus lourde — un facteur d'au moins 150 de marge. Et la figure de
débit montre qu'on encaisse bien plus que la cadence d'un capteur industriel typique à 1 kHz. La table de
droite formalise la conformité **Gap 2** : toutes les configurations passent, RAM et latence. »

## Slide 10 — CŒUR : Résultats board réels

« Au-delà des budgets, est-ce que le modèle apprend *correctement* sur la vraie carte ? Deux expériences.
La première, sur CMAPSS, en **régression de durée de vie résiduelle** avec EWC : sur la carte, j'obtiens
une **RMSE de 21,23**, soit un ratio board sur PC de **0,94** — la carte fait quasiment aussi bien que le
PC. La latence est déterministe, autour de 130 µs en inférence seule et 400 µs avec la mise à jour, et
l'empreinte est de **66,7 Ko sur 256, soit 26 %**.

La seconde, le **mode dual** : une *seule* trame UART déclenche deux modèles en séquence — la régression
RUL puis une classification de faute. La latence combinée est de **637 µs**, très loin sous les 100 ms :
Gap 2 validé même en double modèle.

Un point d'honnêteté scientifique, important devant ce public : sur le volet classification de faute, le
F1 est faible. J'ai vérifié — ce n'est **pas un bug de portage**. La **parité numérique board contre PC
est exacte** : la carte calcule exactement les mêmes valeurs que Python. La cause est côté modèle —
de l'oubli catastrophique et un jeu de features mixtes — donc une limitation d'apprentissage, hors du
périmètre du portage. Le portage, lui, est fidèle au bit près.

Pour cadrer cette comparaison carte↔PC, je distingue **deux régimes**, et c'est important. Pour **EWC et
Mahalanobis**, les poids sont entraînés sur PC puis *exportés* dans la carte : on est en régime de **parité
exacte** — la carte rejoue Python au bit près, taux de concordance **1,000**, zéro divergence. Pour **HDC et
TinyOL**, au contraire, la carte calcule *par elle-même* : HDC projette en dimension 1000 là où le PC est en
1024 et s'initialise en ligne ; TinyOL a une architecture board distincte, sans checkpoint. Ces deux-là sont
donc en régime que j'appelle **« matériel seul »** — ils **divergent par construction**. Et ça se voit : sur
CWRU, HDC fait **0,887 sur la carte contre 0,935 sur PC**, un écart modéré et attendu ; mais sur le dataset
monitoring, HDC **s'effondre à 0,11**, parce que les **cinq features câblées** dans le firmware sont
hors-domaine pour ce dataset et que l'init en ligne n'a pas le temps de converger. Le message : cette
divergence n'est **pas un défaut de portage**, c'est une **décision d'architecture assumée** — et savoir
distinguer ces deux régimes, parité exacte contre matériel seul, c'est le cœur de l'analyse comparée. »

## Slide 10bis — CŒUR : Paires de modèles + méta-modèle

« Une fois qu'un modèle tient sur la carte, on peut en faire tourner **plusieurs en parallèle**. J'ai donc
mis en place des **paires** : un détecteur non supervisé, Mahalanobis, associé à un modèle supervisé —
EWC ou HDC. Sur la carte, les deux s'exécutent en séquence sur la même trame, en mode que j'appelle
PAIR_MODE, pour une latence combinée de **256 microsecondes** avec EWC, **657** avec HDC : l'overhead de
co-exécution est quasi nul. L'intérêt, c'est de combiner leurs forces : sur CWRU, Mahalanobis seul plafonne
à un F1 de 0,38, mais l'**ensemble** des deux remonte à **0,99**.

Et je suis allé un cran plus loin avec un **méta-modèle de stacking** : une petite régression logistique
qui *arbitre* les deux sorties de la paire — c'est le TRIPLE_MODE. Elle apprend quand faire confiance à
qui. Résultat : un F1 de **0,997**, soit mieux que la meilleure règle d'ensemble fixe, pour une latence de
seulement **258 microsecondes** — le méta-modèle est négligeable. Le firmware complet, qui embarque tout
cela, reste à **104,6 Ko**, et — point que j'aime souligner — la **parité entre la carte et le PC est
exactement de 1,000** sur 300 échantillons : la carte reconstruit le même verdict que numpy. »

## Slide 11 — CŒUR : Gap 3, quantification INT8

« Dernière brique, le Gap 3 : quantifier en **INT8 pendant l'apprentissage**, pas seulement à l'inférence.
J'ai mené un benchmark INT8 contre FP32 sur quatre modèles et cinq datasets — c'est le tableau à l'écran.
Les gains mémoire sont nets : **×2,33 pour HDC**, **×4 pour EWC et Mahalanobis**. Et surtout, la performance
est **préservée** : pour EWC l'écart de métrique est inférieur à 0,006, pour HDC il est nul. Un cas résiste :
Mahalanobis se dégrade en INT8 à cause de la grande dynamique de l'inverse de covariance ; ma recommandation
est un **repli en Q15**, un format virgule fixe 16 bits. Donc oui, on peut apprendre en continu en entiers
sur ce MCU, presque sans perte. »

---

## Slide 12 — Synthèse triple gap

« Si je récapitule sur ma grille : **Gap 1 comblé** — validation sur roulements et turbofans réels, CWRU,
Pronostia, CMAPSS, Paderborn. **Gap 2 comblé** — moins de 100 Ko de RAM, avec des chiffres mesurés au
cycle près par le DWT. **Gap 3 comblé** — quantification INT8 pendant l'entraînement, en Python et en C,
validée sur carte. Et côté robustesse logicielle : près de **96 tests** automatisés, Unity côté firmware
et pytest côté Python, avec une parité board-PC vérifiée — jusqu'à 1,000. À ma connaissance, c'est le premier
travail à satisfaire les trois simultanément. »

## Slide 13 — Perspectives

« Pour la suite, plusieurs pistes en cours ou ouvertes. D'abord une **étude de sensibilité au seuil** qui
sépare le sain du défaillant — le fameux seuil RUL : je quantifie comment il déplace le ratio d'exemples
positifs et la performance, et je vérifie que le matériel, lui, n'en dépend pas. Ensuite, pousser la
**quantification INT8 jusque sur le firmware embarqué**, de bout en bout, pour fermer complètement le Gap 3
sur la carte. Puis **généraliser au-delà des cinq features câblées** dans le firmware actuel, vers les
dimensions natives variables des datasets. Et enfin l'ajout de nouveaux jeux de données, notamment la
dégradation de **batteries Li-ion**. Merci de votre attention, je suis à vous pour les questions. »

---

## Notes anti-débordement (si je dois couper)

- **Compressibles** : slide 5 (datasets) → garder le tableau, citer 2 datasets ; slide 10bis (paires+méta)
  → garder la figure, énoncer le seul méta F1 0,997 + parité 1,000 ; slide 13 (perspectives) → une phrase.
- **À ne jamais sacrifier** : slides 7-11 (le cœur du stage) et le point d'honnêteté de la slide 10.
- Si une question porte sur un détail (trames octet par octet, Fisher, tests) → renvoyer aux **slides
  d'annexe**.

---

## Annexe — Conditions d'exécution PC ↔ carte & impact par modèle

> Notes de support pour le Q&A — **non destinées à être lues à voix haute**. Elles détaillent pourquoi
> PC (Python) et carte (NUCLEO-F439ZI, C) n'opèrent pas dans les mêmes conditions, et l'effet sur chaque
> modèle. Tous les chiffres viennent de `experiments/exp_S33_*`, `exp_S32_board_sweep_summary.json` et des
> `#define` du firmware (`firmware/stm32f4_blink/inc/*.h`) — aucun n'est inventé.

### Conditions d'exécution

| Axe | PC (Python) | Carte (Cortex-M4) | Impact |
|-----|-------------|-------------------|--------|
| Processeur | x86-64, GHz variable | ARM Cortex-M4 @ 180 MHz | la carte est plus lente par instruction, mais la latence absolue reste ≪ 100 ms |
| Précision flottante | `float64` numpy → downcast FP32 | **FP32 natif** (FPU 1 cycle) | arrondis ULP différents ; sans incidence verdict quand parité |
| Dimensions features | natives : CMAPSS 5, Battery 7, **Pronostia 13** | **câblées à 5** (`EWC_IN=5`, `MAHA_DIM=5`, `TINYOL_IN=5`, `HDC_N_FEATURES=5`) | top-5 par mutual-info (`configs/*_feature_subset.yaml`) ; perte d'info hors CMAPSS |
| Dimension HDC | 1024 | **`HDC_DIM=1000`** | projections non comparables → pas de parité HDC |
| Init des poids | PyTorch (seed-dépendant) | **LCG seed=42** déterministe, ou poids exportés | divergence si non exporté |
| Mesure latence | `time.perf_counter()` (~µs) | **DWT CYCCNT** @180 MHz (~5,5 ns/cycle) | la carte mesure au cycle près, plus fiable |
| Mesure RAM | `tracemalloc` (heap) | **`.bss`** au link-time (firmware complet 104,6 Ko) | mesures non superposables : statique total vs heap |
| INT8 | fake-quant, arithmétique FP32 | INT8/INT16 réel | sur Cortex-M4, INT8 **plus lent** que FP32 (pas de gain FPU) mais RAM ÷2,3–4 |

### Impact par modèle

| Modèle | Classe de parité | Cause de divergence | Chiffre PC ↔ carte | Verdict |
|--------|------------------|---------------------|--------------------|---------|
| **EWC** | `parity` | aucune (poids exportés, CMAPSS = 5 features natives) | parité exacte, `parity_rate=1,0` | board = PC au bit près |
| **Mahalanobis** | `parity` | aucune (poids exportés) | parité exacte, `parity_rate=1,0` | board = PC au bit près |
| **HDC** | `hw_only` | dim 1000 ≠ 1024 + init en ligne (LCG vs `np.random`) | CWRU 0,887 vs 0,935 ; monitoring **0,113** vs 0,850 | diverge par construction ; effondrement monitoring = features hors-domaine |
| **TinyOL** | `hw_only` | archi board distincte, pas de checkpoint | CWRU 0,893 vs 0,979 | diverge par construction, écart modéré |

### Pourquoi ça compte

Les écarts carte↔PC tombent toujours dans l'une de deux catégories : **soit nuls** (régime `parity` :
EWC, Mahalanobis — la carte refait Python à l'identique), **soit explicables par une cause matérielle ou
d'architecture identifiée** (régime `hw_only` : HDC, TinyOL — dimension, init en ligne, features câblées).
Il n'y a **jamais de bug de portage silencieux** : chaque divergence est tracée à sa source. C'est
précisément ce qui rend la validation du Gap 1 (données réelles) et du Gap 2 (budgets mesurés) crédible —
on sait exactement ce que la carte calcule, et pourquoi, par rapport à la référence PC.

---

## Annexe — Détail des figures (notes de support Q&R)

> Réponses aux questions « d'où viennent ces chiffres ? / quel cadre d'exécution ? / est-ce des moyennes
> par dataset ? » pour chaque figure. **Non lues à voix haute.**
> Source des figures « board » : [`scripts/generate_presentation_plots.py`](../../scripts/generate_presentation_plots.py)
> (fonctions `plot_*`), rappelée par [`presentation_plots.ipynb`](presentation_plots.ipynb).
> Quatre figures sont *data-driven* (lues depuis `experiments/comparison_sprint23.json` via
> [`notebooks/board_benchmark_all_datasets.ipynb`](../../notebooks/board_benchmark_all_datasets.ipynb)).

### Slide 6 — Front de Pareto : accuracy vs oubli, **PC vs board** (`pareto_acc_forgetting.png`)

- **Axes** : abscisse = `acc_final` (→ droite = plus précis), ordonnée = `-avg_forgetting` (→ haut =
  retient mieux). **Coin idéal = en haut à droite.** Un point = un couple (modèle × dataset).
- **Quel cadre d'exécution ?** La figure a **deux panneaux** : `PC` (`platform=pc`) et
  `Board (NUCLEO-F439ZI)` (`platform=nucleo_f439zi`), plus le **3ᵉ panneau « overlay »** que j'ai ajouté
  (flèche PC→board pour chaque couple présent sur les deux). Source : `experiments/comparison_sprint23.json`,
  clés `acc_final` / `avg_forgetting`.
- **Est-ce des moyennes par dataset ?** Chaque point est **agrégé sur les tâches d'UN scénario CL d'UN
  dataset** (c'est la moyenne inter-tâches que définissent `acc_final`/`avg_forgetting`), mais **il n'y a
  pas de moyenne inter-datasets** : chaque dataset garde son point. C'est volontaire — on veut voir la
  dispersion par dataset, pas un chiffre unique.
- **⚠️ Expliquer les différences PC ↔ board (le point clé).** Attention à **ne pas confondre deux régimes** :
  - **Ce panneau Pareto = comportement CL *end-to-end*** : sur board, le modèle **s'entraîne EN LIGNE**
    (single-pass, flag `--update`, **5 features câblées**) ; sur PC, entraînement multi-époques sur les
    **dimensions natives**. Ce sont donc **deux points de fonctionnement réellement différents** — c'est
    normal qu'ils ne coïncident pas.
  - **À distinguer de la « parité bit-exact » du Sprint 32** : là, on **exporte les poids PC** dans la
    carte et on compare l'**inférence** → identique au bit près. Deux choses différentes : ici on compare
    *l'apprentissage*, pas *l'inférence à poids figés*.
  - **Le sens du décalage varie** (chiffres réels du JSON, acc/forgetting) :
    - **EWC perd de la rétention en on-board single-pass** : CWRU **PC 1,000 / 0,000 → board 0,883 / 0,175** ;
      Pronostia **0,982 / 0,000 → 0,917 / 0,110** ; monitoring **0,982 / 0,001 → 0,904 / 0,054**.
    - **Mais parfois le board fait MIEUX** (éval binarisée board + conditions différentes) : Mahalanobis
      CWRU **0,160 → 0,629**, Paderborn **0,356 → 0,613** ; EWC Paderborn **0,667 / 0,500 → 0,931 / 0,077**.
  - **Message à transmettre** : l'écart PC↔board n'est pas un « bug », c'est la signature d'un **régime
    d'apprentissage différent** (en ligne, contraint à 5 features). Là où on veut la *parité*, on exporte
    les poids (S32) et elle est exacte.

### Slide 7 — Répartition Flash vs SRAM (`06_memory_breakdown.png`)

- **Quel cadre d'exécution ?** Empreinte **statique au link-time** (pas un pic runtime). Chaque barre
  empile deux sections de l'ELF :
  - **Flash = `.rodata`** → poids **constants** : backbone gelé TinyOL **5,7 Ko**, vecteurs de base HDC
    **2,0 Ko**, paramètres Mahalanobis **0,12 Ko** (EWC = 0, ses poids sont entraînables donc en SRAM).
  - **SRAM = `.bss`** → poids **entraînables / buffers** modifiés en ligne.
- **Est-ce des valeurs moyennes par dataset ?** **Non.** Ce sont des tailles **fixées par les `#define`**
  d'architecture (`EWC_IN=5`, `MAHA_DIM=5`, `HDC_DIM=1000`, etc.), donc **invariantes au dataset** —
  confirmé au Sprint 32 : `.bss` **invariant au seuil RUL et au dataset** (le firmware est câblé à 5
  features). Aucune moyenne n'est calculée, c'est une grandeur de compilation.
- **Comment on a eu ces valeurs ?** Lecture du fichier `.map` produit par le link (`make size`,
  `arm-none-eabi-size`) : tailles des sections `.bss`/`.rodata` par objet. Rien de mesuré au runtime,
  rien d'inventé.
- **Les 4 modèles isolés** sont chacun *compilés seuls* ; la 5ᵉ barre **« Firmware paires+méta »** est le
  binaire complet (tous les modèles + PAIR + TRIPLE liés ensemble) = **104,6 Ko** de `.bss`.

### Slide 9 — Empreinte RAM (`01_ram_budget.png`)

- **Quel cadre d'exécution ?** Identique à `06` : **empreinte statique link-time** (lue dans le `.map`,
  `make size`), **pas un pic runtime**, **pas une moyenne par dataset** (tailles fixées par `#define`,
  invariantes — cf. Sprint 32). Même source de mesure, autre vue.
- **Cette figure ne montre QUE la SRAM (`.bss`)** — la grandeur que le **Gap 2** contraint. Valeurs
  (en octets, link-time) : Mahalanobis **200 B**, EWC **9 728 B (9,7 Ko)**, TinyOL **400 B**, HDC
  **28 000 B (28 Ko)**, firmware complet **104 596 B (104,6 Ko)**.
- **Lignes de budget** : pointillé rouge **64 Ko** (budget Gap 2 par modèle), pointillé sombre **256 Ko**
  (SRAM totale NUCLEO-F439ZI). Le firmware complet = **39,9 % de 256 Ko**.
- **Différence avec `06_memory_breakdown.png`** — c'est la clé à retenir :
  - `01_ram_budget` = **vue SRAM seule** (ce qui compte pour le budget Gap 2 : la RAM vivante).
  - `06_memory_breakdown` = **vue SRAM + Flash empilées** (montre *où vivent* les poids : un modèle
    « léger en RAM » comme TinyOL — 400 B `.bss` — cache **5,7 Ko de backbone en Flash**).
  - Exemple concret : TinyOL fait **400 B** dans `01_ram_budget` (SRAM seule) mais **6,1 Ko** dans
    `06_memory_breakdown` (400 B SRAM + 5,7 Ko Flash). Aucune contradiction : deux sections mémoire
    différentes, même valeur `.bss`.

### Slide 9 — Latence inférence + mise à jour CL (`02_latency.png`) — **valeurs complétées**

- **Valeurs manquantes désormais COMPLÉTÉES.** Les barres `inf+update` de Mahalanobis, HDC et TinyOL
  étaient affichées « à mesurer » tant que la **campagne board S33** n'avait pas tourné. Elles sont
  maintenant **mesurées** et le plot les charge depuis
  `experiments/exp_S33_board_latency/latency_summary.json` (le fallback codé en dur du script a aussi été
  synchronisé sur ces mêmes valeurs) :

  | Modèle | Inférence | Inférence + update CL | Commentaire |
  |--------|-----------|------------------------|-------------|
  | **Mahalanobis** | 5 µs | **5 µs** | update Welford = O(features), négligeable |
  | **EWC** (MC 5-feat) | 50 µs | **251 µs** | backprop + pénalité de Fisher → ×5 |
  | **HDC** | 585 µs | **653 µs** | bundling dans la mémoire associative → +68 µs |
  | **TinyOL** | 5 µs | **5 µs** | seule la tête OtO bouge → quasi gratuit |

- **Comment on a eu ces valeurs ?** Mesure sur **carte réelle** par **DWT CYCCNT** (@180 MHz, ~5,5 ns/cycle),
  via streaming `sensor_stream.py` avec le flag `--update` armé ; driver `run_board_gap1_completion.py`
  → P50 par modèle dans `latency_summary.json`. Pas une estimation : un compteur de cycles matériel.
- **Pourquoi update ≈ inf pour Maha et TinyOL ?** Leur mise à jour ne touche qu'un état minuscule
  (moyenne/variance incrémentale ; une seule couche de sortie). Pour **EWC**, l'update inclut une
  rétropropagation complète + le terme de Fisher → coût dominant. Toutes restent **≪ 100 ms** (Gap 2 ✅).
- **Note** : l'EWC ici (50/251 µs) est la **tête multi-classe à 5 features** (S32), distincte de l'**EWC
  RUL** de la slide 10 (130 µs inf / 403 µs inf+update) — architecture et tâche différentes.

### Slide 9 — Débit vs cadence capteur (`12_throughput.png`)

- **Quel cadre d'exécution ?** C'est le **débit soutenu *end-to-end* du streaming** mesuré **sur carte**
  (expérience E18-01), pas le compute brut. Le **cycle UART complet** est compté : RX DMA → décodage +
  CRC8 → inférence (+update) → réponse TX. Source : `experiments/exp_S18_01_board/profiling.json`
  (`throughput_mean_ips = 34235`, `throughput_min_ips = 16915`, `latency_mean_us = 3.67`).
- **D'où vient le 34 235 ips ?** Du débit *soutenu* incluant la communication. À ne pas confondre avec :
  (a) le **compute brut** seul ≈ 1/3,67 µs ≈ **272 k ips** (latence d'inférence pure, slide 9) ; (b) la
  variante **PC/dry-run** `exp_S18_01` qui affiche **333 333 ips** (3 µs sans UART). Le chiffre de la figure
  est le plus **honnête côté système** : ce que la carte tient réellement en flux.
- **Ce que ça veut dire** : la ligne de référence **1 kHz** = cadence d'un **capteur industriel typique**
  (1 échantillon/ms). La carte soutient **~34× cette cadence** (×34 vs capteur), avec un plancher
  (`min_ips`) encore à **~17×** → elle fait inférence **et** apprentissage en ligne **sans jamais prendre
  de retard** sur le flux. C'est la traduction « temps-réel » du Gap 2.

---

## Slide 11 (détail) — Benchmark INT8 vs FP32, Gap 3 (Sprint 28)

> Détail complet pour le Q&R. Figure : `14_int8_benchmark.png`. Données :
> `experiments/exp_S28_PC_ewc_hdc/` + `experiments/exp_S28_PC_tinyol_maha/` (20 JSON, 4 modèles × 5
> datasets). Métrique = AUROC (détection binaire normal-vs-faute) sauf HDC (F1-macro).

- **Cadre** : benchmark **sur PC**, *fake-quant* INT8 vs FP32, mêmes données, même seed. Chaque cellule
  compare la métrique FP32 → INT8 et le ratio RAM. Verdict Gap 3 = (métrique préservée **et** RAM réduite).
- **Gains RAM mesurés** : **×4,00 pour EWC et Mahalanobis** (32 b → 8 b plein), **×2,33 pour HDC**
  (mémoire associative int16, pas int8 → gain moindre mais réel), **×3,5–3,8 pour TinyOL**.
- **Préservation de la métrique — résultats détaillés (Δ = INT8 − FP32)** :

  | Modèle | Dataset | Métrique FP32 | INT8 | Δ | Verdict |
  |--------|---------|---------------|------|----|---------|
  | **EWC** | CMAPSS | 0,768 | 0,773 | +0,006 | ✅ |
  | EWC | CWRU | 1,000 | 1,000 | −0,0002 | ✅ |
  | EWC | Monitoring | 0,939 | 0,939 | −0,0007 | ✅ |
  | EWC | Pronostia | 0,988 | 0,988 | −0,0001 | ✅ |
  | EWC | Paderborn | N/A | N/A | — | tâches test mono-classe |
  | **HDC** | CMAPSS/CWRU/Monit./Pronostia | 0,72 / 0,93 / 0,76 / 0,71 | identique | **0,000** | ✅ identique |
  | HDC | Paderborn | N/A | N/A | — | feature_bounds non calibrés |
  | **TinyOL** | CMAPSS | 0,720 | 0,740 | +0,020 | ✅ fake-quant régularisante |
  | TinyOL | CWRU | 0,707 | 0,762 | +0,055 | ✅ (amélioré) |
  | TinyOL | Monit./Pronostia | 0,899 / 0,716 | 0,907 / 0,715 | +0,008 / −0,001 | ✅ |
  | **Mahalanobis** | Monitoring | 0,972 | 0,972 | +0,00003 | ✅ |
  | Mahalanobis | CMAPSS | 0,655 | 0,649 | −0,006 | ✅ |
  | Mahalanobis | **CWRU** | 0,475 | 0,239 | **−0,236** | ⚠️ dégradé |
  | Mahalanobis | **Pronostia** | 0,857 | 0,620 | **−0,238** | ⚠️ dégradé |

- **Lecture** : **12/16 cellules** mesurables préservent la métrique (|Δ| ≤ 0,02). EWC et HDC sont les
  meilleurs élèves (Δ ≤ 0,006 et Δ = 0). TinyOL **s'améliore** même parfois (la fake-quant agit comme
  une régularisation). Le seul **point dur = Mahalanobis sur CWRU/Pronostia** (−0,236 / −0,238) :
  la quantification INT8 de l'inverse de covariance Σ⁻¹ écrase sa **grande dynamique** → recommandation
  **repli en Q15** (virgule fixe 16 bits), ce qui confirme le `TODO(arnaud)` du Sprint 28.
- **Conclusion Gap 3** : oui, on peut **apprendre en continu en entiers** sur ce MCU presque sans perte,
  avec un gain RAM ×2,3 à ×4 — modulo le cas Σ⁻¹ à fort dynamique, traité en Q15.

---

## Annexe — Références bibliographiques (articles cités)

> Clés BibTeX du manuscrit (`references.bib`). À citer telles quelles dans les slides / le mémoire.
> Famille CL indiquée pour rappel.

| Clé BibTeX | Référence | Rôle dans la présentation |
|------------|-----------|----------------------------|
| `Kirkpatrick2017EWC` | Kirkpatrick, J. *et al.* (2017). « Overcoming catastrophic forgetting in neural networks ». *PNAS*, 114(13), 3521–3526. | EWC original (régularisation) — slides 3, 4 |
| `Ren2021TinyOL` | Ren, H., Anicic, D., Runkler, T. A. (2021). « TinyOL: TinyML with Online-Learning on Microcontrollers ». *IJCNN 2021*. | TinyOL (architecture, validé MCU) — slide 4 |
| `Benatti2019HDC` | Benatti, S., Montagna, F., Rahimi, A. *et al.* (2019). « Online Learning and Classification of EMG-Based Gestures on a Parallel Ultra-Low-Power Platform Using Hyperdimensional Computing ». *IEEE TBioCAS*, 13(3). | HDC online sur MCU — slides 3, 4 |
| `Ravaglia2021QLRCL` | Ravaglia, L. *et al.* (2021). « A TinyML Platform for On-Device Continual Learning with Quantized Latent Replay ». *IEEE JETCAS*, 11(4). | QLR-CL (rejeu UINT8) — **écarté** (~1,3 Mo RAM), slide 3 |
| `Kwon2023LifeLearner` | Kwon, Y. D. *et al.* (2023). « LifeLearner: Hardware-Aware Meta Continual Learning System for Embedded Computing Platforms ». *ACM SenSys 2023*. | Rejeu embarqué (STM32H747) — slide 3 |
| `DeLange2021Survey` | De Lange, M. *et al.* (2021). « A Continual Learning Survey: Defying Forgetting in Classification Tasks ». *IEEE TPAMI*, 44(7). | Taxonomie CL (3 familles) — slide 3 |
| `Capogrosso2023TinyML` | Capogrosso, L. *et al.* (2024). « A Machine-Learning-Oriented Survey on Tiny Machine Learning ». *IEEE Access*, 12. | Survey TinyML de référence — contexte |
| `Hurtado2023CLPdM` | Hurtado, J. *et al.* (2023). Continual Learning appliqué à la maintenance prédictive. | CL × maintenance prédictive — motivation, slide 2 |

> Les entrées canoniques (DOI, pages exactes) font foi dans le `references.bib` du manuscrit. Pour
> `Benatti2019HDC`, `Capogrosso2023TinyML` et `Hurtado2023CLPdM`, vérifier l'année/le volume exacts
> dans ce fichier avant impression.
