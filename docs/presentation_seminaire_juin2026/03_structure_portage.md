# Présentation technique — Portage C & pipeline expérimental board (structure)

> **Titre** : Du modèle Python au capteur intelligent — portage C et expériences sur NUCLEO-F439ZI
> **Auteur** : Léonard Rivals — ISAE-SUPAERO (DISC) / ENAC (LII) / Edge Spectrum
> **Format** : présentation technique « deep-dive » (~30 min) · complément du séminaire généraliste
> ([`01_structure.md`](01_structure.md) / [`02_script.md`](02_script.md))
> **Public** : experts en informatique embarquée — maîtrisent le MCU, veulent voir **comment** on
> traite les données, du PC jusqu'au C sur la carte.
>
> **Narratif** : suivre le chemin d'**un sample** de bout en bout — dataset → features → trame UART →
> routage modèle → inférence + mise à jour → réponse → JSON d'expérience. **Focus 100 % mécanique** :
> on explique le pipeline et le portage, **pas les résultats** (les chiffres de performance sont dans la
> présentation séminaire). Tous les supports sont des PNG du dépôt (support autonome).
>
> Le texte parlé de chaque slide est dans [`04_script_portage.md`](04_script_portage.md).
> Le recueil des figures est dans [`portage_plots.ipynb`](portage_plots.ipynb).
>
> **Source de vérité** (à ne jamais contredire) : `firmware/stm32f4_blink/inc/pipeline.h`,
> `scripts/sensor_stream.py`, `docs/presentation_board_sprint16_20.md` (doc de référence à jour Sprints 16–31).

---

## Vue d'ensemble du minutage

| # | Slide | Durée | Bloc |
|---|-------|-------|------|
| 1 | Titre + cadrage : le chemin d'un sample | 1 min | Intro |
| 2 | La cible NUCLEO-F439ZI : contraintes de portage | 2 min | Hardware |
| 3 | Toolchain & Makefile dual (même code testé host + board) | 2 min | Toolchain |
| 4 | Architecture firmware : routeur + têtes modèles + mémoire | 3 min | Firmware |
| 5 | Du PC au C : workflow d'export des poids | 3 min | Données → C |
| 6 | Pipeline de données côté PC : features → Z-score | 3 min | Données |
| 7 | **CŒUR** — Trame UART requête, octet par octet | 3 min | Protocole |
| 8 | **CŒUR** — FLAGS : sélecteur de mode par nibble | 3 min | Protocole |
| 9 | **CŒUR** — Cycle d'une inférence sur la carte | 4 min | Exécution |
| 10 | **CŒUR** — Réponses & parité board↔PC | 2 min | Protocole |
| 11 | **CŒUR** — Profiling : comment on mesure | 2 min | Mesure |
| 12 | **CŒUR** — Mise en place d'une expérience | 3 min | Expériences |
| — | Annexe / slides de secours | — | Q&R |
| | **Total** | **≈ 31 min** | |

---

## Slide 1 — Titre + cadrage : le chemin d'un sample (1 min)

- **Éléments** : une phrase de cadrage — *« on a 4 méthodes de Continual Learning validées sur PC ;
  cette présentation montre comment elles vivent réellement sur un microcontrôleur »*. Annoncer le
  fil rouge : **suivre un seul échantillon capteur** depuis le dataset Python jusqu'au JSON de résultat,
  en passant par la carte. Préciser le périmètre : on parle **mécanique et traitement de données**, pas
  performances.
- **Affiché** : [`portage_04_dataflow.png`](../figures/presentation_board/portage_04_dataflow.png)
  (vue d'ensemble du flux, sert de plan).
- **Transition** → « D'abord : sur quelle cible, et quelles contraintes cela impose-t-il au code ? »

## Slide 2 — La cible NUCLEO-F439ZI : contraintes de portage (2 min)

- **Éléments** : **STM32F439ZI** — Cortex-M4 @ **180 MHz** (mesuré DWT), **256 Ko SRAM** (192 Ko + 64 Ko CCM),
  2 Mo Flash, **FPU matériel FP32** (clé des latences µs), **pas de NPU** → forward *et* backprop tournent
  sur le CPU. Ce que la carte ne fait **pas** : pas d'OS (bare-metal, boucle infinie), **pas de `malloc`**
  (toute la RAM statique), pas de FP64. Règle de conception : écrire pour tenir dans **64 Ko** (budget Gap 2 /
  futur portage STM32N6) même si la carte en offre 256.
- **Affiché** : tableau caractéristiques MCU (repris de `presentation_board_sprint16_20.md` §1.2) +
  rappel STM32N6 (cible initiale indisponible).
- **Transition** → « Comment compile-t-on et teste-t-on du code pour cette cible ? »

## Slide 3 — Toolchain & Makefile dual (2 min)

- **Éléments** : `arm-none-eabi-gcc` (flags `-mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard -Os`),
  OpenOCD + ST-LINK pour flasher, Renode pour la CI sans carte, **Unity** pour les tests. Point central :
  le **Makefile dual** — `make all` produit le binaire ARM, `make test` recompile **le même code C** sur
  x86 avec `-DTEST_MODE` / `-DTEST_HOST` (HAL désactivée, DWT mocké). Conséquence : les tests couvrent le
  vrai code de production → garantie que le portage est fidèle.
- **Affiché** : schéma du flux de dev (édition → `make test` CI → `make all` → `make flash` → série) —
  repris de `presentation_board_sprint16_20.md` §2.3.
- **Transition** → « À l'intérieur du binaire : comment le code est-il organisé ? »

## Slide 4 — Architecture firmware : routeur + têtes modèles + mémoire (3 min)

- **Éléments** : `pipeline.c` = **orchestrateur unique** (réception, CRC, normalisation, routage, émission) ;
  les modèles sont des **têtes indépendantes** (`mahalanobis.c`, `ewc_head*.c`, `hdc.c`, `tinyol.c`,
  `meta_head.c`), chacune avec ses variantes INT8 (`*_int8.c`) aux mêmes signatures. **Trois zones mémoire** :
  **Flash `.rodata`** (poids `*_INIT`, stats Z-score, projection HDC — immuable), **`.bss` SRAM** (poids
  vivants modifiés par le SGD, Fisher, métriques), **stack** (temporaires du forward : `h1[32]`, `h2[16]`,
  `hv[D]`). Au boot, `pipeline_init()` fait `memcpy` Flash → `.bss` : les poids doivent être **modifiables**
  pour apprendre en ligne. **Zéro `malloc`**, tailles via `#define`.
- **Affiché** : [`portage_01_firmware_arch.png`](../figures/presentation_board/portage_01_firmware_arch.png).
- **Transition** → « Ces poids `*_INIT`, d'où viennent-ils ? Du PC. »

## Slide 5 — Du PC au C : workflow d'export des poids (3 min)

- **Éléments** : un modèle entraîné en PyTorch/sklearn (`.pt`/`.pkl`) → script `export_weights_*.py`
  (`export_weights_c.py` pour Mahalanobis + EWC binaire + méta via `--meta`, `export_weights_tinyol.py`,
  `export_weights_ewc_rul.py`, `export_weights_ewc_multiclass.py`) → génère des tableaux
  `static const float` dans `model_weights.h` / `model_weights_rul.h` / `model_weights_multiclass.h` /
  `meta_weights.h`. **Règle projet stricte** : ces headers ne sont **jamais édités à la main** — toujours
  régénérés par le script (reproductibilité). Distinction clé : la version Flash est la **référence θ\***
  (immuable) ; la copie `.bss` est ce que l'apprentissage embarqué modifie.
- **Affiché** : [`portage_05_weights_export.png`](../figures/presentation_board/portage_05_weights_export.png).
- **Transition** → « Côté PC, qu'envoie-t-on exactement comme données ? »

## Slide 6 — Pipeline de données côté PC : features → Z-score (3 min)

- **Éléments** : les 6 datasets industriels (Pump, Monitoring, **CWRU**, **Pronostia**, **CMAPSS**, **Paderborn**) →
  extraction de **features fréquentielles/temporelles** (RMS, kurtosis, facteur de crête, skewness…) →
  **sélection top-5** features (offline, par importance) → **normalisation Z-score** avec les *mêmes*
  moyennes/écarts-types `ZSCORE_MEAN`/`ZSCORE_STD` que ceux figés dans `model_weights.h`. Point important :
  la normalisation se fait côté PC **et** côté carte avec les **mêmes constantes** → cohérence d'entrée
  indispensable à la parité numérique. Piège connu : changer de dataset impose de recalculer les stats
  Z-score et le seuil, puis de re-flasher.
- **Affiché** : tableau datasets (type · label · scénario CL) + encart « 5 features normalisées ».
- **Transition** → « Ces 5 flottants, comment voyagent-ils jusqu'à la carte ? »

## Slide 7 — CŒUR : Trame UART requête, octet par octet (3 min)

- **Éléments** : communication **binaire** sur USART3 @ **115 200 bauds** (ST-LINK VCP → `/dev/ttyACM0`).
  Trame requête construite par `build_frame_v2()` — ordre **réel** :
  `[MAGIC 0xAB 0xCD : 2B][VERSION : 1B][TASK_ID : 1B][TIMESTAMP_MS : 4B LE][N : 1B][features f32×N LE][LABEL : 1B][FLAGS : 1B][CRC8 : 1B]`.
  Header = `struct.pack("<HBBIB", …)` = 9 B, puis 4·N octets de features, puis label+flags (2 B), puis
  **CRC8** (polynôme 0x07, identique côté PC et carte). **Total = 32 octets pour N=5.** Le CRC8 protège
  l'intégralité de la charge utile ; un CRC invalide → trame rejetée (LED rouge).
- **Affiché** : [`portage_02_uart_frame.png`](../figures/presentation_board/portage_02_uart_frame.png)
  (partie haute — trame requête avec offsets exacts).
- **Transition** → « Un octet décide de tout le comportement : le FLAGS. »

## Slide 8 — CŒUR : FLAGS, sélecteur de mode par nibble (3 min)

- **Éléments** : l'octet `FLAGS` est coupé en deux nibbles.
  - **Nibble bas = actions combinables** : `0x01` UPDATE (un pas de SGD avec ce sample), `0x04` CONSOLIDATE
    (frontière de tâche → `ewc_consolidate()` / `hdc_binarize()`), `0x08` RESET, `0x02` PROFILING.
  - **Nibble haut = mode/modèle**, valeur **unique** testée en *exact-match* : single
    (`0x10` EWC · `0x20` HDC · `0x40` EWC INT8 · `0x80` TinyOL), composites
    (`0x30` EWC multiclasse · `0x50` RUL · `0x60` HDC INT8 · `0x70` DUAL · `0xC0` TinyOL INT8),
    ensembles (`0x90`/`0xA0`/`0xB0` PAIR Maha+supervisé · `0xD0`/`0xE0` TRIPLE +méta).
  - **Ordre de dispatch crucial** (sinon collisions de bits : `0x70 & 0x30 == 0x30`) :
    TRIPLE → PAIR → DUAL → MULTICLASS → RUL → HDC_INT8 → TINYOL_INT8 → single → défaut Mahalanobis.
- **Affiché** : [`portage_03_mode_dispatch.png`](../figures/presentation_board/portage_03_mode_dispatch.png).
- **Transition** → « Une fois le mode choisi, que se passe-t-il à chaque cycle ? »

## Slide 9 — CŒUR : Cycle d'une inférence sur la carte (4 min)

- **Éléments** : `pipeline_run()` est **bloquant sur la réception UART**. Cycle en 7 temps :
  1. **RX** octet par octet jusqu'à trame complète ;
  2. **Décodage + CRC8** (rejet si invalide) ;
  3. **Normalisation Z-score** (`x_norm = (x − mean) / std`) ;
  4. **Routage** selon le nibble haut → la bonne tête modèle ;
  5. **Forward** (FP32 sur le FPU) → prédiction + confiance ;
  6. **Update si UPDATE** : SGD selon le modèle — **EWC** = gradient cross-entropy + pénalité Fisher
     `λ·F·(θ−θ*)` ; **HDC** = accumulation d'hypervecteurs (pas de gradient) ; **Mahalanobis** = mise à jour
     EMA/Welford de la moyenne. **Consolidation si CONSOLIDATE** (fin de tâche) ;
  7. **Métriques en ligne** (accuracy, AUROC, forgetting, RMSE, F1) puis **TX**.
  Tout en stack-local → zéro malloc. Souligner que **TinyOL ne fait pas de backprop sur carte** (tête OtO).
- **Affiché** : [`portage_04_dataflow.png`](../figures/presentation_board/portage_04_dataflow.png)
  (zoom sur la rangée « Carte ») + extrait commenté du forward EWC (annexe).
- **Transition** → « La carte répond — mais que contient la réponse, et comment sait-on qu'elle est juste ? »

## Slide 10 — CŒUR : Réponses & parité board↔PC (2 min)

- **Éléments** : la réponse varie selon le mode, et **sa longueur l'identifie** (`sensor_stream.py` désambiguïse) :
  **v2** 14 B (`<BfIHHB`) · **v3** 23 B (`<BfIHfff`, ajoute ACC/AUROC/FORGET calculés *sur la carte*) ·
  **DUAL** 25 B (`<BffIfff`, RUL+faute) · **PAIR** 22 B (`<BfBfIff`, Maha+supervisé) ·
  **TRIPLE** 27 B (`<BfBfIffBf`, PAIR + verdict méta). **Garde-fou méthodologique** : on vérifie la
  **parité numérique board↔PC** (même entrée → même sortie ; ex. parité méta = 1.000 sur 300 échantillons).
  C'est ce qui distingue un **bug de portage** d'une **limite de modèle** : si board == PC, le portage est
  fidèle, point.
- **Affiché** : [`portage_02_uart_frame.png`](../figures/presentation_board/portage_02_uart_frame.png)
  (partie basse — tailles de réponse).
- **Transition** → « Et les chiffres de latence / RAM, comment sont-ils obtenus ? »

## Slide 11 — CŒUR : Profiling, comment on mesure (2 min)

- **Éléments** : **latence** via le compteur de cycles matériel **DWT CYCCNT** —
  `profiling_start()` capture `t0`, `profiling_stop()` calcule `Δcycles`, puis
  `latence_µs = Δ / 180` (180 MHz ⇒ 1 µs = 180 cycles, résolution ≈ 5.5 ns). Le **périmètre mesuré = RX → TX**
  (inclut l'UART). **RAM statique** via les symboles du linker : `bss_bytes = _ebss − _sbss`, calculé au
  runtime et encodé dans la réponse. **Vérification externe** indépendante : `arm-none-eabi-size build/*.elf`.
- **Affiché** : [`portage_06_profiling.png`](../figures/presentation_board/portage_06_profiling.png).
- **Transition** → « On sait envoyer un sample et mesurer. Comment orchestre-t-on une expérience CL complète ? »

## Slide 12 — CŒUR : Mise en place d'une expérience (3 min)

- **Éléments** : `board_experiment_recorder.py` orchestre une **séquence multi-tâches** : pour chaque tâche,
  envoyer N samples avec `FLAGS = PROFILING (+UPDATE)` puis poser `CONSOLIDATE` au **dernier sample de chaque
  tâche** (frontière de domaine). Mode `--dry-run` : valide toute l'orchestration **sans carte** (loopback
  Python) avant la session réelle. Sorties dans `experiments/exp_*/` : `dataset.csv` (réponses brutes),
  `results.json` (métriques CL obligatoires : `acc_final`, `avg_forgetting`, `backward_transfer`,
  `ram_peak_bytes`, `inference_latency_ms`), `config_snapshot.yaml`, `profiling.json`. Variantes :
  `board_pair_recorder.py` (paires, `--triple` pour le méta). Rappel : ne jamais modifier le protocole sans
  synchroniser `sensor_stream.py` **et** `pipeline.c`.
- **Affiché** : [`portage_04_dataflow.png`](../figures/presentation_board/portage_04_dataflow.png)
  (bandeau `experiments/exp_*/`) + arborescence d'un dossier d'expérience.
- **Clôture** → « Tout est mesuré, reproductible, et vérifié par parité. » → questions.

## Annexe — Slides de secours (Q&R)

- **INT8 firmware** : Q7 (EWC, ±128, biais FP32) / Q8 (HDC) / TinyOL int8 ; sur Cortex-M4 FPU **sans SIMD INT8**,
  la latence INT8 est **×1.8–3.3 plus lente** que FP32 (gain = RAM ×2.33–4.0, pas vitesse) ; **fallback Q15
  recommandé** pour Mahalanobis (grande dynamique de Σ⁻¹). Source : `docs/triple_gap.md`.
- **Détail `ewc_head.c`** : init Xavier via LCG (seed=42, déterministe, sans `rand()`), forward
  5→32→16→2 stack-local, perte `L_CE + (λ/2)·Σ Fᵢ(θᵢ−θ*ᵢ)²`, consolidation = snapshot θ* + Fisher EMA.
- **Couverture de test** : ~96 tests Unity (host) + pytest `sensor_stream` / `board_recorder` ; parité board↔PC.
- **Note d'honnêteté** : un F1 faible (ex. faute en DUAL) n'est **pas** un défaut de portage — la parité
  board↔PC est exacte ; la cause est une limite modèle (oubli catastrophique / features mixtes), hors périmètre.
- **Choix toolchain** : arm-none-eabi-gcc, OpenOCD, CMake, Renode CI, Unity.
