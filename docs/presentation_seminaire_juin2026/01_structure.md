# Présentation séminaire ONERA — Structure (plan slide par slide)

> **Titre** : Apprentissage Incrémental pour Capteurs Intelligents à Ressources Limitées
> **Auteur** : Léonard Rivals — ISAE-SUPAERO (DISC) / ENAC (LII) / Edge Spectrum
> **Date** : séminaire du 25 juin 2026 · **Durée cible** : 30 min (+ 5 min questions)
> **Public** : chercheurs en informatique embarquée (LAAS, ONERA, IRT, Supaéro/ENAC) —
> maîtrisent le MCU, **pas** le Continual Learning → introduire chaque notion CL.
>
> **Narratif** : chronologique (état de l'art → données → résultats PC → **portage MCU** → perspectives),
> **pondéré vers l'implémentation et les résultats board** (slides 7-11 ≈ 16 min, le cœur du stage).
> Pas de démo live. Tous les supports sont des PNG déjà présents dans le dépôt (support autonome,
> aucune dépendance réseau — la présentation se fait depuis le laptop de Claire).
>
> Le texte parlé de chaque slide est dans [`02_script.md`](02_script.md).

---

## Vue d'ensemble du minutage

| # | Slide | Durée | Bloc |
|---|-------|-------|------|
| 1 | Titre + contexte | 1 min | Intro |
| 2 | Problématique & objectif (triple gap) | 2 min | Intro |
| 3 | État de l'art CL condensé | 4 min | Théorie |
| 4 | Les 3+1 modèles retenus | 3 min | Théorie |
| 5 | Données industrielles | 2 min | Données |
| 6 | Résultats CL sur PC | 3 min | Résultats PC |
| 7 | **CŒUR** — Portage C sur NUCLEO-F439ZI | 3 min | **Implémentation** |
| 8 | **CŒUR** — Protocole UART PC↔carte | 3 min | **Implémentation** |
| 9 | **CŒUR** — Profiling & Gap 2 | 3 min | **Résultats board** |
| 10 | **CŒUR** — Résultats board réels | 3 min | **Résultats board** |
| 10bis | **CŒUR** — Paires + méta-modèle (S30-31) | 2 min | **Résultats board** |
| 11 | **CŒUR** — Gap 3 : quantification INT8 | 2 min | **Résultats board** |
| 12 | Synthèse triple gap | 1 min | Conclusion |
| 13 | Perspectives | 1 min | Conclusion |
| — | Annexe / slides de secours | — | Q&R |
| | **Total** | **≈ 33 min** | |

---

## Slide 1 — Titre + contexte (1 min)

- **Éléments** : titre du stage · auteur · institutions (ISAE-SUPAERO / ENAC / Edge Spectrum) ·
  période (mars–août 2026) · une phrase de problématique : *« apprendre en continu, sur un
  microcontrôleur, sans oublier »*.
- **Affiché** : page de titre (sobre, logos institutions).
- **Transition** → « De quel problème industriel part-on ? »

## Slide 2 — Problématique & objectif : le triple gap (2 min)

- **Éléments** : maintenance prédictive embarquée → notion de *capteur intelligent* (le modèle
  vit sur le capteur, pas dans le cloud) · pourquoi apprendre *sur place* (drift, pas de réseau) ·
  l'objectif du stage = porter 3 méthodes de Continual Learning de Python vers le C sur MCU ·
  positionnement scientifique = **triple gap** (Gap 1 données industrielles réelles ·
  Gap 2 < 100 Ko RAM mesurés · Gap 3 quantification INT8 pendant l'apprentissage).
- **Affiché** : [`docs/figures/presentation_board/07_sprint_timeline.png`](../figures/presentation_board/07_sprint_timeline.png)
  comme **fil rouge** du stage (Phase 1 PC → Phase 2 MCU).
- **Transition** → « Avant le code embarqué : qu'est-ce que le Continual Learning et pourquoi c'est dur ? »

## Slide 3 — État de l'art CL condensé (4 min)

- **Éléments** : les 3 défis du CL —
  1. *Distribution drift* (l'usure/l'environnement déplacent la distribution des données) ;
  2. *Oubli catastrophique* (apprendre la tâche B dégrade brutalement la tâche A) ;
  3. *Compromis stabilité-plasticité* (front de Pareto retenir ↔ apprendre).
  Puis la **taxonomie en 3 familles** : Régularisation (EWC) · Rejeu (QLR-CL, LifeLearner — **écarté** :
  ~1.3 Mo RAM > 256 Ko de la carte) · Architecture (TinyOL, HDC).
- **Affiché** : [`10_forgetting_intuition.png`](../figures/presentation_board/10_forgetting_intuition.png)
  (courbes accuracy avec/sans EWC → illustre l'oubli).
- **Transition** → « Concrètement, quels modèles ai-je retenus et pourquoi ? »

## Slide 4 — Les 3+1 modèles retenus (3 min)

- **Éléments** : tableau modèles —
  - **EWC** (régularisation) : MLP + pénalité de Fisher diagonale, λ règle stabilité↔plasticité.
  - **HDC** (architecture, non-neuronal) : hypervecteurs binaires accumulés, pas de backprop.
  - **TinyOL** (architecture) : backbone auto-encodeur **gelé** en Flash + petite tête OtO entraînée en ligne.
  - **Mahalanobis** (baseline non-supervisée) : distance à la distribution, mise à jour incrémentale (Welford).
  Critère commun : tenir dans 256 Ko, FP32, online.
- **Affiché** : tableau modèles (M1–M4, repris de CLAUDE.md) +
  [`04_ewc_lambda_impact.png`](../figures/presentation_board/04_ewc_lambda_impact.png) (effet de λ).
- **Transition** → « Sur quelles données réelles les a-t-on validés ? »

## Slide 5 — Données industrielles (2 min)

- **Éléments** : 6 datasets de maintenance prédictive (Pump, Equipment Monitoring, **CWRU**,
  **Pronostia/FEMTO**, **CMAPSS**, **Paderborn**) · pipeline d'extraction de features fréquentielles/
  temporelles (RMS, kurtosis, facteur de crête, skewness…) · scénarios CL (domain- / class-incremental,
  ex. *by_equipment* : pompe → turbine → compresseur).
- **Affiché** : tableau datasets (type · label · scénario CL).
- **Transition** → « Que donnent les modèles sur PC, avant même de toucher au C ? »

## Slide 6 — Résultats CL sur PC (3 min)

- **Éléments** : comportement comparé des algos · **front de Pareto accuracy/oubli** ·
  validation cross-dataset = **Gap 1 comblé** sur roulements réels (Pronostia : EWC AA 0.982,
  oubli AF 0.000, BWT +0.005).
- **Affiché** : [`pareto_acc_forgetting.png`](../figures/pareto_acc_forgetting.png) +
  [`gap1_gap2_heatmap_acc.png`](../figures/gap1_gap2_heatmap_acc.png) +
  vision optimale par modèle : [`gap1_heatmap_acc_best_pc.png`](../figures/gap1_heatmap_acc_best_pc.png)
  et [`gap1_heatmap_f1_best_pc.png`](../figures/gap1_heatmap_f1_best_pc.png).
- **Message clé** : l'**accuracy seule est trompeuse** sur la détection de panne déséquilibrée —
  la heatmap **F1 (classe faulty)** montre où la classe `faulty` s'effondre (Mahalanobis CWRU F1≈0,25
  malgré une accuracy « correcte » ; cf. Sprint 26, F1_MC=0,243).
- **Transition** → « Le défi principal du stage commence ici : faire tourner tout ça sur le MCU. »

---

## Slide 6bis — Impact du nombre de features (board) — Sprint 35 (2 min)

- **Éléments** : 3 conditions de features comparées **sur la carte** — `5feat` (référence figée
  Sprint 32), `all` (dims natives du dataset), `best` (sous-ensemble optimal par modèle, permutation
  importance, k optimisé sur F1 val), **F1 ET accuracy**.
- **Affiché** : panel board [`gap1_heatmap_f1_{5feat,all,best}_board.png`](../figures/) +
  [`gap1_heatmap_acc_{5feat,all,best}_board.png`](../figures/).
- **Message clé** : le **5-feat board** est un bon compromis perf/RAM ; passer à `best`/`all`
  améliore surtout EWC sur cmapss (F1 board 0,38→0,62 en `5feat`→`all`, dims natives 21).
  **Footnote** « board = 5 features » + correction **HDC×monitoring 0,113 → 0,867** (artefact
  zéro-padding 4→5 levé).
- **Transition** → « Le défi principal du stage commence ici : faire tourner tout ça sur le MCU. »

---

## Slide 7 — CŒUR : Portage C sur NUCLEO-F439ZI (3 min)

- **Éléments** : la cible **NUCLEO-F439ZI** (Cortex-M4 @ 180 MHz, 256 Ko SRAM, FPU matériel,
  **pas de NPU** → forward *et* backprop en FP32 sur le CPU) · architecture firmware :
  `pipeline.c` orchestrateur + têtes modèles (`ewc_head.c`, `tinyol.c`, `mahalanobis.c`, `hdc.c`) ·
  contrainte de conception : **allocation 100 % statique** (`.bss`), **zéro `malloc`**, tailles via `#define`.
- **Affiché** : [`08_firmware_architecture.png`](../figures/presentation_board/08_firmware_architecture.png) +
  [`06_memory_breakdown.png`](../figures/presentation_board/06_memory_breakdown.png) (Flash vs SRAM par modèle).
- **Transition** → « Comment le PC alimente-t-il la carte en données capteur ? »

## Slide 8 — CŒUR : Protocole UART PC↔carte (3 min)

- **Éléments** : `sensor_stream.py` rejoue le flux capteur en série (USART3, **115 200 bauds**) ·
  cycle complet en **6 étapes** : (1) RX par interruption + buffer circulaire DMA → (2) décodage
  float32 + validation **CRC8** → (3) routage modèle selon la commande (0x01 Mahalanobis / 0x02 EWC /
  0x03 TinyOL) → (4) inférence **+ mise à jour CL** si flag `UPDATE` + métriques en ligne → (5) TX DMA →
  (6) LED (verte OK / rouge erreur CRC / clignotante watchdog) · évolution des trames **v2 → v3**
  (v3 embarque les métriques CL calculées *sur la carte* : ACC, AUROC, FORGET).
- **Affiché** : [`09_uart_protocol.png`](../figures/presentation_board/09_uart_protocol.png)
  (trame requête 32 B / réponse v3 21 B / PAIR 22 B / TRIPLE 27 B).
- **Transition** → « Maintenant qu'on mesure tout, tient-on les budgets matériels ? »

## Slide 9 — CŒUR : Profiling & Gap 2 (3 min)

- **Éléments** : mesure de latence par compteur de cycles matériel **DWT CYCCNT** (précision au cycle) ·
  empreinte RAM par modèle (Mahalanobis ~200 B · EWC ~9.7 Ko · TinyOL ~600 B + 5.8 Ko Flash) et
  firmware complet multi-modèle (**104.6 Ko / 256 Ko**) · latences board mesurées (5 µs Maha →
  657 µs paire Maha+HDC) vs **budget 100 ms** · **débit** vs cadence capteur typique (1 kHz) → **Gap 2 comblé**.
- **Affiché** : [`01_ram_budget.png`](../figures/presentation_board/01_ram_budget.png) ·
  [`02_latency.png`](../figures/presentation_board/02_latency.png) /
  [`03_latency_log.png`](../figures/presentation_board/03_latency_log.png) ·
  [`12_throughput.png`](../figures/presentation_board/12_throughput.png) ·
  [`11_gap2_compliance.png`](../figures/presentation_board/11_gap2_compliance.png).
- **Transition** → « Au-delà des budgets : le modèle apprend-il *correctement* sur la carte réelle ? »

## Slide 10 — CŒUR : Résultats board réels (3 min)

- **Éléments** :
  - **exp_S26** — EWC en **régression RUL** sur CMAPSS, board réel : **RMSE = 21.23** (ratio board/PC = **0.94** ✅),
    latence déterministe (130 µs inférence / 403 µs inférence+update), `.bss` = **66.7 Ko / 256 Ko (26 %)**.
  - **exp_S27** — **DUAL_MODE** : une seule trame UART → EWC_REG (RUL) + EWC_MC (faute) en séquence,
    latence combinée **637 µs ≪ 100 ms** (Gap 2 ✅), RMSE_RUL préservé (22.6).
  - **Honnêteté scientifique** : le F1 faute faible n'est **pas un bug de portage** — **parité numérique
    board↔PC exacte** ; cause = limitation modèle (oubli catastrophique / features mixtes), hors périmètre portage.
- **Affiché** : [`05_all_experiments_comparison.png`](../figures/presentation_board/05_all_experiments_comparison.png).
- **Transition** → « On peut aller plus loin : faire collaborer plusieurs modèles sur la même carte. »

## Slide 10bis — CŒUR : Paires de modèles + méta-modèle (2 min)

- **Éléments** :
  - **exp_S30** — **paires parallèles** Mahalanobis + un supervisé (PAIR_MODE) : sur CWRU, l'**ensemble**
    (règle OR) atteint F1 **0.991** (Maha seul 0.379, EWC seul 1.000) ; latences board combinées
    **256 µs** (Maha+EWC) / **657 µs** (Maha+HDC), overhead de co-exécution ≈ 0.
  - **exp_S31** — **méta-modèle de stacking** (logreg arbitrant les sorties de la paire, TRIPLE_MODE) :
    F1 **0.997** (≥ ensemble, +0.006), latence **258 µs** (méta logreg ≈ négligeable),
    `.bss` = **104.6 Ko / 256 Ko (39.9 %)** invariant au mode, **parité méta board↔PC = 1.000** (300 échantillons).
- **Affiché** : [`13_pairs_meta_results.png`](../figures/presentation_board/13_pairs_meta_results.png).
- **Transition** → « Dernière brique : peut-on quantifier en INT8 *pendant* l'apprentissage ? »

## Slide 11 — CŒUR : Gap 3, quantification INT8 (2 min)

- **Éléments** : benchmark **INT8 vs FP32** (4 modèles × 5 datasets, exp_S28) · gains RAM :
  **HDC ×2.33**, **EWC / Mahalanobis ×4.00** · métrique **préservée** (EWC Δ≤0.006 · HDC Δ=0) ·
  cas Mahalanobis dégradé (grande dynamique de Σ⁻¹) → **fallback Q15 recommandé**.
- **Affiché** : [`14_int8_benchmark.png`](../figures/presentation_board/14_int8_benchmark.png).
- **Transition** → « Récapitulons sur les trois gaps. »

---

## Slide 12 — Synthèse triple gap (1 min)

- **Éléments** : les 3 gaps cochés —
  Gap 1 ✅ (CWRU, Pronostia, CMAPSS, Paderborn réels) ·
  Gap 2 ✅ (< 100 Ko RAM + chiffres DWT précis) ·
  Gap 3 ✅ (INT8 pendant l'entraînement, Python + C, validé board) ·
  robustesse logicielle : **~96 tests Unity + pytest**, parité board↔PC vérifiée (jusqu'à 1.000).
- **Affiché** : récap triple gap (3 colonnes, statut ✅).

## Slide 13 — Perspectives (1 min)

- **Éléments** : **étude de sensibilité au seuil RUL→`faulty`** (Sprint 32 en cours : impact du seuil sur
  le ratio de positifs et la performance, invariance HW confirmée) · **INT8 sur firmware board** (Gap 3
  embarqué de bout en bout) · **généralisation multi-feature** (firmware câblé 5 features → dims natives
  variables) · nouveaux datasets (batteries Li-ion).
- **Affiché** : [`07_sprint_timeline.png`](../figures/presentation_board/07_sprint_timeline.png) (Sprint 32 en cours).
- **Clôture** → remerciements + questions.

## Annexe — Slides de secours (Q&R)

- Détail des trames v3/PAIR/TRIPLE (octet par octet) · architecture détaillée `ewc_head.c` (Fisher + SGD) ·
  liste des 6 datasets + features extraites · couverture de test (~96 Unity + pytest) ·
  empreinte mémoire détaillée par modèle (Flash + SRAM) · choix toolchain (arm-none-eabi-gcc, OpenOCD, CMake, Unity CI).
