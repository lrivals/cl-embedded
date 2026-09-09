# Triple Gap — Positionnement scientifique du projet

> Ce document formalise le positionnement original du stage.  
> Il doit être consulté avant toute décision d'architecture ou d'expérimentation.

---

## Définition du triple gap

Le triple gap désigne l'absence simultanée, dans la littérature existante, de travaux qui :

| Gap | Critère | Status de la littérature (corpus 20 articles, avril 2026) |
|-----|---------|----------------------------------------------------------|
| **Gap 1** | Validation sur données industrielles de séries temporelles réelles avec protocole reproductible | ❌ Aucun article ne satisfait ce critère |
| **Gap 2** | Démonstration d'un CL complet sous 100 Ko RAM avec chiffres précis mesurés par composant | ❌ Aucun article ne satisfait ce critère |
| **Gap 3** | Quantification INT8 appliquée à la phase d'entraînement incrémental (backpropagation) | ❌ Aucun article ne satisfait ce critère |

> **La contribution originale du stage** est d'être le premier travail à adresser ces trois gaps simultanément.

---

## Mapping corpus → triple gap

| Article | Gap 1 | Gap 2 | Gap 3 | Score |
|---------|:-----:|:-----:|:-----:|:-----:|
| TinyOL (Ren et al., 2021) | ❌ | ❌ | ❌ | 0/3 |
| QLR-CL (Ravaglia et al., 2021) | ❌ | ❌ | ⚠️ buffer UINT8 | 0/3 |
| LifeLearner (Kwon et al., 2023) | ❌ | ⚠️ 212 Ko | ❌ | 0/3 |
| EWC (Kirkpatrick et al., 2017) | ❌ | ❌ | ❌ | 0/3 |
| HDC-EMG (Benatti et al., 2019) | ⚠️ EMG (pas PdM) | ✅ < 4 Ko | ❌ | ~1/3 |
| CL×PdM (Hurtado et al., 2023) | ⚠️ PdM mais datasets divers | ❌ | ❌ | 0/3 |
| Gradient Monitoring (Shah et al., 2025) | ⚠️ RUL industriel | ❌ | ❌ | ~0.5/3 |
| Adaptive CL (Wu et al., 2025) | ⚠️ Séries temporelles industrielles | ❌ | ❌ | ~0.5/3 |
| Dataset Distillation (Rüb et al., 2024) | ❌ | ❌ | ❌ | 0/3 |
| AR1* (Pellegrini et al., 2021) | ❌ | ❌ | ❌ | 0/3 |

**Constat** : aucun article ne dépasse 1/3. Plusieurs articles adressent partiellement le Gap 1 (données industrielles), mais aucun ne combine les trois.

---

## Contribution de ce projet au triple gap

### Gap 1 — Données industrielles

✅ **RÉSOLU — 24 avril 2026** : exp_050–055 (FEMTO PRONOSTIA IEEE PHM 2012, by_condition, 3 tâches domain-incremental).

Premier résultat CL publié sur données industrielles réelles de roulements :

- EWC : AA=0.982, AF=0.000, BWT=+0.005, RAM=1.1 Ko
- HDC : AA=0.805, AF=0.045, RAM=14.2 Ko
- TinyOL : AA=0.930, AF=0.020, RAM=3.7 Ko
- KMeans : AA=0.890, AUROC=0.855, RAM=5.4 Ko
- Mahalanobis : AA=0.793, AUROC=0.782, RAM=1.7 Ko
- DBSCAN : AA=0.901, AUROC=0.825, RAM=118 Ko ⚠️

Protocole reproductible : seed=42, `config_snapshot.yaml`, loader `pronostia_dataset.py` validé par 18 tests unitaires (dont 2 intégration sur vrais `.npy`).

**Renforcement Sprint 35 — métrique F1 et choix des features** : la validation Gap 1 est désormais
quantifiée sur **F1 (classe faulty) ET accuracy**, pour 5 datasets × 4 modèles × 3 conditions de
features (`5feat` / `all` / `best`), PC et board (`exp_S35_*`, 12 heatmaps
`docs/figures/gap1_heatmap_{metric}_{condition}_{platform}.png`). Deux apports : (1) l'accuracy seule
est trompeuse en détection déséquilibrée — ex. Mahalanobis × cmapss PC accuracy 0,745 mais F1=0,269 —
le F1 est donc le juge ; (2) le choix des features compte (EWC × cmapss F1 board 0,38→0,62 en
`5feat`→`all`). Artefact HDC×monitoring (acc 0,113, zéro-padding) corrigé à **0,867** (valeur board
réelle, monitoring natif 4-feat). Cf. `docs/sprints/sprint_35/S3512_analysis_update.md`.

~~**Adressé partiellement** par les deux datasets Kaggle (simulés mais industriellement motivés). Référence scientifique : FEMTO PRONOSTIA (Nectoux et al., 2012) dans le manuscrit.~~

~~**Limitation honnête** : les datasets Kaggle sont synthétiques. Le manuscrit mentionnera explicitement cette limitation et positionnera FEMTO PRONOSTIA comme la cible expérimentale de la Phase 2 du stage (post-avril 2026).~~

### Gap 2 — Sub-100 Ko RAM avec chiffres précis

**Adressé** : les trois modèles sont estimés à < 15 Ko en RAM. Le profiling systématique via `tracemalloc` + mesures MCU produira les premiers chiffres précis par composant dans la littérature.

**Métrique clé** : `ram_peak_bytes` dans `evaluation/memory_profiler.py`.

**Renforcement Sprint 36 — latences EWC inférence vs inférence+MAJ CL (board réelle NUCLEO-F439ZI)** :
comparaison appariée PC↔board du modèle EWC sur Pronostia (D4) et Monitoring (D2), conditions
`5feat`/`all`, mesurées par DWT et séparées en deux passes (cf. Sprint 26) :

| Cellule | k | lat **inférence** P50 | lat **inférence+MAJ CL** P50 | Δ MAJ | `.bss` | Gap 2 |
|---------|---|----------------------|------------------------------|-------|--------|-------|
| 5feat·monitoring | 4 | 48 µs | 239 µs | +191 µs | 100 152 B | ✅ |
| all·monitoring | 4 | 48 µs | 239 µs | +191 µs | 100 152 B | ✅ |
| 5feat·pronostia | 5 | 50 µs | 251 µs | +201 µs | 105 036 B | ✅ |
| all·pronostia | 13 | 65 µs | 340 µs | +275 µs | 144 516 B | ✅ |

Toutes les latences (inférence 48–65 µs ; inférence+MAJ CL 239–340 µs) sont **≪ 100 ms** ⇒ **Gap 2
préservé** y compris pour l'apprentissage en ligne (réponse à `FIXME(gap2)` `all` Pronostia 13 feat
+ passe online). Surcoût MAJ CL +191…+275 µs cohérent avec Sprint 26 (130 µs inf vs 403 µs inf+MAJ).
Détail : `experiments/exp_S36_summary.json`, `docs/sprints/sprint_36/`.

**Renforcement Sprint 38 — latence du gate vs SGD permanent (board réelle NUCLEO-F439ZI)** : la mise à
jour EWC **autonome** (gate Mahalanobis + fenêtre glissante, `-DEWC_AUTO_UPDATE`) ajoute un **coût
constant par échantillon** (maha_score + drift_update ≈ **27 µs**, `gate_overhead_us`) mais **économise
les pas de SGD** sur les échantillons NORMAL. Résultat : la latence **moyenne** des politiques gatées
(**79–82 µs**, `update_rate ≈ 0.025`) est **bien inférieure** à `always` (SGD à chaque échantillon :
**238–251 µs**), toutes **≪ 100 ms** ⇒ **Gap 2 préservé**. Le gate économise ~97 % des mises à jour pour
une latence moyenne ~3× plus faible que `always`. Détail : `experiments/exp_S38_summary.json`
(`economy_table`), `docs/sprints/sprint_38/`.

**Renforcement Sprint 45 — détecteurs de drift portés (board réelle NUCLEO-F439ZI)** : les détecteurs
de drift (Page-Hinkley/DDM O(1), PSI O(bins)) sont portés en C sous `-DDRIFT_DETECT` (sélection à la
compilation, wire format V3 inchangé). **Colonne `gas_sensor_drift` mesurée** (128 features, 13 910
échantillons, seed 42, 0 CRC) : **Page-Hinkley et DDM** — latence DWT **270 µs** (P50 ≈ P99) **≪ 100 ms
⇒ Gap 2 préservé**, **parité verdict board↔PC = 1.000** (0 mismatch / 13 910 chacun) : le board décide
exactement comme le Python. **Coût `.bss` du détecteur** : **+36 B** (Page-Hinkley) / **+40 B** (DDM) /
**+132 B** (PSI, histogramme (3·bins+1)·4) sur le build par défaut invariant (105 036 B) — négligeable
dans le budget 256 Ko. **PSI × gas_sensor_drift = N/A honnête (limite Gap 3 mesurée)** : PSI est piloté
à bord par le score Mahalanobis (`signal ← maha_score`), dont la covariance est **O(k²)** ; à k=128
features, `sigma_inv` (128²×4 ≈ 64 Ko) fait **déborder la SRAM** au link (`.bss` overflow ~69 Ko) →
**PSI n'est portable qu'en basse dimension** (le goulot est sa source de signal, pas l'état O(bins) du
détecteur). **Écart proxy-PC ↔ board** : le proxy Python S44 (DDM ≈ 6 µs/update) **n'est pas prédictif**
de la latence board (270 µs, chemin d'inférence EWC dominant, paradoxe FPU S29) — seule la mesure board
fait foi. Agrégat `experiments/exp_S45_summary.json` (`aggregate_sprint45.py`, mesuré-board vs
proxy-PC), `exp_S45_board_*`, `exp_S45_parity_*`, `docs/sprints/sprint_45/`.

**Correction Sprint 49 — RAM rapportée = `.data + .bss + pic de pile`** : les mesures antérieures ne
remontaient que `.bss`, ce qui **sous-estimait** la RAM (la pile vit hors `.bss`). Le CR du 16 juillet
2026 fixe la formule officielle **`RAM totale = .data + .bss + pic de pile`**, généralisée à toutes les
expériences et mesurée **par phase réelle** (idle / inférence / mise à jour CL) via le stack painting
existant (`profiling.c`). **32 cellules board+PC** (`experiments/exp_S49_ram/`, agrégat `summary.json`)
confirment l'invariant `total = .data + .bss + max(pic_inf, pic_upd)` et le fait que **la mise à jour CL
creuse plus la pile que l'inférence** (`pic_update ≥ pic_inference`, marqué chez EWC — SGD backward).
Totaux board **40–42 % de 256 Ko ⇒ Gap 2 toujours largement préservé**, même en comptant la pile. Doc
structurée : `docs/context/ram_report.md` (généré), figures `docs/figures/ram_full/`, `docs/sprints/sprint_49/`.

### Gap 3 — INT8 pendant l'apprentissage incrémental (mis à jour Sprint 29)

**Critère** : ΔAUROC < 0.02 (métrique préservée) **ET** réduction RAM pendant l'entraînement incrémental INT8.

✅ **RÉSOLU multi-modèle (Sprints 22–29)** : quantification INT8 validée Python (PC) sur 4 modèles × 5 datasets
et portée/mesurée sur NUCLEO-F439ZI réelle (EWC, HDC, TinyOL). Premier travail à mesurer ce compromis sur MCU
avec continual learning.

| Modèle | Datasets testés | Δmétrique PC (max \|Δ\|) | RAM ratio (PC / board) | Latence board INT8/FP32 | gap3_metric | gap3_ram |
|--------|-----------------|:----------------------:|:----------------------:|:-----------------------:|:-----------:|:--------:|
| **EWC INT8** | CMAPSS, CWRU, Monitoring, Pronostia, Paderborn | 0.006 (cmapss) | ×4.0 / ×2.70 | ×1.84 ❌ | ✅ | ✅ |
| **HDC INT8** | CMAPSS, CWRU, Monitoring, Pronostia (Paderborn N/A) | 0.000 | ×2.33 / ×3.06 | ×3.26 ❌ | ✅ | ✅ |
| **TinyOL INT8** | CMAPSS, CWRU, Monitoring, Pronostia | +0.020 / +0.054 (améliorations) | ×3.5–3.8 / ×4.00 | ×0.56 ✅* | ⚠️ amélioration | ✅ |
| **Mahalanobis INT8** | CMAPSS, CWRU, Monitoring, Pronostia (PC seulement) | −0.236 / −0.238 ❌ | ×4.0 / — | — | ❌ → fallback Q15 | ✅ |

\* TinyOL board : INT8 plus rapide car chemins **non iso-calcul** (FP32 = autoencodeur encode+decode+MSE ;
INT8 = encodeur + tête OtO linéaire, pas de décodeur) — pas une exception au principe latence ci-dessous.

> **Résultat clé — Latence sur Cortex-M4 FPU (mesuré Sprint 23 + confirmé Sprint 29)**
> L'INT8 est **plus lent** que FP32 sur Cortex-M4 FPU : le FPU exécute les opérations FP32 en 1 cycle, tandis
> que les opérations INT8 scalaires enchaînent `LDRSB` + multiplication entière sans parallélisme SIMD.
> Ce résultat négatif est une contribution honnête du projet : **aucun travail précédent ne l'avait mesuré
> sur MCU avec continual learning**. La réduction RAM (×2.33–4.0) reste un résultat positif solide.
> La cible future pour un speedup INT8 serait le **Cortex-M55** (extension Helium MVE, SIMD vectoriel) ou un NPU.

**Limitation Mahalanobis** : `sigma_inv_` a une dynamique trop large pour l'INT8 (dégradation −0.24 AUROC sur
CWRU/Pronostia) → **fallback Q15 recommandé** (`TODO(arnaud)` S2805).

**Renforcement Sprint 36 (rework) — EWC INT8 board apparié, frozen + online** : comparaison focalisée INT8 vs
FP32 de la tête EWC sur Pronostia + Monitoring × `5feat`/`all`, dans les **mêmes conditions** que la comparaison
FP32 board↔PC du Sprint 36. Le firmware **résout `TODO(dorra)`** (`ewc_int8_from_fp32(&g_ewc_int8, &g_ewc_head)`
après `ewc_head_load_or_init` — le chemin 0x40 tournait jusque-là sur une tête Xavier non entraînée ; **0
régression FP32**). RAM des poids ÷4 structurel (`gap3_ram_ok`), latence INT8 sur FPU **non accélérée** (cohérent
avec le résultat clé ci-dessous), accord INT8↔FP32 board. **8 cellules mesurées board réelle (0 CRC)** :
Gap 2 ✅ (frozen 51–68 µs, online 440–639 µs ≪ 100 ms ; MAJ online INT8 ~2× FP32 = non accélérée FPU) +
RAM ×4.0 ✅, **mais métrique NON préservée** : F1 INT8 **0.07–0.15** ≪ FP32 board ≈ 0.92 (accord INT8↔FP32
0.60–0.74 frozen). La **PTQ embarquée** de la tête EWC binaire dégrade fortement — **cohérent avec le board
Sprint 29** (INT8 EWC AUROC 0.25 vs 0.63) et **distinct du fake-quant QAT PC** (Sprint 28, Δ≤0.006 préservé).
Conclusion honnête : pour EWC, la quantif INT8 *post-training* embarquée ne satisfait **pas** le critère « ΔAUROC
< 0.02 » côté board (≠ HDC INT8 Δ=0) → piste QAT exporté ou Q15 (cf. Mahalanobis Sprint 34). Détail :
`docs/sprints/sprint_36/S3610_int8_fp32_board.md`.

**Renforcement Sprint 39 (Partie A, PC + host) — la perte est corrigeable, cause isolée** : un émulateur
Python bit-exact du chemin C (`src/utils/int8_c_emulation.py`) reproduit la dégradation board **sans flasher**
et permet une ablation chiffrée (`exp_S39_ablation/`). **Cause racine = l'échelle `1/128` non calibrée**
(dominant `per_tensor_calib`, jusqu'à **+0.88 F1**), **pas** l'accumulateur `int16` seul (`fix_acc32` marginal,
et sur Monitoring il *dégrade* transitoirement avant recalibration — l'échelle d'ablation n'est donc **pas**
monotone bout-en-bout). Le sweep `exp_S39_quant_sweep/` confirme : EWC `int8_legacy` s'effondre (monitoring
0.027 / pronostia 0.045) → **`int8_perchannel` récupère ≈ FP32** (0.915 / 0.944), Q15/mixte idem ; Maha INT8
0.77 → **Q15 0.923**. Le kernel C **v2** (`ewc_head_int8_v2.c`, acc int32 + scales par-canal calibrés) est
validé **host** (`make test`, S3909) contre les golden vectors de l'émulateur (parité par construction) — v1
laissé intact pour l'A/B board (S3916). **Bug supplémentaire trouvé & corrigé en Q15** : l'accumulateur int32
déborde (int16×int16 sommé > 2³¹) → `ewc_v2_acc_t` int32 (int8) / **int64 (Q15)**.

**Confirmé sur board réelle (Sprint 39, Partie B — S3915/S3916/S3919, 1er juil. 2026)** : le kernel v2 est
câblé au pipeline par **sélection de compilation `-DEWC_INT8_V2`** (nibble protocole saturé → mirroir
`-DMAHA_INT8` ; le chemin 0x40 route vers `ewc_int8_v2_forward`, **wire UART inchangé**, `.bss` v1 défaut
105 036 B invariant → 0 régression). Sur NUCLEO-F439ZI (`run_s39_board.py`, stream gelé sans `--update`),
**le v2 récupère bien la F1 mesurée matériellement** : pronostia **0.078 (v1) → 0.928 (per-canal) / 0.970
(Q15)**, cmapss **0.133 → 0.400** ; **parité gelée bit-exacte board↔émulateur = 1.000 (0 mismatch)** sur les
5 cellules (la parité host **et** silicium sont maintenant prouvées) ; latence P50 67–75 µs ≪ 100 ms
(**Gap 2 ✅** ; coût +14–22 µs vs v1 = déquant→FP32 sur FPU, cohérent S29) ; `.bss` +1.1–1.8 Ko (2ᵉ tête) ;
0 CRC. Côté PC, `run_s39_matched_compare.py` (S3918) garantit une comparaison *appariée* — le côté PC est
l'**émulateur du schéma board**, jamais le QAT S28 — et fournit la référence bit-à-bit confrontée par S3919.
**S3917 (bench SIMD CMSIS-NN)** reste différé (`TODO(dorra)`, non bloquant). Détail : `docs/sprints/sprint_39/`.

**Synthèse Sprint 40 (article) — récupération émulée établie, board v2 partielle/honnête** : l'article standalone
FR+EN (`docs/article/ewc_int8_mcu/`, S4004–S4007) formalise le fil Gap 3 « effondrement PTQ naïve → récupération
par kernel calibré » en séparant strictement **mesuré board** (S36 FP32+legacy) et **émulé PC bit-exact** (S39
ablation). Le résultat de récupération (`per_tensor_calib` +0.88 F1, Q15 = FP32) est **prouvé par émulation** et
**confirmé sur carte pour la cellule Pronostia per-canal** (S39 board + `exp_S40_board_v2`) ; la grille board v2
complète reste explicitement **« à mesurer »** (règle « aucun chiffre inventé »). Tant que la campagne carte v2
n'est pas complète, le critère « RAM ÷4 sans perte de métrique sur MCU réel » est donc **confirmé par émulation +
un point board**, pas encore généralisé — l'axe honnête émulateur reste la formulation de référence.

**Mise à jour Sprint 40 (refonte S4008–S4010) — la récupération INT8 passe d'« émulée » à « mesurée carte »** :
la campagne board v2 a streamé les **4 cellules `per_channel`** (2 jeux × gelé/en ligne) : F1 **0.9173**
(Monitoring) et **0.8995** (Pronostia) contre 0.9194 / 0.9164 en FP32, **parité gelée 1.000** contre
l'émulateur bit-exact, accord INT8↔FP32 0.9996 / 0.9951, **0 erreur CRC**. Le critère Gap 3 « RAM des poids
÷4 sans perte de métrique sur MCU réel » n'est donc plus adossé à une émulation plus un point isolé : il est
**mesuré sur les deux jeux**. Les 8 cellules restantes (`q15`, A/B `int8_legacy` sous kernel v2) restent
« à mesurer » (`S4010_mesures_manquantes.md`).

Trois nuances mesurées accompagnent ce renforcement, et empêchent de sur-vendre l'INT8 :

1. **RAM des poids ÷4 ≠ RAM système.** La RAM totale (`.data` + `.bss` + pic de pile, Sprint 49) est
   **inchangée** : 105 300 → 105 324 octets sur Monitoring, soit un ratio INT8/FP32 de **1.0002**, à 40 %
   du budget de 256 Ko. À cette échelle de modèle les poids ne sont pas le poste dominant.
2. **Paradoxe de latence, chiffré** (Sprint 50) : 74 µs en INT8 contre 48–50 µs en FP32, la
   **requantification** pesant ≈ 3 777 cycles (un tiers du budget) et le MAC entier n'apportant rien face
   au FPU. Le ratio théorique de **16** en BOPs (S4008) n'est donc pas restitué par le matériel.
3. **Paradoxe d'énergie** (Sprint 53) : 67.47 µJ/inférence en INT8 contre 67.65 en FP32, écart de 0.18 µJ
   pour une incertitude combinée de 2.34 µJ — **aucun effet mesurable**. Le levier énergétique est la
   fréquence, pas le format.

Conclusion Gap 3 en une phrase : sur cette cible, **l'INT8 se justifie par la mémoire des poids, ni par la
vitesse ni par l'énergie** — et le gain sub-INT8 n'existe qu'à condition d'un bit-packing réel (Sprint 48).

**Renforcement Sprint 46 — les trois *moments* de quantification comparés frontalement (PC + board réelle)** :
là où les sprints précédents ont établi le QAT (S28), la PTQ effondrée puis récupérée (S36/S39) et Q15
(S34) de façon **dispersée**, le Sprint 46 les met côte à côte à modèle/dataset/seed fixés, sur **EWC** puis
**TinyOL** × **Monitoring/Pronostia**, selon trois moments : **avant** l'entraînement (QAT / fake-quant),
**après** (PTQ sur FP32 figé), et **les deux** (QAT → export PTQ = le chemin réel du firmware). Message :
*le moment et la calibration dominent la préservation de métrique* ; `before` (fake-quant à l'inférence)
est une **borne haute** que la carte n'atteint pas, `both` (noyau entier) est la seule variante **fidèle au
déploiement**. Cadrage honnête : **HDC** (natif entier, INT8≡FP32 structurel) et **Mahalanobis** (PTQ-only,
axe INT8-vs-Q15) sont documentés en **contexte N/A**, sans cellule 3-way artificielle. Harnais
`scripts/run_s46_quant_moment.py` réutilise `EWCMlpInt8Classifier` + `int8_c_emulation.py` et **câble le
seul maillon manquant** (QAT→`from_state_dict`→`forward_quant`) → `experiments/exp_S46_{ewc,tinyol,context}/`.
**Colonne `both` mesurée sur carte réelle NUCLEO-F439ZI (S4608)** : réconciliation d'architecture — le head
firmware étant multiclasse 2 sorties, un **head QAT multiclasse** (`EWCMlpMulticlassInt8`, nouveau) est
entraîné puis exporté vers le kernel v2 calibré (`-DEWC_INT8_V2`, driver `run_sprint46_board.py`). Résultats
board (frozen, 5feat) : **F1 `both` = 0.9213 (Monitoring) / 0.9072 (Pronostia)**, **parité board↔émulateur
= 1.000** (0 mismatch, par construction), **latence DWT 65 / 68 µs ≪ 100 ms (Gap 2 ✅)**, **`.bss` 101 236 /
106 152 B — RAM poids ÷4 (Gap 3 ✅)**, **0 CRC**. **A/B `both` ≥ `after`** (source FP32,
`experiments/exp_S40_board_v2`) : **+0.004 / +0.008** — le QAT préserve la métrique et **égale** (marginalement
au-dessus) la PTQ calibrée sur ce head : sur la NUCLEO, c'est la **calibration du noyau v2** qui récupère
l'essentiel, le QAT n'ajoutant pas de gain décisif au-delà (constat honnête, pas d'effet inventé). Détail :
`docs/sprints/sprint_46/`.

**Renforcement Sprint 47 — profondeur & schéma de quantification (jusqu'où descendre en bits)** : troisième
axe, orthogonal au *moment* (S46) et au *format* (S34), balayé **EWC-only × {Monitoring, Pronostia}, PC-only**
via l'émulateur bit-exact `int8_c_emulation.py` (28 cellules profondeur `exp_S47_depth/` + 12 cellules symétrie
`exp_S47_symmetry/`). **Jusqu'où descendre à AUROC préservée (Δ ≥ −0.02, per_channel symmetric)** : **Monitoring**
tient jusqu'au **binaire** (Δ=−0.0117 ; int2 Δ=−0.0069) ; **Pronostia** casse au binaire (Δ=−0.0275) mais le
**ternaire** tient (Δ=−0.0153). La **granularité per-channel repousse le cliff** (Pronostia 2-bit : per_tensor
Δ=−0.046 → per_channel Δ=−0.009), tandis que le **zero-point affine ne rachète rien** (S4704 : gain ≤ 0 sur les
6 cellules critiques). **Gain RAM = THÉORIQUE (bit-packé)** : ÷4 (8b) → ÷8 (4b) → ÷16 (2b) → ×20.25 (ternaire) →
×32 (binaire) — un poids INT4 logé dans un `int8_t` n'économise rien de plus que l'INT8 ; **la `.bss` réelle et
la latence sont hors de portée de l'émulateur et seront mesurées board au Sprint 48** (kernel bit-packé, avec/sans
packing). Configs retenues pour le portage (S4708, traçable aux JSON) : frontière **ternaire**, agressive
**binaire** (les 2 datasets), référence **int8** per_channel (déjà porté S39). Détail : `docs/sprints/sprint_47/`.

**Renforcement Sprint 48 — portage board sub-INT8 (RAM `.bss` réelle + latence dépacking, ce que l'émulateur ne
mesure pas)** : les schémas gagnants S47 (INT4/ternaire/binaire × Monitoring/Pronostia) sont **matérialisés sur
NUCLEO-F439ZI réelle** — kernel sub-INT8 câblé dans `pipeline.c` (route 0x40 gardée `EWC_SUBINT8_WEIGHTS_PROVIDED`,
`.bss` défaut invariant 105 036 B, 0 régression), packé (dépack→MAC FPU) et non-packé. **12 cellules mesurées, 0
CRC.** § **Gap 3 (RAM `.bss` réelle)** : le nœud d'honnêteté S47 est **confirmé par la mesure** — le `.bss`
**non-packé est invariant par mode** (Monitoring k=4 : 105 640 B ; Pronostia k=5 : 106 152 B, identiques INT4 ≡
ternaire ≡ binaire car conteneur `int8_t`), et **seul le bit-packing matérialise le gain** : Monitoring 336 B (INT4,
÷2) → 504 B (ternaire, ÷4) → 572 B (binaire, ÷8) ; Pronostia 336/504/604 B (le gain croît quand les bits baissent,
comme prédit théoriquement). L'écart théorie(÷8/÷16)↔`.bss` réelle = **overhead `.bss` fixe partagé** (le packing ne
réduit que les matrices de poids), exposé sans conflation. § **Gap 2 (latence dépacking)** : le dépacking ajoute
**≈ +55 µs** (67→123 µs) mais reste **≪ 100 ms** (P99 board deux ordres de grandeur sous le budget) → Gap 2 préservé
même à profondeur binaire. **Parité board↔émulateur = 1.000 sur les 12 cellules** (`max_score_err ≤ 1.2e-7`) : le
schéma de quantification est porté sans perte. **Clôt les deux `TODO(dorra)` S47** (kernel bit-packé + coût du
dépacking). Détail : `docs/sprints/sprint_48/`.

**Volet énergie (Sprint 33)** : le constat « INT8 réduit la RAM sans accélérer la latence FPU » ouvre une
question énergie potentiellement originale — l'INT8 réduit-il néanmoins les **µJ** (moins d'accès mémoire) ? La
chaîne de mesure est livrée et fonctionnelle : marqueurs de phase GPIO firmware (PA8, `ENERGY_MARKERS`, S3304),
driver PowerShield X-NUCLEO-LPM01A `scripts/energy_capture.py` (segmentation par phase + intégration µJ, S3305),
métriques de coût `compute_cost.py`/`hw_cost_model.py` (FLOPs/**BOPs**/FLOPS-W ; BOPs rend le gain INT8 quantitatif :
`BOPs_fp32/BOPs_int8 = (32/8)² = 16`), et autonomie `src/evaluation/autonomy.py` (Capacité/I_moy). **Réponse
**réponse mesurée (Sprint 50, S5008, 2026-08-04/05)** : le banc LPM01A a été monté et la grille **8/8 mesurée
board réelle** — mais en **courant moyen**, pas en µJ/inférence. **L'INT8 ne réduit PAS la consommation** : à
cadence imposée identique (100 Hz, 3 répétitions, dispersion ≤ 0,05 mA), EWC (−0,003 mA) et Mahalanobis
(−0,030 mA) sont dans le bruit, TinyOL gagne marginalement (−0,087 mA), et **HDC INT8 consomme 7,1 % de courant
en PLUS** que son FP32 (51,220 vs 47,843 mA) — cohérent avec sa latence INT8 dégradée. L'ordre des courants suit
exactement celui des latences (Maha 5 µs → 46,4 mA … HDC INT8 1958 µs → 51,2 mA), ce qui confirme que la part
imputable au modèle est bien mesurée. **L'INT8 reste donc justifié par la RAM (÷4), pas par l'énergie** — le
paradoxe latence FPU se double d'un paradoxe énergie.

**Ce qui reste `"à mesurer"`, et c'est un résultat, pas un manque** : les **µJ/inférence**. La référence « au
repos » du firmware (54,78 ± 0,13 mA) est **plus haute que tous les régimes de flux** (46,40–51,22 mA), y compris
Mahalanobis dont l'inférence occupe ~0,05 % du temps — écart que le taux d'occupation n'explique pas et dont la
**cause n'est pas établie** (le firmware attend la trame UART par scrutation active, donc le « repos » n'est pas
inactif ; d'autres causes de banc ne sont pas exclues). L'énergie marginale par inférence en ressortirait
**négative** : elle n'a pas de sens face à cette référence. Levier identifié pour un sprint futur : mise en
sommeil (`WFI`) de l'attente UART, pour disposer d'un vrai repos. **Autonomie chiffrée** malgré tout, depuis le
courant mesuré : 43,1 h (Maha INT8) à 39,0 h (HDC INT8) sur 2000 mAh — ~10 % d'écart entre le modèle le plus
sobre et le plus gourmand, l'ordre de grandeur qui sert effectivement à arbitrer. Synthèse :
`notebooks/cl_eval/energy_cost/comparison.ipynb` · détail `docs/sprints/sprint_50/S5008_handoff_mesures.md` § 6.

**Levée du verrou (Sprint 53, S5301–S5302, board réelle 2026-08-06) — les µJ/inférence existent enfin.**
Le paragraphe ci-dessus n'est plus la dernière ligne de l'histoire. S5301 a montré que l'écart « repos plus
consommateur que la charge » était un **artefact d'ordre** (référence prise en tête de session), et S5302 a
supprimé la cause de fond : l'attente UART dort désormais (`__WFI`, gardé `-DUART_WFI_IDLE`). Le repos tombe de
**49,767 à 27,370 mA — −45,0 %**, les deux mesurés dans la même session contre-balancée. Il passe alors **sous
toutes les cellules**, le delta redevient positif, et **8/8 cellules sont chiffrées** (134 à 374 µJ). Ces µJ-là
portent cependant la **trame UART** en plus du calcul — le témoin Mahalanobis, dont l'inférence dure 5 µs, sort à
134 µJ. La boucle par lot (`-DINFER_BATCH_N`, `N` jusqu'à 200 avec règle de saturation **mesurée** sur la cadence
atteinte) sépare enfin les deux : **EWC 5,08 µJ et Mahalanobis 0,437 µJ par inférence, r² = 0,9998**, parité de
prédiction 1.000 vs `N = 1`. Le coût d'une inférence rapide est donc **~300× plus petit que celui de la
transaction qui la transporte** — ce qui, rétrospectivement, explique pourquoi la campagne S50 ne pouvait rien
voir. Conséquence pour l'autonomie : ≈ **73 h à 1 Hz sur 2000 mAh** (contre 43 h en flux continu sans veille) —
les autonomies publiées avant ce sprint sont **pessimistes**, car mesurées sans sommeil. Le verdict INT8 du
paragraphe précédent, lui, **n'est pas modifié** : le gain reste en RAM, pas en énergie. Détail :
`docs/sprints/sprint_53/S5302_wfi_repos_reel.md` · `experiments/exp_S53_wfi/`.

**Volet latence INT8 (Sprint 50, S5004) — paradoxe FPU MESURÉ au cycle près.** Le CR demandait de
**détailler le coût de latence INT8** (le processeur étant FP32, la quantification ajoute des étapes de
déquant/requant). Instrumentation DWT **cycle-level** du kernel EWC INT8 v2 (`-DINT8_SEGMENT_PROFILE`, build
défaut invariant), board réelle NUCLEO-F439ZI, 2 datasets, 0 CRC (`exp_S50_int8_latency/`) : par inférence,
**MAC entier ≈ 6 785 cyc** (aucun gain vs FPU), **requantification FP32→INT8 ≈ 3 777 cyc (surcoût dominant,
`lroundf`)**, déquantification int→FP32 ≈ 200 cyc → **total INT8 74 µs vs FP32 48–50 µs, soit +24 à +26 µs
(~+50 %)**. C'est le **paradoxe latence FPU du Sprint 29, désormais chiffré poste par poste** : sur un
Cortex-M4 à FPU sans NPU ni SIMD entier, l'INT8 est un **coût net en latence** ; son bénéfice est strictement
la **RAM des poids ÷4** (Gap 3), pas la latence (Gap 2). Toutes latences ≪ 100 ms → **Gap 2 préservé**. Un vrai
gain latence exigerait une carte INT8 natif (STM32N6/NPU, indisponible) ou des noyaux SIMD CMSIS-NN.
`docs/context/int8_latency_breakdown.md` + `docs/context/int8_cost_benefit.md` (analyse coût/bénéfice :
reproduit-littérature vs contribution propre).

**Volet énergie (Sprint 53) — le verdict énergétique de l'INT8, et la marge de latence convertie.**
La campagne X-NUCLEO-LPM01A ferme la question laissée ouverte par le paradoxe ci-dessus : *l'INT8, qui
coûte en latence, rachète-t-il ce coût en énergie ?* **Non.** Par régression `I = I_base + pente · rate`
sur carte réelle (S5304, 7 cellules, 0 CRC), l'énergie par inférence de l'EWC diffère de **−0,2 µJ entre
INT8 et FP32 pour une incertitude combinée de ±2,3 µJ** : l'écart n'est pas séparable du bruit du banc.
**Le bénéfice de l'INT8 reste donc strictement la RAM des poids ÷4 (Gap 3)** — ni la latence (S29/S50),
ni l'énergie. Deux garde-fous rendent ce chiffre publiable : la pente est ajustée **pondérée par `1/σ²`**,
et la saturation est détectée sur la **cadence ATTEINTE** — au-delà de ~209 inf/s l'UART sature *en
silence*, sans perdre de trame ni lever de CRC, et les points saturés **sous-estiment la pente**.

**Gap 2 — la marge de latence est convertible en autonomie (S5303).** Le budget de 100 ms est tenu avec
plusieurs ordres de grandeur de marge ; le balayage SYSCLK mesure ce que cette marge vaut. De 45 à
180 MHz, **l'énergie par inférence croît de +26,2 %** (170,1 → 214,6 µJ) et le courant de base de **+54 %**,
pendant que **Gap 2 reste tenu 12,8×** à 45 MHz (pire cas HDC fp32 2 338 µs). Ralentir le MCU réduit donc
à la fois le coût par inférence et le courant permanent, sans menacer le critère. La mise en veille de
l'attente UART (`__WFI()`, S5302) abaisse en outre le repos de **45 %** (49,8 → 27,4 mA) et porte
l'autonomie duty-cyclée à **≈73 h à 1 Hz sur 2000 mAh** — les autonomies du Sprint 50 étaient
**pessimistes**, mesurées sur une carte jamais endormie. Réserve : la cellule 90 MHz sort en **N/A**
(r² 0,783 pondéré), la tendance ne s'appuie donc que sur ses deux extrémités.

**Ce qui reste `"à mesurer"`** : `energy_uj_per_update` (S5306, séance 1/3 — les deux régressions du build
S38 sont non publiables, r² 0,366 et 0,696), le profil par phase (S5305 — le mode dynamique de la sonde
ne se rouvre qu'à 45 MHz et la segmentation sur le seul courant est **mesurément insuffisante** ; la voie
PA8→D7 exige une soudure) et `by_component` (S5308, câblage séparé MCU/périphériques). Aucun de ces champs
ne porte de valeur inventée : chacun garde la valeur littérale `"à mesurer"` **accompagnée de sa raison
mesurée**, ce qui les distingue d'un « pas encore fait ».

**Note Sprint 38 — RAM du gate de mise à jour autonome** : le gate de nouveauté embarqué
(`SlidingWindowDriftDetector`, `-DEWC_AUTO_UPDATE`) coûte **+300 B** de `.bss` (`g_drift` fenêtre
glissante O(W=50) + `g_n_updates` + `g_last_verdict`) au-dessus de l'EWC seul (`.bss` défaut **105 036 B**
invariant → gate **105 336 B**), soit ~0,12 % des 256 Ko. Coût RAM négligeable pour rendre la mise à jour
**100 % autonome** (sans hôte). Détail : `docs/sprints/sprint_38/S3805_board_autonomous.md`.

**Question ouverte pour Dorra** : CMSIS-NN/CMSIS-DSP fournit des kernels INT8 pour l'inférence
(`arm_dot_prod_q7`, `arm_nn_vec_mat_mult_t_s8`) ; le prototype SIMD S2908 est **bloqué** (toolchain
`arm-none-eabi` sans `libarm_cortexM4lf_math.a` ni `arm_math.h`, installation manuelle proscrite) — `TODO(dorra)`.

---

## Utilisation dans le code

Chaque fonction critique du projet doit être annotée par rapport au triple gap :

```python
def update_oto_head(self, z: Tensor, y: Tensor) -> float:
    """
    Mise à jour en ligne de la tête OtO.
    
    Gap 2 relevance: Cette fonction s'exécute en RAM, sur Cortex-M55.
    Empreinte mesurée : voir experiments/exp_003/results/metrics.json
    
    Gap 3 relevance: Mise à jour en FP32. L'extension INT8 est dans
    l'extension buffer UINT8 (docs/models/tinyol_spec.md §7).
    """
```

---

## Critères de succès expérimentaux

Pour que ce projet "ferme" les gaps de manière crédible :

| Gap | Critère de succès minimal |
|-----|--------------------------|
| Gap 1 | Accuracy > 80 % sur Dataset 1 (temporel) avec protocole CL documenté |
| Gap 2 | `ram_peak_bytes` < 65 536 mesuré à l'exécution pour les 3 modèles |
| Gap 3 | Démonstration que le buffer UINT8 dégrade < 2 % la précision vs FP32 |
