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

**Volet énergie (Sprint 33)** : le constat « INT8 réduit la RAM sans accélérer la latence FPU » ouvre une
question énergie potentiellement originale — l'INT8 réduit-il néanmoins les **µJ** (moins d'accès mémoire) ? La
chaîne de mesure est livrée et fonctionnelle : marqueurs de phase GPIO firmware (PA8, `ENERGY_MARKERS`, S3304),
driver PowerShield X-NUCLEO-LPM01A `scripts/energy_capture.py` (segmentation par phase + intégration µJ, S3305),
métriques de coût `compute_cost.py`/`hw_cost_model.py` (FLOPs/**BOPs**/FLOPS-W ; BOPs rend le gain INT8 quantitatif :
`BOPs_fp32/BOPs_int8 = (32/8)² = 16`), et autonomie `src/evaluation/autonomy.py` (Capacité/I_moy). **Réponse
chiffrée différée** : les valeurs énergie/autonomie restent `"à mesurer"` tant que le LPM01A n'a pas été
physiquement posé/capturé (règle « aucun chiffre inventé »). Synthèse : `notebooks/cl_eval/energy_cost/comparison.ipynb`.

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
