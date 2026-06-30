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
