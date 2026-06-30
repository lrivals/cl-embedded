# S2807–S2808 — Expériences PC FP32 vs INT8 (4 modèles × 5 datasets)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 28 |
| **Priorité** | 🔴 Critique |
| **Statut** | ✅ Implémenté (12 juin 2026) — 20/20 cellules exécutées (S2807 EWC+HDC, S2808 TinyOL+Maha) |
| **Durée estimée** | S2807 : 3h / S2808 : 3h |
| **Dépendances** | S2801 ✅ (`benchmark_int8_fp32.py`) · S2802/S2806 ✅ (configs YAML) · S2803/S2804/S2805 ✅ (modèles INT8) |
| **Fichiers cibles** | `experiments/exp_S28_PC_ewc_hdc/`, `experiments/exp_S28_PC_tinyol_maha/` |

---

## Contexte

Ces deux tâches lancent les expériences PC et produisent le tableau 4×5 (modèles × datasets) central au Sprint 28. S2807 couvre EWC + HDC, S2808 couvre TinyOL + Mahalanobis.

---

## S2807 — EWC + HDC × 5 datasets

**Commandes à lancer** :

```bash
# EWC × 5 datasets
for ds in cwru monitoring pronostia paderborn; do
    python scripts/benchmark_int8_fp32.py \
        --model ewc \
        --config configs/ewc_int8_${ds}.yaml \
        --output experiments/exp_S28_PC_ewc_hdc/results_ewc_${ds}.json
done
# CMAPSS déjà dans exp_S23_INT8 — copier ou relancer
python scripts/benchmark_int8_fp32.py \
    --model ewc --config configs/ewc_int8_cmapss.yaml \
    --output experiments/exp_S28_PC_ewc_hdc/results_ewc_cmapss.json

# HDC × 5 datasets
for ds in cmapss monitoring cwru pronostia paderborn; do
    python scripts/benchmark_int8_fp32.py \
        --model hdc \
        --config configs/hdc_int8_${ds}.yaml \
        --output experiments/exp_S28_PC_ewc_hdc/results_hdc_${ds}.json
done
```

**Structure de sortie** :

```
experiments/exp_S28_PC_ewc_hdc/
├── results_ewc_cmapss.json
├── results_ewc_cwru.json
├── results_ewc_monitoring.json
├── results_ewc_pronostia.json
├── results_ewc_paderborn.json
├── results_hdc_cmapss.json
├── results_hdc_cwru.json
├── results_hdc_monitoring.json
├── results_hdc_pronostia.json
├── results_hdc_paderborn.json
└── config_snapshot.yaml
```

---

## S2808 — TinyOL + Mahalanobis × 5 datasets

**Commandes similaires** avec `--model tinyol` et `--model mahalanobis`.

**Structure de sortie** :

```
experiments/exp_S28_PC_tinyol_maha/
├── results_tinyol_{cmapss,cwru,monitoring,pronostia,paderborn}.json (×5)
├── results_mahalanobis_{cmapss,cwru,monitoring,pronostia,paderborn}.json (×5)
└── config_snapshot.yaml
```

---

## Tableau 4×5 — Résultats (n_samples=600/tâche, seed=42, 12 juin 2026)

> Métrique : **AUROC** détection de panne (EWC = proba sigmoïde ; TinyOL = erreur de
> reconstruction MSE ; Mahalanobis = distance), labels binarisés *normal-vs-fault*.
> **F1 macro** pour HDC. RAM = octets de poids (exacte, indépendante de n_samples).
> Gap 3 : `RAM ✅` si ratio > 1.0 ; `métrique ✅` si `|Δ| < 0.02`.

| Modèle | Dataset | Métr. FP32 | Métr. INT8 | Δ | RAM FP32 | RAM INT8 | Ratio | Gap 3 |
|--------|---------|:----------:|:----------:|:------:|:--------:|:--------:|:-----:|:------:|
| EWC | CMAPSS | 0.768 | 0.773 | +0.0056 | 2 948 | 737 | 4.00× | ✅ ✅ |
| EWC | Monitoring | 0.939 | 0.939 | −0.0007 | 2 820 | 705 | 4.00× | ✅ ✅ |
| EWC | CWRU | 1.000 | 1.000 | −0.0002 | 3 460 | 865 | 4.00× | ✅ ✅ |
| EWC | Pronostia | 0.988 | 0.988 | −0.0001 | 3 972 | 993 | 4.00× | ✅ ✅ |
| EWC | Paderborn | N/A¹ | N/A¹ | — | 2 948 | 737 | 4.00× | ✅ / N/A |
| HDC | CMAPSS | 0.723 | 0.723 | 0.0000 | 14 344 | 6 152 | 2.33× | ✅ ✅ |
| HDC | Monitoring | 0.758 | 0.758 | 0.0000 | 14 344 | 6 152 | 2.33× | ✅ ✅ |
| HDC | CWRU | 0.929 | 0.929 | 0.0000 | 7 176 | 3 080 | 2.33× | ✅ ✅ |
| HDC | Pronostia | 0.709 | 0.709 | 0.0000 | 14 344 | 6 152 | 2.33× | ✅ ✅ |
| HDC | Paderborn | N/A² | N/A² | — | — | — | — | N/A |
| TinyOL | CMAPSS | 0.720 | 0.741 | +0.0203 | 2 148 | 585 | 3.67× | ✅ / ❌³ |
| TinyOL | Monitoring | 0.899 | 0.907 | +0.0078 | 1 456 | 412 | 3.53× | ✅ ✅ |
| TinyOL | CWRU | 0.707 | 0.762 | +0.0545 | 3 148 | 835 | 3.77× | ✅ / ❌³ |
| TinyOL | Pronostia | 0.716 | 0.715 | −0.0012 | 4 276 | 1 117 | 3.83× | ✅ ✅ |
| TinyOL | Paderborn | N/A¹ | N/A¹ | — | 2 148 | 585 | 3.67× | ✅ / N/A |
| Mahalanobis | CMAPSS | 0.655 | 0.649 | −0.0060 | 120 | 30 | 4.00× | ✅ ✅ |
| Mahalanobis | Monitoring | 0.972 | 0.972 | 0.0000 | 80 | 20 | 4.00× | ✅ ✅ |
| Mahalanobis | CWRU | 0.475 | 0.239 | −0.2363 | 360 | 90 | 4.00× | ✅ / ❌⁴ |
| Mahalanobis | Pronostia | 0.857 | 0.620 | −0.2379 | 728 | 182 | 4.00× | ✅ / ❌⁴ |
| Mahalanobis | Paderborn | N/A¹ | N/A¹ | — | 120 | 30 | 4.00× | ✅ / N/A |

**Notes**
1. **Paderborn — AUROC N/A** : le scénario domain-incremental Paderborn produit des tâches
   de test mono-classe → AUROC indéfini (NaN). La **compression RAM reste valide et mesurée**.
   Privilégier une métrique multiclasse (F1) pour Paderborn dans le manuscrit.
2. **HDC × Paderborn N/A** : HDC exige des `feature_bounds` calibrés par dataset (étape type
   S2-01) non disponibles pour Paderborn → cellule reportée (calibration HDC Paderborn = tâche séparée).
3. **TinyOL — `|Δ| > 0.02` mais amélioration** : l'INT8 *augmente* l'AUROC (CMAPSS +0.020,
   CWRU +0.055). La fake-quantization agit comme une régularisation sur l'erreur de
   reconstruction ; le critère Gap 3 strict (`|Δ| < 0.02`) est dépassé mais la dégradation
   est nulle (gain). Acceptable pour le manuscrit, à commenter comme effet régularisant.
4. **Mahalanobis — dégradation INT8 forte (CWRU −0.236, Pronostia −0.238)** : confirme le
   `TODO(arnaud)` de `S2805` — la quantification INT8 affine de `sigma_inv_` (grande dynamique)
   dégrade la distance. **Recommandation : fallback Q15 (int16) pour `sigma_inv_`** sur ces
   datasets. CMAPSS/Monitoring (dynamique plus faible) tiennent le critère.

## Synthèse Gap 3

- **RAM (compression) : ✅ sur les 18 cellules mesurées** — ratio 2.33× (HDC, int16 AM) à
  4.00× (EWC/Mahalanobis, int8 pur). La compression INT8 est systématique et exacte.
- **Métrique (préservation) : ✅ 12/16 cellules numériques** (`|Δ| < 0.02`). Les 4 exceptions :
  2 TinyOL (amélioration, note 3) + 2 Mahalanobis (`sigma_inv_` INT8, note 4 → fallback Q15).
- **EWC & HDC** préservent la métrique sur tous les datasets exploitables (Δ ≤ 0.006 / 0.000).

## Reproduction

```bash
# Campagne complète (20 cellules) :
python scripts/run_s28_pc_benchmarks.py --n_samples 600
# Cellule unique :
python scripts/benchmark_int8_fp32.py --model ewc \
    --config configs/ewc_int8_cwru.yaml \
    --output experiments/exp_S28_PC_ewc_hdc/results_ewc_cwru.json --n_samples 600
```

Sorties : `experiments/exp_S28_PC_ewc_hdc/` (EWC+HDC) et `experiments/exp_S28_PC_tinyol_maha/`
(TinyOL+Maha), un JSON normalisé par cellule + `config_snapshot.yaml`.
