# Sprint 22 — Nouveaux datasets temporels + Gap 3 INT8 Python+C

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 22 |
| **Semaine** | 7 – 21 juin 2026 |
| **Statut** | ✅ Terminé |
| **Priorité globale** | 🔴 Critique — Gap 1 (2 nouveaux datasets) + Gap 3 (INT8 backprop Python+C) |
| **Durée estimée totale** | ~30h |
| **Dépendances** | Sprint 21 ✅ (3 modèles C validés board, protocole v3, Pronostia+Monitoring board) |

---

## Objectifs

Sprint 22 s'attaque simultanément à deux contributions manquantes pour le manuscrit :

1. **Gap 1 — Nouveaux datasets** : ajouter 2 datasets temporels industriels non encore traités :
   - **CMAPSS** (NASA turbofan, 21 capteurs, RUL) — scénario CL natural (FD001→FD004)
   - **Paderborn** (bearing electrique, courant + vibration) — diversité signal courant moteur

2. **Gap 3 — INT8 backprop** : implémentation SGD INT8 sur la tête MLP d'EWC, en Python (simulation fake-quant) et portage C complet pour validation board Sprint 23.

```
CMAPSS (RUL, 21 capteurs)       Paderborn (bearing, courant moteur)
         ↓                                   ↓
   data/raw/cmapss/           data/raw/paderborn/
         ↓                                   ↓
   EDA + feature eng.         EDA + feature eng.
   (RUL capping, top-5)       (FFT features, top-5)
         ↓                                   ↓
   Expériences CL PC           Expériences CL PC
   EWC / HDC / TinyOL /        EWC / Mahalanobis
   Mahalanobis                              ↓
         ↓                         notebooks + plots
   notebooks + plots
         ↓
   Gap 3 : ewc_mlp_int8.py (fake quant)
         ↓
   ewc_head_int8.c (C, sans board — board = Sprint 23)
         ↓
   exp_S22_INT8_* : FP32 vs INT8 PC comparison
```

**Critères de succès** :
1. `pytest tests/ -k cmapss` vert
2. `pytest tests/ -k paderborn` vert
3. 6 expériences CL PC dans `experiments/` avec métriques complètes
4. `ewc_mlp_int8.py` : AUROC_INT8 ≥ AUROC_FP32 − 0.02 (critère Gap 3)
5. `ewc_head_int8.c` compilable `arm-none-eabi-gcc` sans erreur (board = Sprint 23)

---

## Datasets

### CMAPSS — NASA C-MAPSS Turbofan Engine Degradation

| Propriété | Valeur |
|-----------|--------|
| Source | NASA Prognostics Center / Kaggle |
| Chemin | `data/raw/cmapss/` |
| Taille | ~10 Mo, 4 fichiers (FD001–FD004) |
| Type | Séries temporelles multivariées (21 capteurs) |
| Label | RUL (Remaining Useful Life, continu) → binarisé en `faulty` (RUL ≤ 30) |
| Scénario CL | Domain-incremental : FD001 → FD002 → FD003 → FD004 (conditions opératoires) |
| Références | `Hurtado2023CLPdM`, `DeLange2021Survey` |

**Features** : op_setting_1, op_setting_2, op_setting_3 + 21 capteurs (T2, T24, T30, T50, P2, P15, P30, Nf, Nc, epr, Ps30, phi, NRf, NRc, BPR, farB, htBleed, Nf_dmd, PCNfR_dmd, W31, W32)

**Feature selection** : top-5 par mutual info avec RUL (S2202)

### Paderborn University — Bearing Electrical Fault Dataset

| Propriété | Valeur |
|-----------|--------|
| Source | Paderborn University / KAt-DataCenter |
| Chemin | `data/raw/paderborn/` |
| Taille | ~2 Go (signaux bruts) → features extraites ~50 Mo |
| Type | Signaux courant moteur + vibration accéléromètre |
| Label | État roulement : K001 (sain), OR, IR, combinés → binarisé `faulty` |
| Scénario CL | Domain-incremental : sain → défaut OR → défaut IR |
| Références | `Benatti2019HDC` (signal EEG similaire), `Capogrosso2023TinyML` |

**Feature engineering** : FFT (énergie par bande), RMS, kurtosis, crest factor → ~20 features → top-5 mutual info

---

## Tâches

### O1 — CMAPSS : EDA + Loader + Feature Engineering

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2201 | Download CMAPSS (NASA/Kaggle) → `data/raw/cmapss/` | 🔴 | ✅ | `data/raw/cmapss/` | 30 min |
| S2202 | `src/data/cmapss_loader.py` : loader + RUL capping (cap=125) + normalisation MinMax + feature selection top-5 | 🔴 | ✅ | `src/data/cmapss_loader.py` | 2h |
| S2203 | Notebook EDA `notebooks/eda_cmapss.ipynb` : distribution RUL, drift FD001→FD004, corrélations, top-5 features visualisation | 🔴 | ✅ | `notebooks/eda_cmapss.ipynb` | 2h |
| S2204 | `configs/cmapss_config.yaml` : tâches CL, features, seuil RUL→faulty, seed | 🟡 | ✅ | `configs/cmapss_config.yaml` | 30 min |

### O2 — CMAPSS : Expériences CL PC (4 modèles)

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2205 | exp_S22_01 : EWC / CMAPSS — `python scripts/train_ewc.py --config configs/cmapss_config.yaml` | 🔴 | ✅ | `experiments/exp_S22_01/` | 1h |
| S2206 | exp_S22_02 : HDC / CMAPSS — `python scripts/train_hdc.py --config configs/cmapss_config.yaml` | 🟡 | ✅ | `experiments/exp_S22_02/` | 1h |
| S2207 | exp_S22_03 : TinyOL / CMAPSS — `python scripts/train_tinyol.py --config configs/cmapss_config.yaml` | 🟡 | ✅ | `experiments/exp_S22_03/` | 1h |
| S2208 | exp_S22_04 : Mahalanobis / CMAPSS | 🟡 | ✅ | `experiments/exp_S22_04/` | 30 min |
| S2209 | Notebook `notebooks/results_cmapss_cl.ipynb` : courbes AUROC/AF/BWT per tâche, comparaison 4 modèles | 🔴 | ✅ | `notebooks/results_cmapss_cl.ipynb` | 2h |

### O3 — Paderborn : EDA + Loader + Feature Engineering

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2210 | Download Paderborn subset (K001 + 2 défauts OR/IR) → `data/raw/paderborn/` | 🔴 | ✅ | `data/raw/paderborn/` | 1h |
| S2211 | `src/data/paderborn_loader.py` : FFT features (RMS, kurtosis, crest, énergie 4 bandes) + top-5 sélection | 🔴 | ✅ | `src/data/paderborn_loader.py` | 3h |
| S2212 | Notebook EDA `notebooks/eda_paderborn.ipynb` : FFT spectrum, drift par condition, distribution features | 🟡 | ✅ | `notebooks/eda_paderborn.ipynb` | 2h |
| S2213 | `configs/paderborn_config.yaml` | 🟡 | ✅ | `configs/paderborn_config.yaml` | 30 min |

### O4 — Paderborn : Expériences CL PC

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2214 | exp_S22_05 : EWC / Paderborn (sain → OR → IR) | 🔴 | ✅ | `experiments/exp_S22_05/` | 1h |
| S2215 | exp_S22_06 : Mahalanobis / Paderborn (baseline légère) | 🟡 | ✅ | `experiments/exp_S22_06/` | 30 min |
| S2216 | Notebook `notebooks/results_paderborn_cl.ipynb` : courbes + comparaison avec CWRU (même famille vibration) | 🟡 | ✅ | `notebooks/results_paderborn_cl.ipynb` | 2h |

### O5 — Gap 3 : INT8 Backprop Python (simulation fake-quant)

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2217 | `src/models/ewc/ewc_mlp_int8.py` : SGD INT8 simulé par fake-quantization (torch.quantization) sur la tête MLP | 🔴 | ✅ | `src/models/ewc/ewc_mlp_int8.py` | 4h |
| S2218 | exp_S22_INT8_01 : EWC FP32 vs INT8 sur CWRU (référence) | 🔴 | ✅ | `experiments/exp_S22_INT8_01/` | 1h |
| S2219 | exp_S22_INT8_02 : EWC FP32 vs INT8 sur CMAPSS (nouveau dataset) | 🔴 | ✅ | `experiments/exp_S22_INT8_02/` | 1h |
| S2220 | Notebook `notebooks/int8_vs_fp32_comparison.ipynb` : tableau AUROC / AF / BWT × {FP32, INT8} × {CWRU, CMAPSS} | 🔴 | ✅ | `notebooks/int8_vs_fp32_comparison.ipynb` | 2h |

### O6 — Gap 3 : INT8 Portage C (validation board = Sprint 23)

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2221 | `firmware/stm32f4_blink/src/ewc_head_int8.c` : implémentation INT8 fixed-point (Q7 / Q15) — forward + update | 🔴 | ✅ | `firmware/stm32f4_blink/src/ewc_head_int8.c` | 4h |
| S2222 | Tests Unity `test_ewc_int8.c` : vérifier cohérence output INT8 vs FP32 (delta < 0.05) sur x86 | 🔴 | ✅ | `firmware/stm32f4_blink/tests/test_ewc_int8.c` | 2h |

### O7 — Tests + Documentation

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2223 | `tests/test_cmapss_loader.py` + `tests/test_paderborn_loader.py` | 🟡 | ✅ | `tests/` | 1h |
| S2224 | `docs/datasets_analysis.md` : section CMAPSS + Paderborn (stats, scénario CL, résultats clés) | 🟡 | ✅ | `docs/datasets_analysis.md` | 1h |
| S2225 | Roadmap update (Sprint 22 clôturé, Sprint 23 preview) | 🟡 | ✅ | `docs/roadmap_phase2.md` | 30 min |

---

## Métriques attendues

| Expérience | Modèle | Dataset | acc_final attendu | AF attendu |
|-----------|--------|---------|:-----------------:|:----------:|
| exp_S22_01 | EWC | CMAPSS | ≥ 0.75 | ≤ 0.15 |
| exp_S22_02 | HDC | CMAPSS | ≥ 0.65 | ≤ 0.20 |
| exp_S22_03 | TinyOL | CMAPSS | ≥ 0.70 | ≤ 0.20 |
| exp_S22_04 | Mahalanobis | CMAPSS | ≥ 0.60 | ≤ 0.25 |
| exp_S22_05 | EWC | Paderborn | ≥ 0.80 | ≤ 0.10 |
| exp_S22_06 | Mahalanobis | Paderborn | ≥ 0.65 | ≤ 0.20 |
| exp_S22_INT8_01/02 | EWC INT8 | CWRU+CMAPSS | Δ AUROC < 0.02 | — |

---

## Livrables

1. `src/data/cmapss_loader.py` + `src/data/paderborn_loader.py`
2. `configs/cmapss_config.yaml` + `configs/paderborn_config.yaml`
3. 6 dossiers `experiments/exp_S22_*/` avec `results.json` + `config_snapshot.yaml`
4. 2 dossiers `experiments/exp_S22_INT8_*/` avec tableau FP32 vs INT8
5. 5 notebooks : `eda_cmapss`, `eda_paderborn`, `results_cmapss_cl`, `results_paderborn_cl`, `int8_vs_fp32_comparison`
6. `firmware/stm32f4_blink/src/ewc_head_int8.c` + tests Unity

---

## Notes et risques

- **Paderborn** : les fichiers bruts sont lourds (~2 Go). Extraire uniquement K001 (health) + OR + IR (3 conditions) pour rester < 500 Mo stockés.
- **CMAPSS RUL → classification** : binariser RUL ≤ 30 en `faulty=1` pour compatibilité avec les scripts existants. Documenter ce choix dans `eda_cmapss.ipynb`.
- **INT8 C** : utiliser Q7 (8 bits) pour les activations, Q15 pour les accumulateurs. S'inspirer de `ewc_head.c` existant. La validation board (latence INT8 vs FP32) est remise au Sprint 23 (S2307).
- `TODO(arnaud)` : valider le choix de binarisation RUL ≤ 30 comme seuil de défaillance imminente.
