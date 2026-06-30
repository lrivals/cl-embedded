# Sprint 28 — Analyse INT8 vs FP32 : Modèles Python × 5 Datasets (PC)

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 28 |
| **Semaine** | 16–20 juin 2026 |
| **Statut** | ⬜ À démarrer |
| **Priorité globale** | 🔴 Critique — compléter modèles INT8 Python + expériences PC avant portage board (Sprint 29) |
| **Durée estimée totale** | ~22h |
| **Dépendances** | Sprint 22 ✅ (`ewc_mlp_int8.py`, `src/utils/quantization.py`) · Sprint 23 ✅ (`exp_S23_INT8` référence EWC CMAPSS) · Sprint 24 ✅ (HDC INT8 partiel, ONNX exports) |

---

## Contexte et motivation

Les Sprints 22–24 ont validé Gap 3 partiellement : EWC INT8 sur CMAPSS avec ΔAUROC < 0.02 et réduction RAM ×2.7. Mais l'analyse est limitée à **un seul modèle × un seul dataset**. Sprint 28 étend cette étude à **4 modèles × 5 datasets** côté Python/PC pour :

1. Quantifier l'impact de l'INT8 de façon exhaustive (tableau 4×5)
2. Compléter les modèles INT8 manquants (HDC, TinyOL, Mahalanobis)
3. Fournir une base solide pour le portage firmware (Sprint 29) et le manuscrit

**Résultats Sprint 23 connus (baseline)** :

| Métrique | EWC FP32 | EWC INT8 | Critère Gap 3 |
|---------|:--------:|:--------:|:-------------:|
| ΔAUROC | — | 0.013 | < 0.02 ✅ |
| RAM poids | 9 728 B | 3 600 B | réduction ×2.7 ✅ |
| Latence PC | — | — | (mesure board S29) |

**Résultats Sprint 24 connus (EWC UINT8 Monitoring)** :

| Exp | Modèle | RAM FP32 | RAM UINT8 | Compression | AA |
|-----|--------|:--------:|:---------:|:-----------:|:--:|
| exp_S24_01 | EWC / Monitoring | 2 820 B | 705 B | 4.0× | 0.911 |
| exp_S24_02 | HDC / Monitoring | 49 152 B (hyp.) | 18 432 B (natif) | 2.67× | 0.870 |

```
Sprint 22/23/24 ✅                         Sprint 28
──────────────────────────────    ──────────────────────────────────────────
ewc_mlp_int8.py (EWC seul)   ──▶  S2803 hdc_int8.py
quantization.py (primitives) ──▶  S2804 tinyol_int8.py
HDC INT8 partiel Sprint 24   ──▶  S2805 mahalanobis_int8.py
                                          ↓
                               S2801 benchmark_int8_fp32.py (script unifié)
                                          ↓
                               S2802 configs YAML INT8 ×19
                               S2806 configs YAML INT8 HDC/TinyOL/Maha
                                          ↓
                               S2807 exp_S28_PC_ewc_hdc/ (EWC+HDC × 5 datasets)
                               S2808 exp_S28_PC_tinyol_maha/ (TinyOL+Maha × 5)
                                          ↓
                               S2809 tests/test_int8_benchmark.py
                               S2810 scripts/generate_int8_heatmaps.py
                               S2811 notebooks/sprint28_int8_pc.ipynb
```

**Critères de succès** :
1. `python scripts/benchmark_int8_fp32.py --model ewc --config configs/ewc_int8_cwru.yaml` — JSON valide
2. `pytest tests/test_int8_benchmark.py -v` — tous verts
3. 40 fichiers JSON dans `experiments/exp_S28_PC_ewc_hdc/` + `experiments/exp_S28_PC_tinyol_maha/`
4. Notebook `sprint28_int8_pc.ipynb` exécutable end-to-end (Run All sans erreur)
5. ΔAUROC < 0.05 pour tous les modèles × datasets (critère souple PC, critère strict < 0.02 pour le manuscrit)
6. RAM INT8 < RAM FP32 pour tous les modèles (propriété structurelle)

---

## Tâches

### O1 — Script benchmark unifié

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2801 | Créer `scripts/benchmark_int8_fp32.py` : arguments `--model {ewc,hdc,tinyol,mahalanobis}` `--config <yaml>` `--output <json>`, exécution séquentielle FP32 puis INT8, JSON normalisé `{model, dataset, metric_fp32, metric_int8, delta_metric, ram_fp32_bytes, ram_int8_bytes, ram_ratio, latency_fp32_ms, latency_int8_ms}` | 🔴 | ✅ | `scripts/benchmark_int8_fp32.py` | 3h |

### O2 — Modèles INT8 Python manquants

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2803 | Implémenter `src/models/hdc/hdc_int8.py` : encoder hypervecteurs base INT8 (±1 → `int8_t`), associative memory INT16, `encode_int8()`, update bundle en ligne INT8, `get_memory_footprint_int8()` | 🔴 | ✅ | `src/models/hdc/hdc_int8.py` | 3h |
| S2804 | Implémenter `src/models/tinyol/tinyol_int8.py` : autoencoder activations UINT8 (pattern Sprint 4 `quantize_uint8`), OtOHead poids INT8, `forward_int8()`, `get_memory_footprint_int8()` | 🔴 | ✅ | `src/models/tinyol/tinyol_int8.py` | 3h |
| S2805 | Implémenter `src/models/unsupervised/mahalanobis_int8.py` : quantification `mu_` (INT8 affine) + `sigma_inv_` (INT8 par-matrice avec scale séparé), `score_int8()` avec dequant intermédiaire, `get_memory_footprint_int8()` | 🔴 | ✅ | `src/models/unsupervised/mahalanobis_int8.py` | 2h |

### O3 — Configs YAML INT8

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2802 | Créer configs YAML INT8 EWC manquantes (CMAPSS déjà couvert via exp_S23) : `configs/ewc_int8_{cwru,monitoring,pronostia,paderborn}.yaml` | 🔴 | ⬜ | `configs/ewc_int8_*.yaml` ×4 | 1h |
| S2806 | Créer configs YAML INT8 HDC, TinyOL, Mahalanobis × 5 datasets : pattern = config FP32 existante + `quantization: int8` (15 fichiers) | 🟡 | ⬜ | `configs/{hdc,tinyol,mahalanobis}_int8_*.yaml` ×15 | 2h |

### O4 — Expériences PC

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2807 | Expériences PC EWC + HDC × 5 datasets : lancer `benchmark_int8_fp32.py`, sauver `experiments/exp_S28_PC_ewc_hdc/results_{model}_{dataset}.json` | 🔴 | ⬜ | `experiments/exp_S28_PC_ewc_hdc/` | 3h |
| S2808 | Expériences PC TinyOL + Mahalanobis × 5 datasets, sauver `experiments/exp_S28_PC_tinyol_maha/results_{model}_{dataset}.json` | 🔴 | ⬜ | `experiments/exp_S28_PC_tinyol_maha/` | 3h |

### O5 — Tests

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2809 | Tests Python `tests/test_int8_benchmark.py` : mini-dataset synthétique N=50 d=4, vérifier `delta_metric < 0.05` tous modèles, `ram_int8_bytes < ram_fp32_bytes` | 🟡 | ⬜ | `tests/test_int8_benchmark.py` | 2h |

### O6 — Visualisations + Notebook

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2810 | `scripts/generate_int8_heatmaps.py` : heatmap 4×5 ΔAUROC/F1/RMSE, heatmap 4×5 RAM ratio INT8/FP32, barplot latence PC INT8 vs FP32, sauver `docs/figures/sprint28_*.png` | 🟡 | ⬜ | `scripts/generate_int8_heatmaps.py` | 2h |
| S2811 | Notebook `notebooks/sprint28_int8_pc.ipynb` : synthèse résultats PC, 4 sections modèles, heatmaps importées, tableau conformité Gap 3 par modèle×dataset (critères ΔAUROC < 0.02, RAM ratio, latence) | 🔴 | ⬜ | `notebooks/sprint28_int8_pc.ipynb` | 2h |

---

## Ordre d'exécution recommandé

```
S2803 + S2804 + S2805  (modèles INT8 Python — parallèle, pas de dépendance entre eux)
     ↓
S2801 (script benchmark — requiert les 4 modèles chargés)
     ↓
S2802 + S2806 (configs YAML INT8 — parallèle)
     ↓
S2807 + S2808 (expériences PC — parallèle, datasets indépendants)
S2809 (tests Python — parallèle avec S2807/S2808)
     ↓
S2810 (heatmaps — requiert résultats JSON)
     ↓
S2811 (notebook final)
```

---

## Nomenclature des expériences

| Exp ID | Modèles | Datasets | Métriques PC |
|--------|---------|----------|--------------|
| exp_S28_PC_ewc_hdc | EWC FP32/INT8, HDC FP32/INT8 | CMAPSS, Monitoring, CWRU, Pronostia, Paderborn | ΔAUROC, RAM ratio, latence PC |
| exp_S28_PC_tinyol_maha | TinyOL FP32/INT8, Mahalanobis FP32/INT8 | CMAPSS, Monitoring, CWRU, Pronostia, Paderborn | ΔAUROC, RAM ratio, latence PC |

---

## Notes d'implémentation

**S2801 `benchmark_int8_fp32.py`** : prendre modèle sur `scripts/compare_experiments.py` pour le pattern de chargement de configs. Le JSON de sortie doit être normalisé pour que `generate_int8_heatmaps.py` puisse ingérer tous les résultats uniformément. Utiliser `tracemalloc` de `src/evaluation/memory_profiler.py` pour RAM.

**S2803 `hdc_int8.py`** : HDC est architecturalement entier — les hypervecteurs de base (après binarisation) ont des valeurs ±1 stockés en `int8`. L'associative memory accumule des bundles en `int16` (évite overflow sur addition de N hypervecteurs). La méthode `get_memory_footprint_int8()` est déjà esquissée dans `docs/sprints/sprint_24/S2402_uint8_ewc_hdc.md`. S'appuyer sur ce document et sur `src/models/hdc/hdc_classifier.py` (pattern existant).

**S2805 `mahalanobis_int8.py`** : La quantification porte sur `mu_` (vecteur d, `int8` affine) et `sigma_inv_` (matrice d×d, `int8` affine avec scale séparé par élément). La distance de Mahalanobis se recalcule avec dequantification intermédiaire en `float32`. Pas de mise à jour de poids nécessaire (fit est offline). Référence budget : `d=4 → 80 B @ FP32 / 20 B @ INT8` (commentaire déjà dans `mahalanobis_detector.py`).

---

## Budget mémoire estimé PC

| Modèle | RAM FP32 (poids) | RAM INT8 (poids) | Ratio est. |
|--------|:----------------:|:----------------:|:-----------:|
| EWC (5→32→16→2) | ~9 728 B | ~3 600 B | ×2.7 (mesuré S23) |
| HDC (D=2048, n=9) | ~49 152 B (hyp.) | ~18 432 B (natif) | ×2.67 (mesuré S24) |
| TinyOL (enc 9→32→16) | ~5 700 B | ~1 425 B | ×4.0 (estimé) |
| Mahalanobis (d=5) | ~120 B | ~30 B | ×4.0 (estimé) |

---

## Questions ouvertes

- `TODO(dorra)` : HDC est nativement entier — parler de "quantification INT8" ou de "profil mémoire natif entier" dans le manuscrit ? Impacte la revendication Gap 3.
- `TODO(arnaud)` : Mahalanobis INT8 : quantification `sigma_inv_` par-matrice (INT8 affine) suffit-elle pour tenir ΔAUROC < 0.02, ou faut-il Q15 (représentation plus fine) ?
- `FIXME(gap3)` : Résultat négatif latence attendu (INT8 plus lent sur Cortex-M4 FPU) — confirmer sur PC également ou spécifique au FPU hardware ?

---

## Livrables

1. `scripts/benchmark_int8_fp32.py` — script benchmark unifié
2. `src/models/hdc/hdc_int8.py` — HDC INT8 Python
3. `src/models/tinyol/tinyol_int8.py` — TinyOL INT8 Python
4. `src/models/unsupervised/mahalanobis_int8.py` — Mahalanobis INT8 Python
5. 19 configs YAML INT8 dans `configs/`
6. 2 répertoires `experiments/exp_S28_PC_*/` avec 20 JSON chacun (4 modèles × 5 datasets × FP32/INT8)
7. `tests/test_int8_benchmark.py` — tests Python
8. `scripts/generate_int8_heatmaps.py` + figures `docs/figures/sprint28_*.png`
9. `notebooks/sprint28_int8_pc.ipynb` — notebook PC exécutable

---

## Bilan (à compléter)

| Tâche | Statut | Temps réel | Notes |
|-------|:------:|:----------:|-------|
| S2801 benchmark_int8_fp32.py | ✅ | ~3h | Registre d'adaptateurs ; EWC complet (FP32+INT8 réels), HDC best-effort (natif INT8), TinyOL/Maha → `NotImplementedError` (S2804/S2805). Support `extends:` ajouté à `config_loader`. Config exemple `ewc_int8_monitoring.yaml`. 6/6 tests PASS. |
| S2802 configs EWC INT8 ×4 | ⬜ | — | — |
| S2803 hdc_int8.py | ✅ | ~1h | `HDCClassifierInt8` : base vectors INT8 + AM INT16 (vs INT32), `encode_int8`/`update_int8`/`predict`, footprint réel. Réutilise `encode_observation`/`base_vectors.py`. Métrique INT8 == FP32 (binarisation exacte). Smoke test OK. |
| S2804 tinyol_int8.py | ✅ | ~1h | `TinyOLAutoencoderInt8` (fake-quant poids INT8 + activations UINT8, `calibrate_int8`/`forward_int8`/`reconstruction_error_int8`) + `OtOHeadInt8` (SGD fake-quant, master weights FP32). Enveloppe l'archi **réelle** (enc 25→32→16→8, OtOHead 9→1 ≠ chiffres spec). Loss OtO ↓ 0.75→0.51. |
| S2805 mahalanobis_int8.py | ✅ | ~0.5h | `MahalanobisDetectorInt8(MahalanobisDetector)` : `mu_`/`sigma_inv_` quantifiés affine par-tenseur, `calibrate_int8`/`score_int8`/`predict_int8`, footprint (30 B poids purs d=5). mean‖Δscore‖=0.0017 vs FP32. `TODO(arnaud)` Q15 documenté en commentaire. |
| S2806 configs HDC/TinyOL/Maha INT8 ×15 | ⬜ | — | — |
| S2807 exp PC EWC+HDC | ⬜ | — | — |
| S2808 exp PC TinyOL+Maha | ⬜ | — | — |
| S2809 tests Python | ⬜ | — | — |
| S2810 heatmaps | ⬜ | — | — |
| S2811 notebook sprint28 | ⬜ | — | — |
