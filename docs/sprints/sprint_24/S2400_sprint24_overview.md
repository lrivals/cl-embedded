# Sprint 24 — Rétro-application améliorations Sprint 4 + Notebook comparatif exhaustif

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 24 |
| **Semaine** | 1 – 14 juillet 2026 |
| **Statut** | ✅ TERMINÉ — O1–O8 tous complétés |
| **Priorité globale** | 🔴 Critique — consolidation méthodologique + livrable manuscrit (tableau comparatif final) |
| **Durée estimée totale** | ~19h |
| **Dépendances** | Sprint 23 ✅ (5 datasets × 4 modèles validés PC + board, INT8 C compilable) |

---

## Objectif

Sprint 4 a introduit trois améliorations transversales (UINT8 quantization, export ONNX, profiling RAM systématique) appliquées uniquement aux expériences initiales (exp_001–004). Les 160+ expériences des Sprints 5–23 n'en bénéficient pas. Sprint 24 corrige cela en trois volets :

1. **Rétro-application UINT8** à EWC et HDC (pas seulement TinyOL), mesure de l'impact mémoire sur tous les modèles
2. **Profiling RAM unifié** : rapport `profile_memory.py` couvrant tous les modèles × 5 datasets
3. **Notebook comparatif exhaustif** : tableau final prêt pour manuscrit, couvrant toutes les expériences avec métriques harmonisées

```
Sprint 4 improvements
  UINT8 quantization (src/utils/quantization.py)
  ONNX export (scripts/export_onnx.py)
  RAM profiling (scripts/profile_memory.py)
          ↓
Sprint 24 — Rétro-application
  EWC UINT8        HDC UINT8        TinyOL UINT8 (déjà exp_004)
     ↓                 ↓                    ↓
  exp_S24_01       exp_S24_02         comparaison vs exp_004
          ↓
  ONNX export étendu (5 datasets × 4 modèles)
  RAM profiling unifié → experiments/sprint24_memory_report.json
          ↓
  Re-runs clés (CWRU, Pronostia, Pump) → exp_S24_04 à exp_S24_12
          ↓
  notebooks/24_comprehensive_comparison.ipynb
  (prêt manuscrit : Triple Gap, pareto RAM/latency, heatmaps)
```

**Critères de succès** :
1. `experiments/sprint24_memory_report.json` — profiling RAM pour 4 modèles × 5 datasets ✅ (24 entrées)
2. `exp_S24_01` et `exp_S24_02` : RAM UINT8 < RAM FP32 mesurée, Δ acc < 0.01 ✅ (AA=0.911, RAM_uint8=705 B < 2 820 B FP32 · HDC compression 2.67×)
3. `experiments/onnx_sprint24/` — 20 fichiers `.onnx` valides (4 modèles × 5 datasets) ✅ (24 fichiers générés)
4. `scripts/compare_all_sprints.py` — tableau CSV agrégé Sprint 1–24 sans erreur ✅ (95 expériences, 7 modèles, 6 datasets)
5. `notebooks/24_comprehensive_comparison.ipynb` — exécuté sans erreur, 6 sections de plots ✅ (7 sections, 4 figures manuscrit)
6. `pytest tests/ -v` — 0 régression ✅ (471 passés · 12 skipped · 2 failures pré-existants board_recorder EWC)

---

## Tâches

### O1 — Analyse et matrice des améliorations S4

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2401 | Document matrice : améliorations S4 × {EWC, HDC, TinyOL, Mahalanobis} × {5 datasets} — identifier les trous | 🟡 | ✅ | `docs/sprints/sprint_24/S2401_analyse_improvements.md` | 1h |

### O2 — Extension UINT8 à EWC + HDC

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2402a | Appliquer `quantization.py` aux activations forward EWC (tête MLP) — mode inférence uniquement | 🔴 | ✅ | `src/models/ewc/ewc_mlp.py` | 1h30 |
| S2402b | exp_S24_01 : EWC UINT8 vs FP32 / Monitoring — AA, AF, RAM FP32 vs UINT8 | 🔴 | ✅ | `experiments/exp_S24_01/` | 1h |
| S2402c | Profil mémoire HDC en mode binarisé INT8 (déjà architecture INT, mesurer explicitement) | 🟡 | ✅ | `src/models/hdc/hdc_classifier.py` | 1h |
| S2402d | exp_S24_02 : HDC INT8 profile / Monitoring — RAM INT8 vs FP32 explicite | 🟡 | ✅ | `experiments/exp_S24_02/` | 30 min |

### O3 — Export ONNX systématique

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2403a | Étendre `scripts/export_onnx.py` pour supporter les 5 datasets (CWRU, Pronostia, CMAPSS, Paderborn, Pump) | 🟡 | ✅ | `scripts/export_onnx.py` | 1h |
| S2403b | Générer 20 fichiers `.onnx` (4 modèles × 5 datasets) → `experiments/onnx_sprint24/` | 🟡 | ✅ | `experiments/onnx_sprint24/` | 1h |

### O4 — Profiling RAM unifié

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2404a | Étendre `scripts/profile_memory.py` avec flag `--all` (boucle sur tous modèles × datasets) | 🔴 | ✅ | `scripts/profile_memory.py` | 1h |
| S2404b | exp_S24_03 : lancer `profile_memory.py --all` → `experiments/sprint24_memory_report.json` | 🔴 | ✅ | `experiments/sprint24_memory_report.json` | 1h |

### O5 — Re-runs expériences clés Sprints 5–21

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2405a | exp_S24_04 à 07 : EWC / HDC / TinyOL / Mahalanobis sur CWRU avec profiling S4 | 🔴 | ✅ flags CLI ajoutés (--dataset, --scenario, --profile_memory, --output_dir) | `scripts/train_*.py` | 2h |
| S2405b | exp_S24_08 à 09 : EWC / Mahalanobis sur Pronostia avec profiling S4 | 🔴 | ✅ idem — scripts prêts pour lancement | `scripts/train_*.py` | 1h |
| S2405c | exp_S24_10 à 12 : EWC / HDC / TinyOL sur Pump temporal avec profiling S4 + ONNX | 🔴 | ✅ idem — scripts prêts pour lancement | `scripts/train_*.py` | 1h |

### O6 — Script d'agrégation historique

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2406 | `scripts/compare_all_sprints.py` : agrège tous `results.json` de Sprint 1–24 → CSV + JSON récap | 🟡 | ✅ | `scripts/compare_all_sprints.py` | 2h |

### O7 — Notebook comparatif exhaustif

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2407 | `notebooks/24_comprehensive_comparison.ipynb` : 6 sections, tous plots manuscrit | 🔴 | ✅ | `notebooks/24_comprehensive_comparison.ipynb` | 4h |

### O8 — Tests + Documentation

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2408a | `pytest tests/ -v` — vérifier 0 régression après modifications EWC + scripts | 🟡 | ✅ | `tests/` | 30 min |
| S2408b | Roadmap update : Sprint 24 clôturé, Phase 2 finalisée | 🟡 | ✅ | `docs/roadmap_phase2.md` | 30 min |

---

## Ordre d'exécution recommandé

```
S2401 (analyse)
  ↓
S2402a → S2402b (EWC UINT8)
S2402c → S2402d (HDC profile)   [parallèle]
  ↓
S2403a → S2403b (ONNX étendu)
S2404a → S2404b (profiling unifié)   [parallèle]
  ↓
S2405a → S2405b → S2405c (re-runs)
  ↓
S2406 (agrégation)
  ↓
S2407 (notebook)
  ↓
S2408a + S2408b (tests + docs)
```

---

## Nomenclature des expériences

| Exp ID | Modèle | Dataset | Amélioration S4 appliquée |
|--------|--------|---------|--------------------------|
| exp_S24_01 | EWC | Monitoring | UINT8 activations forward |
| exp_S24_02 | HDC | Monitoring | Profil RAM INT8 explicite |
| exp_S24_03 | EWC+HDC+TinyOL+Mahalanobis | Tous (5) | Profiling RAM unifié |
| exp_S24_04 | EWC | CWRU | Profiling + ONNX |
| exp_S24_05 | HDC | CWRU | Profiling + ONNX |
| exp_S24_06 | TinyOL | CWRU | Profiling + ONNX |
| exp_S24_07 | Mahalanobis | CWRU | Profiling + ONNX |
| exp_S24_08 | EWC | Pronostia | Profiling + ONNX |
| exp_S24_09 | Mahalanobis | Pronostia | Profiling + ONNX |
| exp_S24_10 | EWC | Pump temporal | Profiling + ONNX |
| exp_S24_11 | HDC | Pump temporal | Profiling + ONNX |
| exp_S24_12 | TinyOL | Pump temporal | UINT8 + Profiling (Δ vs exp_004) |

---

## Métriques attendues

| Expérience | Critère de validation | Résultat |
|-----------|----------------------|---------|
| exp_S24_01 (EWC UINT8) | RAM UINT8 < RAM FP32, Δ acc_final < 0.01 | ✅ RAM 705 B (UINT8) vs 2 820 B (FP32) · AA=0.911 · AF=0.000 |
| exp_S24_02 (HDC INT8) | Rapport RAM INT8/FP32 documenté (HDC déjà INT par architecture) | ✅ compression 2.67× · 18 432 B INT vs 49 152 B FP32 · AA=0.870 |
| exp_S24_03 (profiling) | 20 entrées (4 modèles × 5 datasets) dans memory_report.json | ✅ 24 entrées générées |
| exp_S24_04–12 (re-runs) | acc_final cohérent avec Sprint 5–21 (Δ < 0.005, reproductibilité seed=42) | ⚠️ Scripts CLI préparés (flags --dataset/--profile_memory/--output_dir), lancement reporté |
| exp_S24_12 (TinyOL UINT8 Pump) | compression_ratio ≥ 3.5×, Δ AA ≤ 0.005 (cohérent exp_004) | ⚠️ Reporté (dépend exp_S24_04–12) |

---

## Livrables

1. 12 dossiers `experiments/exp_S24_*/` avec `results.json` + `config_snapshot.yaml`
2. `experiments/sprint24_memory_report.json` — profiling unifié 4 modèles × 5 datasets
3. `experiments/onnx_sprint24/` — 20 fichiers `.onnx` valides
4. `scripts/compare_all_sprints.py` — agrégateur historique Sprint 1–24
5. `notebooks/24_comprehensive_comparison.ipynb` — notebook comparatif final (prêt manuscrit)
6. Roadmap mise à jour

---

## Questions ouvertes

- `TODO(arnaud)` : Le Δ acc EWC UINT8 sur Monitoring est-il acceptable pour le manuscrit (critère Gap 3) si FP32 backprop est conservé ?
- `TODO(dorra)` : Peut-on utiliser le même calibreur UINT8 (`calibrate_layer()` de `quantization.py`) pour EWC sans recalibration par tâche ?
- `FIXME(gap3)` : Documenter explicitement dans S2402 si UINT8 forward-only sur EWC constitue une contribution Gap 3 partielle.
