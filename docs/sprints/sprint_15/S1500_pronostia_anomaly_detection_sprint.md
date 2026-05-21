# Sprint 15 — Anomaly Detection Pronostia + CWRU : 6 modèles × 2 datasets

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 15 |
| **Semaine** | 12–15 mai 2026 + extension CWRU 15–20 mai |
| **Statut** | ✅ CLÔTURÉ |
| **Priorité globale** | 🔴 Critique — datasets de roulements en anomaly detection |
| **Durée estimée totale** | ~25h |
| **Dépendances** | Sprint 14 terminé — `EWCOneClassDetector` implémenté, `run_anomaly_detection_scenario()` validé sur Monitoring |

---

## Objectif

Déployer les 6 modèles d'anomaly detection sur les deux datasets de roulements : **Pronostia** (~90% normal, 13D) et **CWRU** (~20% normal, 9D). Les deux scénarios CWRU (by_severity et by_fault_type) sont couverts car la question `TODO(arnaud)` a reçu une réponse implicite : les deux apportent des informations complémentaires.

---

## Partie A — Pronostia FEMTO (by_bearing_condition)

### Scénario retenu : by_bearing_condition

Scénario naturellement aligné avec la dégradation temporelle Pronostia (début de vie → fin de vie).

```
Tâche 0 (early_life)  : train = normaux début de vie (~95% du dataset)
                         test  = normaux test + anomalies précoces
Tâche 1 (mid_life)    : train = normaux milieu de vie
                         test  = normaux test + dégradation en cours
Tâche 2 (end_of_life) : train = normaux fin de vie (peu de données)
                         test  = normaux test + rupture imminente
```

| Condition | Ratio normal estimé | Difficulté one-class |
|-----------|:-------------------:|:--------------------:|
| Early life | ~95% | Facile (beaucoup de normaux) |
| Mid life | ~80% | Moyen |
| End of life | ~50% | Plus difficile |

Gradient de difficulté croissant → AUROC attendu croissant avec la sévérité de la dégradation.  
Point d'attention : 13 features spectrales — espace haute dimension, Mahalanobis peut être instable en end_of_life (peu de normaux). `REG_COVAR=1e-5` recommandé.

### Tâches Pronostia

| ID | Tâche | Priorité | Fichier(s) cible(s) | Statut |
|----|-------|:---:|---------------------|:------:|
| S15-01 | Implémenter `get_pronostia_dataloaders_anomaly_detection()` — train=normal, test=normal+faulty, 3 tâches by_condition | 🔴 | `src/data/pronostia_dataset.py` | ✅ |
| S15-02 | Mettre à jour `configs/unsupervised_config.yaml` avec bloc Pronostia (`INPUT_DIM=13`, `FAILURE_RATIO`, `condition_ids`) | 🔴 | `configs/unsupervised_config.yaml` | ✅ |
| S15-03 | exp_137 — HDC Pronostia by_condition refit | 🔴 | `experiments/exp_137/` | ✅ |
| S15-04 | exp_138 — TinyOL AE Pronostia by_condition refit | 🔴 | `experiments/exp_138/` | ✅ |
| S15-05 | exp_139 — KMeans Pronostia by_condition refit | 🔴 | `experiments/exp_139/` | ✅ |
| S15-06 | exp_140 — Mahalanobis Pronostia by_condition refit | 🔴 | `experiments/exp_140/` | ✅ |
| S15-07 | exp_141 — DBSCAN Pronostia by_condition refit | 🔴 | `experiments/exp_141/` | ✅ |
| S15-08 | exp_142 — EWC one-class Pronostia by_condition refit | 🔴 | `experiments/exp_142/` | ✅ |
| S15-09 | exp_137b–142b — 6 modèles Pronostia by_condition accumulate | 🟡 | `experiments/exp_137b–142b/` | ✅ |
| S15-10 | RAM profiling Pronostia (input_dim=13) | 🔴 | `evaluation/memory_profiler.py` | ✅ |
| S15-11 | Notebook Pronostia anomaly detection | 🟡 | `notebooks/cl_eval/pronostia_anomaly_detection/notebook_pronostia_anomaly_detection.ipynb` | ✅ |
| S15-12 | Tests `get_pronostia_dataloaders_anomaly_detection()` | 🟡 | `tests/test_pronostia_anomaly.py` | ✅ |

### Résultats Pronostia — 6 modèles refit (by_condition)

| Exp | Modèle | Stratégie | avg_AUROC |
|-----|--------|-----------|-----------|
| exp_137 | HDC | refit | 0.7231 |
| exp_138 | TinyOL AE | refit | 0.7268 |
| exp_139 | KMeans | refit | 0.7402 |
| exp_140 | Mahalanobis | refit | 0.6673 |
| exp_141 | DBSCAN | refit | 0.7034 |
| exp_142 | EWC one-class | refit | 0.7165 |
| exp_137b | HDC | accumulate | 0.7231 |
| exp_138b | TinyOL AE | accumulate | 0.7243 |
| exp_139b | KMeans | accumulate | 0.7243 |
| exp_140b | Mahalanobis | accumulate | 0.7101 |
| exp_141b | DBSCAN | accumulate | 0.7107 |
| exp_142b | EWC one-class | accumulate | 0.7224 |

> **Note** : les `config_snapshot.yaml` de exp_137–142 contiennent des exp_ids erronés (artefact de ré-exécution depuis des configs anciennes). Les `metrics_anomaly.json` sont la référence fiable — ils contiennent les bons `model`, `dataset`, `scenario`.

---

## Partie B — CWRU Bearing Dataset

### Contexte

La question `TODO(arnaud)` (sprint 13) sur le choix du scénario CWRU (by_severity vs by_fault_type) a été résolue en exécutant les deux — ils apportent des perspectives complémentaires :
- **by_severity** : modélise la dégradation progressive d'un défaut existant (drift temporel)
- **by_fault_type** : modélise l'apparition successive de types de défauts inconnus (déploiement incrémental)

### Tâches CWRU

| ID | Tâche | Priorité | Fichier(s) cible(s) | Statut |
|----|-------|:---:|---------------------|:------:|
| S15-13 | Implémenter `get_cwru_dataloaders_anomaly_detection()` — train=normal (classe "Time_Normal"), test=normal+faulty | 🔴 | `src/data/cwru_dataset.py` | ✅ |
| S15-14 | Mettre à jour `configs/unsupervised_config.yaml` avec bloc CWRU (INPUT_DIM=9) | 🔴 | `configs/unsupervised_config.yaml` | ✅ |
| S15-15 | exp_143–148 — 6 modèles CWRU by_severity refit | 🔴 | `experiments/exp_143/` → `experiments/exp_148/` | ✅ |
| S15-16 | exp_143b–148b — 6 modèles CWRU by_severity accumulate | 🟡 | `experiments/exp_143b–148b/` | ✅ |
| S15-17 | exp_149–154 — 6 modèles CWRU by_fault_type refit | 🔴 | `experiments/exp_149/` → `experiments/exp_154/` | ✅ |
| S15-18 | Notebook CWRU anomaly detection (by_severity + by_fault_type) | 🟡 | `notebooks/cl_eval/cwru_anomaly_detection/` | ✅ |
| S15-19 | Tests `get_cwru_dataloaders_anomaly_detection()` | 🟡 | `tests/test_cwru_anomaly.py` | ✅ |

### Résultats CWRU — 6 modèles refit

#### by_severity (exp_143–148)

| Exp | Modèle | Stratégie | avg_AUROC |
|-----|--------|-----------|-----------|
| exp_143 | HDC | refit | 0.9906 |
| exp_144 | TinyOL AE | refit | 1.0000 |
| exp_145 | KMeans | refit | 1.0000 |
| exp_146 | Mahalanobis | refit | 1.0000 |
| exp_147 | DBSCAN | refit | 1.0000 |
| exp_148 | EWC one-class | refit | 1.0000 |

#### by_fault_type (exp_149–154)

| Exp | Modèle | Stratégie | avg_AUROC |
|-----|--------|-----------|-----------|
| exp_149 | HDC | refit | 0.9934 |
| exp_150 | TinyOL AE | refit | 1.0000 |
| exp_151 | KMeans | refit | 1.0000 |
| exp_152 | Mahalanobis | refit | 1.0000 |
| exp_153 | DBSCAN | refit | 1.0000 |
| exp_154 | EWC one-class | refit | 1.0000 |

> **Note** : les `config_snapshot.yaml` de exp_143–154 contiennent des exp_ids erronés. Les `metrics_anomaly.json` sont la référence fiable.

---

## Partie C — Synthèse cross-dataset (Phase Anomaly Detection)

### Tâche

| ID | Tâche | Priorité | Fichier(s) cible(s) | Statut |
|----|-------|:---:|---------------------|:------:|
| S15-20 | Finaliser `summary_anomaly_detection.ipynb` — tableau AUROC complet 6 modèles × 3 datasets (Monitoring S14, Pronostia + CWRU S15), recommandations embarquées STM32N6 | 🔴 | `notebooks/cl_eval/summary_anomaly_detection.ipynb` | ✅ |

### Structure du notebook summary (livrable manuscrit)

Le notebook couvre les expériences de toute la Phase Anomaly Detection (Sprints 13–15) :

| Dataset | Scénario | Stratégie | Expériences |
|---------|----------|-----------|-------------|
| Equipment Monitoring | by_equipment, by_location | refit + accumulate | exp_086–089, exp_123–136 (Sprint 14) |
| Pronostia | by_bearing_condition | refit + accumulate | exp_137–142, exp_137b–142b |
| CWRU | by_severity | refit + accumulate | exp_143–148, exp_143b–148b |
| CWRU | by_fault_type | refit | exp_149–154 |

Sections clés du notebook :
1. **Tableau AUROC cross-dataset** — 6 modèles × 3 datasets × stratégie
2. **Impact ratio normal** — scatter AUROC vs ratio normal (~10% CWRU / ~50% Monitoring / ~90% Pronostia)
3. **Impact dimensionnalité** — scatter AUROC vs input_dim (4D / 9D / 13D)
4. **RAM cross-dataset** — barplot `ram_peak_bytes`, ligne de référence 64 Ko
5. **Recommandations embarquées** — meilleur modèle par critère (AUROC, RAM, robustesse)
6. **Position Triple Gap** — Gap 1 (données industrielles réelles ✅), Gap 2 (RAM < 64 Ko ✅ pour KMeans/Mahalanobis/HDC), Gap 3 (INT8 pendant entraînement → Sprint 16+)

### Clarification exp_155–160

> Ces dossiers **ne contiennent pas** d'expériences Pronostia anomaly detection.  
> Contenu réel : CWRU classification supervisée (`metrics_cl.json`) — runs déjà couverts dans Sprint 13.  
> exp_160 : export ONNX/INT8 EWC — livrable Sprint 16 (MCU portage).

---

## Bilan Sprint 15

| Tâche | Statut | Notes |
|-------|:------:|-------|
| Pronostia loader | ✅ | `get_pronostia_dataloaders_anomaly_detection()` — 3 tâches by_condition |
| exp_137–142 Pronostia refit | ✅ | AUROC 0.67–0.74 (cas difficile : ratio 90% normal mais peu de données faulty test) |
| exp_137b–142b Pronostia accumulate | ✅ | AUROC similaire au refit |
| CWRU loader | ✅ | `get_cwru_dataloaders_anomaly_detection()` — scénarios by_severity et by_fault_type |
| exp_143–148 CWRU by_severity refit | ✅ | AUROC 0.99–1.00 (cas favorable malgré seulement ~20% normal) |
| exp_143b–148b CWRU by_severity accumulate | ✅ | AUROC identique au refit |
| exp_149–154 CWRU by_fault_type refit | ✅ | AUROC 0.99–1.00 |
| Notebooks bearing datasets | ✅ | `cwru_anomaly_detection/` + `pronostia_anomaly_detection/` |
| Tests | ✅ | `test_pronostia_anomaly.py`, `test_cwru_anomaly.py` |

**Résultat clé** : CWRU anomaly detection est le cas le plus favorable malgré le faible ratio de données normales (20%). Les features spectrales très discriminantes permettent une séparation quasi-parfaite. Pronostia (AUROC ~0.72) est plus difficile — les features agrégées sur des séquences longues lissent les signatures de défaut précoce.

---

## Sous-documents

**Pronostia (Partie A) :**
- [S1501_pronostia_anomaly_loader.md](S1501_pronostia_anomaly_loader.md)
- [S1502_pronostia_config_update.md](S1502_pronostia_config_update.md)
- [S1503_exp137_142_pronostia_refit.md](S1503_exp137_142_pronostia_refit.md)
- [S1504_ram_profiling_pronostia.md](S1504_ram_profiling_pronostia.md)
- [S1505_exp137b_142b_accumulate.md](S1505_exp137b_142b_accumulate.md)
- [S1506_notebook_pronostia.md](S1506_notebook_pronostia.md)
- [S1507_tests_pronostia_anomaly.md](S1507_tests_pronostia_anomaly.md)

**CWRU (Partie B) :**

- [S1508_cwru_anomaly_loader.md](S1508_cwru_anomaly_loader.md)
- [S1509_cwru_config_update.md](S1509_cwru_config_update.md)
- [S1510_exp143_148_cwru_severity_refit.md](S1510_exp143_148_cwru_severity_refit.md)
- [S1511_exp143b_148b_cwru_severity_accumulate.md](S1511_exp143b_148b_cwru_severity_accumulate.md)
- [S1512_exp149_154_cwru_fault_type_refit.md](S1512_exp149_154_cwru_fault_type_refit.md)
- [S1513_notebook_cwru.md](S1513_notebook_cwru.md)
- [S1514_tests_cwru_anomaly.md](S1514_tests_cwru_anomaly.md)

**Synthèse (Partie C) — contenu absorbé depuis Sprint 19 (supprimé) :**
- Notebook summary : `notebooks/cl_eval/summary_anomaly_detection.ipynb`
- Specs détaillées scénario Pronostia by_bearing_condition : voir S1901 (supprimé, contenu intégré ci-dessus)
- Specs summary cross-dataset : voir S1906 (supprimé, contenu intégré ci-dessus)
- Tests Pronostia anomaly detection : `tests/test_pronostia_anomaly.py`

---

> **Après ce sprint** : Sprint 16 (portage MCU) clôturé ✅. La Phase Anomaly Detection (Sprints 13–15) est complète sur les 3 datasets : Monitoring (S14), Pronostia + CWRU (S15). Sprint 19 supprimé — contenu intégré ici.
