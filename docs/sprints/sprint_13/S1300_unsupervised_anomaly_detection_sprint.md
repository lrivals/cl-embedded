# Sprint 13 — Détection d'anomalies non supervisée : KMeans, Mahalanobis, DBSCAN

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 13 |
| **Semaine** | 28 avril – 5 mai 2026 |
| **Priorité globale** | 🔴 Critique — Phase Anomaly Detection (Sprints 13–16) |
| **Durée estimée totale** | ~10h |
| **Dépendances** | Sprint 12 terminé — loaders CWRU validés ; unsupervised_config.yaml existant (Sprint 5) |

---

## Objectif

Adapter les trois détecteurs non supervisés existants (KMeans, Mahalanobis, DBSCAN) au scénario **one-class anomaly detection** : entraînement uniquement sur données normales, évaluation sur normal + défaillant. Lancer les premières expériences CL sur les datasets Monitoring et Pronostia.

**Critère de succès** : exp_100–111 enregistrées avec `metrics_cl.json` ; RAM peak mesurée pour KMeans, Mahalanobis et DBSCAN ; décision STM32N6-compatibilité documentée.

---

## Tâches

| ID | Tâche | Priorité | Fichier(s) cible(s) | Durée est. | Statut |
|----|-------|:---:|---------------------|:---:|--------|
| S13-01 | Adapter `KMeansDetector` — seuil EMA inter-tâches, stratégie refit/accumulate | 🔴 | `src/models/unsupervised/kmeans_detector.py` | 3h | ✅ |
| S13-02 | Adapter `MahalanobisDetector` — Welford online update, régularisation Σ | 🔴 | `src/models/unsupervised/mahalanobis_detector.py` | 3h | ✅ |
| S13-03 | Intégrer `DBSCANDetector` — scoring par distance aux core points, profiling RAM | 🔴 | `src/models/unsupervised/dbscan_detector.py` | 2h | ✅ |
| S13-04 | exp_100–111 — KMeans + Mahalanobis + DBSCAN × Monitoring + Pronostia (refit) | 🔴 | `experiments/exp_100–111/` | 2h | ✅ |
| S13-05 | Profiling RAM et latence — 3 modèles × 3 datasets, comparaison limite 64 Ko | 🟡 | `evaluation/memory_profiler.py` | 1h | ✅ |

---

## Numérotation expériences

### Monitoring — by_equipment et by_location

| Exp | Modèle | Dataset | Scénario | Stratégie |
|-----|--------|---------|----------|-----------|
| exp_100 | KMeans | CWRU | by_fault_type | refit |
| exp_101 | KMeans | CWRU | by_severity | refit |
| exp_102 | KMeans | Pronostia | by_condition | refit |
| exp_103 | Mahalanobis | CWRU | by_fault_type | welford |
| exp_104 | Mahalanobis | CWRU | by_severity | welford |
| exp_105 | Mahalanobis | Pronostia | by_condition | welford |
| exp_112 | KMeans | Monitoring | by_equipment | refit |
| exp_113 | Mahalanobis | Monitoring | by_equipment | welford |
| exp_116 | KMeans | Monitoring | by_location | refit |
| exp_117 | Mahalanobis | Monitoring | by_location | welford |
| exp_120 | DBSCAN | Monitoring | by_equipment | refit |
| exp_121 | DBSCAN | Monitoring | by_location | refit |
| exp_122 | DBSCAN | Pronostia | by_condition | refit |

---

## Bilan Sprint 13

| Tâche | Statut | Notes |
|-------|:------:|-------|
| S13-01 KMeans adaptation | ✅ | EMA alpha=0.3, K sélection silhouette par défaut |
| S13-02 Mahalanobis Welford | ✅ | reg_covar=1e-6, welford_min_samples=10 |
| S13-03 DBSCAN intégration | ✅ | eps auto-estimé via k-NN elbow |
| S13-04 Expériences | ✅ | exp_100–105, 112–113, 116–117, 120–122 |
| S13-05 RAM profiling | ✅ | KMeans 5.4 KB ✅ / Mahalanobis 1.3 KB ✅ / DBSCAN 73–121 KB ❌ |

**Résultat clé** : KMeans et Mahalanobis valident la contrainte 64 Ko STM32N6. DBSCAN écarté du portage embarqué (voir [S1303_dbscan_ram_disqualification.md](S1303_dbscan_ram_disqualification.md)).

---

## Sous-documents

- [S1301_kmeans_anomaly_detection.md](S1301_kmeans_anomaly_detection.md) — KMeans : architecture, CL, résultats
- [S1302_mahalanobis_anomaly_detection.md](S1302_mahalanobis_anomaly_detection.md) — Mahalanobis : Welford, RAM ultra-compact
- [S1303_dbscan_ram_disqualification.md](S1303_dbscan_ram_disqualification.md) — DBSCAN : analyse RAM, décision d'exclusion STM32N6

---

## Questions ouvertes résolues (Sprint 15)

- ~~`TODO(arnaud)` : scénario CWRU anomaly detection — by_fault_type ou by_severity ?~~ → **Résolu Sprint 15** : les deux scénarios ont été exécutés (exp_143–148 by_severity, exp_149–154 by_fault_type). AUROC 0.99–1.00 dans les deux cas.
- `TODO(dorra)` : DBSCAN avec CoreSet borné pour rester sous 64 Ko ? → DBSCAN écarté du portage (73–121 Ko, voir [S1303_dbscan_ram_disqualification.md](S1303_dbscan_ram_disqualification.md))
- ~~`TODO(fred)` : seuil AUROC acceptable pour portage industriel ?~~ → **Résolu Sprint 15** : CWRU AUROC ≥ 0.99 (5 modèles sur 6), Pronostia AUROC ~0.72. Le seuil de 0.90 est atteint sur CWRU ; Pronostia reste en dessous.
