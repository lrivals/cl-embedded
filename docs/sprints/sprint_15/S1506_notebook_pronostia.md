# S15-06 — Notebook Pronostia Anomaly Detection

| Champ | Valeur |
|-------|--------|
| **ID** | S15-06 |
| **Sprint** | Sprint 15 |
| **Priorité** | 🟡 Important |
| **Durée estimée** | 3h |
| **Dépendances** | S15-03 → S15-08 (+ S15-05 si temps) |
| **Fichier cible** | `notebooks/cl_eval/pronostia_anomaly_detection/notebook_pronostia_anomaly_detection.ipynb` |

---

## Objectif

Produire le notebook d'analyse Pronostia Anomaly Detection : comparaison des 6 modèles, analyse de l'impact du FAILURE_RATIO, et visualisation des scores d'anomalie dans le temps (spécifique Pronostia).

---

## Structure du notebook

### Section 1 — Chargement résultats

Chargement des `metrics_anomaly.json` de exp_137–142 (et 137b–142b si disponibles).

### Section 2 — Tableau AUROC synthèse

| Modèle | Tâche 1 (C1) | Tâche 2 (C2) | Tâche 3 (C3) | Moyenne |
|--------|:------------:|:------------:|:------------:|:-------:|
| HDC | — | — | — | — |
| TinyOL AE | — | — | — | — |
| KMeans | — | — | — | — |
| Mahalanobis | — | — | — | — |
| DBSCAN | — | — | — | — |
| EWC one-class | — | — | — | — |

### Section 3 — Visualisation score temporel (spécifique Pronostia)

Pour le meilleur modèle : courbe du score d'anomalie au cours du temps sur une séquence Pronostia. Le score doit augmenter en fin de vie (validation qualitative du détecteur).

```python
# Charger une séquence complète Pronostia condition 1
X_seq, y_seq = load_pronostia_sequence(condition=1, bearing=1)
scores = best_model.predict_score(X_seq)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
ax1.plot(scores, label="Score anomalie")
ax1.axhline(threshold, color="r", linestyle="--", label="Seuil")
ax1.set_title("Score d'anomalie dans le temps — Pronostia C1 Bearing 1")
ax2.plot(y_seq, label="Label réel (0=normal, 1=faulty)")
```

### Section 4 — Comparaison RAM Pronostia vs Monitoring

Barplot RAM (13D vs 4D) pour les modèles dont la taille varie (HDC, EWC one-class, Mahalanobis).

### Section 5 — Impact FAILURE_RATIO (si S15-13 fait)

Courbe AUROC moyen (KMeans) en fonction de FAILURE_RATIO ∈ {0.05, 0.10, 0.20}.

### Section 6 — Conclusions Pronostia

Points de synthèse pour le manuscrit :
- Quel modèle est le plus robuste sur Pronostia (3 tâches, 13D, 90% normal) ?
- Le ratio élevé de données normales favorise-t-il les modèles one-class ?
- Comparaison qualitatif avec les résultats Monitoring (4D, 3 tâches, ~50% normal)

---

## Figures à sauvegarder

```
notebooks/figures/anomaly_detection/pronostia/
├── auroc_table_pronostia.png
├── score_temporal_best_model.png
├── ram_comparison_pronostia_vs_monitoring.png
└── failure_ratio_sensitivity.png   # si S15-13 fait
```

---

## Critères d'acceptation

- [ ] Notebook exécutable end-to-end sans erreur
- [ ] Tableau AUROC présent avec toutes les valeurs remplies (exp_137–142)
- [ ] Figure score temporel présente (section 3)
- [ ] Section "Conclusions Pronostia" rédigée avec ≥ 3 observations pour le manuscrit

## Statut

⬜ À faire
