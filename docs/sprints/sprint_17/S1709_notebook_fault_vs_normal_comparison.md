# S17-09 — Notebook comparaison fault detection mode vs normal mode

| Champ | Valeur |
|-------|--------|
| **ID** | S17-09 |
| **Sprint** | Sprint 17 |
| **Priorité** | 🟡 Important |
| **Durée estimée** | 2h |
| **Dépendances** | S17-05 (notebook CWRU anomaly detection terminé, exp_143–148 disponibles) |
| **Fichier cible** | `notebooks/cl_eval/cwru_anomaly_detection/notebook_fault_vs_normal_mode_comparison.ipynb` |

---

## Objectif

Comparer sur CWRU les performances obtenues en **mode anomaly detection** (entraînement one-class sur données normales uniquement) avec celles obtenues en **mode fault detection supervisé** (entraînement avec les labels de pannes). L'objectif est de quantifier le coût de ne pas avoir de labels de défauts, et d'identifier les hypothèses explicatives des écarts observés.

Ce notebook produit une analyse utile pour le manuscrit : il situe les résultats non-supervisés par rapport à une borne supérieure supervisée, et justifie les limitations du mode anomaly detection sur CWRU.

---

## Structure du notebook

### Section 1 — Rappel des deux modes

- **Mode anomaly detection** (non-supervisé) : modèle entraîné uniquement sur les données normales → score d'anomalie → AUROC (résultats exp_143–148)
- **Mode fault detection supervisé** : modèle entraîné avec labels (Normal vs faulty) → classification binaire → AUROC ou accuracy

Tableau récapitulatif des différences de protocole :

| | Anomaly Detection | Fault Detection |
|--|:-:|:-:|
| Labels d'entraînement | Normal uniquement | Normal + faulty |
| Paradigme | One-class / reconstruction | Supervisé binaire |
| Données requises | Peu (normaux seuls) | Plus (les deux classes) |
| Applicable sans historique de pannes | ✅ | ❌ |
| Implémenté dans le projet | exp_143–148 | Baseline S17-09 |

### Section 2 — Baseline supervisée : évaluation des 6 modèles

Pour chaque modèle, adapter ou reconfigurer en mode supervisé binaire et évaluer sur CWRU :

```python
# Exemples de baselines supervisées par modèle
# HDC : encoder les deux classes, classifier par distance au prototype
# KMeans : 2 clusters (normal, faulty), assign par proximité
# EWC : MLP binaire avec EWC, entraîné sur Normal + faulty
# TinyOL AE : comparer le MSE entre reconstruction normal et faulty (déjà en mode anomaly)
# Mahalanobis : distance de Mahalanobis depuis le centre de chaque classe
# DBSCAN : non applicable en supervisé → remplacer par KNN (k=5) comme proxy
```

> Note : certains modèles (TinyOL AE, DBSCAN) ne se prêtent pas directement au mode supervisé. Utiliser un proxy raisonnable et documenter le choix.

### Section 3 — Tableau comparatif AUROC

| Modèle | AUROC Anomaly Detection | AUROC Fault Detection | Écart (Δ) |
|--------|:-:|:-:|:-:|
| HDC | — | — | — |
| TinyOL AE | — | — | — |
| KMeans | — | — | — |
| Mahalanobis | — | — | — |
| DBSCAN / proxy | — | — | — |
| EWC one-class / MLP | — | — | — |

### Section 4 — Analyse et explication des écarts

Hypothèses à explorer et documenter pour chaque écart significatif (Δ > 0.1) :

1. **Ratio 20% normal** : avec ~77 échantillons normaux par tâche, le seuil de reconstruction (percentile 95) est instable — haute variance du threshold entre tâches
2. **Distribution des scores d'anomalie** : visualiser l'overlap entre scores normaux et faulty (histogramme ou violinplot) pour détecter si le recouvrement est structurellement élevé
3. **Dimensionnalité** : CWRU à 9D est entre Monitoring (4D) et Pronostia (13D) — l'espace de features est-il discriminant pour les défauts ?
4. **Nature du signal** : les features CWRU (vibration fréquentielle) sont-elles plus informatives en supervisé qu'en reconstruction ?
5. **Sévérité vs type** : si le scénario est by_severity, les données inter-tâches sont plus similaires → difficulté intrinsèque pour une approche one-class

```python
# Visualisation : distributions des scores par mode
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, model_name in zip(axes.flat, model_names):
    ax.hist(scores_normal[model_name], alpha=0.6, label="Normal", bins=30)
    ax.hist(scores_faulty[model_name], alpha=0.6, label="Faulty", bins=30)
    ax.set_title(model_name)
    ax.legend()
```

### Section 5 — Conclusions pour le manuscrit

Points clés à rédiger :

- Quantification du coût de l'absence de labels : Δ AUROC moyen entre les deux modes
- Quel modèle est le plus affecté ? Lequel est le plus robuste ?
- Dans quel scénario industriel le mode anomaly detection reste-t-il pertinent malgré l'écart ? (absence totale de données de pannes en démarrage de production)
- Recommandation : seuil AUROC anomaly detection au-delà duquel le mode non-supervisé est suffisant sans labels

---

## Figures à sauvegarder

```
notebooks/figures/anomaly_detection/cwru/
├── auroc_comparison_fault_vs_normal_mode.png   # barplot comparatif
├── score_distributions_by_model.png             # histogrammes overlap Section 4
└── auroc_gap_analysis.png                       # Δ AUROC par modèle avec annotations
```

---

## Critères d'acceptation

- [ ] Notebook exécutable end-to-end sans erreur
- [ ] Tableau comparatif AUROC (Section 3) rempli pour les 6 modèles
- [ ] Section 4 présente avec ≥ 3 hypothèses analysées et visualisations
- [ ] Section 5 "Conclusions manuscrit" rédigée avec recommandation claire sur le mode anomaly detection
- [ ] Figures sauvegardées dans `notebooks/figures/anomaly_detection/cwru/`

## Statut

⬜ À faire (après S17-05)
