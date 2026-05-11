# S13-02 — `MahalanobisDetector` : détection d'anomalies one-class (mode CL)

| Champ | Valeur |
|-------|--------|
| **ID** | S13-02 |
| **Sprint** | Sprint 13 |
| **Priorité** | 🔴 Critique |
| **Fichier source** | `src/models/unsupervised/mahalanobis_detector.py` |
| **Config** | `configs/unsupervised_config.yaml` — section `mahalanobis:` |
| **Expériences** | exp_103, exp_104, exp_105, exp_113, exp_117 |
| **Statut** | ✅ Complété |

---

## Principe

`MahalanobisDetector` modélise la distribution normale par une gaussienne multivariée (μ, Σ) et détecte les anomalies par distance de Mahalanobis :

```
score(x) = sqrt( (x − μ)ᵀ Σ⁻¹ (x − μ) )
```

Un échantillon est classifié comme anormal si `score(x) > threshold_`.

**Avantage clé** : le modèle stocke uniquement μ (d flottants) et Σ⁻¹ (d² flottants) — taille **fixe** indépendante du nombre d'échantillons d'entraînement.

---

## Empreinte mémoire

`count_parameters() = d + d²` (moyenne + matrice de covariance inverse) :

| Dataset | d | Paramètres | Modèle @ FP32 | Modèle @ INT8 |
|---------|---|:----------:|:-------------:|:-------------:|
| Monitoring | 4 | 20 | 80 B | 20 B |
| CWRU | 9 | 90 | 360 B | 90 B |
| Pronostia | 13 | 182 | 728 B | 182 B |

> **Ultra-compact** : le modèle Mahalanobis est **le plus petit du projet** toutes méthodes confondues. Sur Monitoring (d=4), 80 octets suffisent — facteur 60× plus petit que DBSCAN dans le même scénario.

---

## Stratégies CL

| `cl_strategy` | Comportement | Cas d'usage |
|--------------|--------------|-------------|
| `"welford"` (défaut anomaly detection) | Mise à jour incrémentale en ligne de μ et Σ via l'algorithme de Welford — aucun stockage des données passées | Online learning, MCU avec RAM limitée |
| `"refit"` | Recalcul complet de μ et Σ sur la tâche courante uniquement | Adaptation rapide à chaque tâche, oubli intentionnel |
| `"accumulate"` | Recalcul sur toutes les données vues (croissance mémoire) | Maximise l'accuracy ; PC uniquement |

### Welford — algorithme de mise à jour en ligne

L'algorithme de Welford permet de calculer μ et Σ de manière incrémentale, en un seul passage sur les données, sans stocker tous les échantillons :

```python
# Pour chaque échantillon x reçu :
n += 1
delta = x − mean
mean += delta / n
delta2 = x − mean
M2 += outer(delta, delta2)  # accumule les moments centrés
# À la fin : Σ = M2 / (n−1)
```

**Mise à jour CL** : à chaque nouvelle tâche, les statistiques Welford sont réinitialisées ou poursuivies selon `update_sigma_every`. Le modèle ne requiert aucun buffer de données passées — compatible MCU.

---

## Régularisation

Σ peut devenir singulière avec peu de données ou des features corrélées. Régularisation diagonale :

```python
Σ_reg = Σ + reg_covar × I    avec reg_covar = 1e-6
```

Cette valeur est suffisante pour les datasets testés. À augmenter si `np.linalg.inv(Σ)` lève une `LinAlgError`.

---

## Résultats expérimentaux

| Exp | Dataset | Scénario | Accuracy | AF | BWT | RAM peak | Latence | STM32N6 |
|-----|---------|----------|:--------:|:--:|:---:|:--------:|:-------:|:-------:|
| exp_103 | CWRU | by_fault_type | 0.160 | 0.026 | −0.026 | 1.3 KB | 0.0036 ms | ✅ |
| exp_104 | CWRU | by_severity | 0.195 | — | — | 1.3 KB | 0.0034 ms | ✅ |
| exp_105 | Pronostia | by_condition | 0.898 | 0.010 | −0.007 | 1.4 KB | 0.0036 ms | ✅ |
| exp_113 | Monitoring | by_equipment | 0.954 | 0.000 | +0.001 | 1.2 KB | 0.0061 ms | ✅ |
| exp_117 | Monitoring | by_location | 0.949 | — | — | 1.2 KB | 0.0060 ms | ✅ |

Nombre de mises à jour Welford enregistrées :

| Exp | Task 0 | Task 1 | Task 2 |
|-----|-------:|-------:|-------:|
| exp_105 (Pronostia) | 2 939 | 1 366 | 1 721 |
| exp_113 (Monitoring) | 2 027 | 2 052 | 2 058 |

### Analyse CWRU (exp_103–104)

Accuracy très faible (0.16–0.20) — le dataset CWRU contient ~20% de données normales, insuffisantes pour estimer μ et Σ de manière robuste. La distance de Mahalanobis est particulièrement sensible à l'estimation de Σ : avec peu d'échantillons normaux, Σ est mal conditionnée même avec régularisation.

### Analyse Pronostia (exp_105)

Accuracy 0.898 — AF faible (0.010) grâce à la stratégie Welford qui préserve les statistiques des tâches passées tout en s'adaptant aux nouvelles. Les 3 conditions (charge légère / normale / lourde) ont des distributions suffisamment distinctes pour que Mahalanobis les sépare.

### Analyse Monitoring (exp_113, exp_117)

Meilleurs résultats (0.954) avec **AF=0** — la stratégie Welford n'efface pas les statistiques accumulées entre tâches, résultant en un oubli catastrophique quasi-nul. Latence : 0.006 ms — **la plus rapide du projet**.

---

## Configuration

Section `mahalanobis:` dans `configs/unsupervised_config.yaml` :

```yaml
mahalanobis:
  ANOMALY_PERCENTILE: 95
  REG_COVAR: 1.0e-6
  CL_STRATEGY: "welford"
  WELFORD_MIN_SAMPLES: 10
  UPDATE_SIGMA_EVERY: 1
```

---

## Conclusion STM32N6

**Mahalanobis est le candidat prioritaire pour le portage STM32N6.**

- RAM modèle réel : 80–728 B @ FP32 selon le dataset (tous sous 64 Ko)
- RAM peak Python : 1.2–1.4 KB (overhead minimal)
- Latence : 0.003–0.006 ms — **meilleure latence du projet** (facteur 50× plus rapide que KMeans)
- Stratégie Welford : aucun buffer de données requis — compatible avec "pas d'accès dataset complet en RAM" (contrainte STM32N6)
- AF≈0 sur Monitoring — bonne rétention inter-tâches sans mémoire épisodique

> `FIXME(gap2)` : mesurer la RAM réelle sur STM32N6 — le tableau Σ⁻¹ (d²×4 octets) doit être alloué dans la SRAM statique.
> `FIXME(gap3)` : étudier la quantification INT8 de Σ⁻¹ — possible perte de précision sur les valeurs proches de zéro.
