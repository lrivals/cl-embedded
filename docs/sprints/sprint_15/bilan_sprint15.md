# Bilan Sprint 15 — Anomaly Detection Pronostia + CWRU (6 modèles × 2 datasets × 3 scénarios)

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 15 |
| **Semaine** | 12–20 mai 2026 |
| **Statut global** | ✅ CLOSED (S15-01 → S15-20) |
| **Expériences** | exp_137 – exp_154 + suffixes b (36 expériences) |
| **Dépendances Sprint 16** | KMeans / Mahalanobis / EWC OC / TinyOL AE candidats portage MCU |

---

## 1. Tâches

| ID | Livrable | Priorité | Statut |
|----|----------|:--------:|:------:|
| S15-01 | Loader Pronostia anomaly (3 tâches by_condition) | 🔴 | ✅ |
| S15-02 | Config Pronostia (`INPUT_DIM=13`, condition_ids) | 🔴 | ✅ |
| S15-03→08 | exp_137–142 — 6 modèles Pronostia refit | 🔴 | ✅ |
| S15-09 | exp_137b–142b — 6 modèles Pronostia accumulate | 🟡 | ✅ |
| S15-10 | RAM profiling Pronostia (input_dim=13) | 🔴 | ✅ |
| S15-11 | Notebook Pronostia anomaly detection | 🟡 | ✅ |
| S15-12 | Tests `get_pronostia_dataloaders_anomaly_detection()` | 🟡 | ✅ |
| S15-13 | Loader CWRU anomaly (by_severity + by_fault_type) | 🔴 | ✅ |
| S15-14 | Config CWRU (`INPUT_DIM=9`) | 🔴 | ✅ |
| S15-15 | exp_143–148 — 6 modèles CWRU by_severity refit | 🔴 | ✅ |
| S15-16 | exp_143b–148b — 6 modèles CWRU by_severity accumulate | 🟡 | ✅ |
| S15-17 | exp_149–154 — 6 modèles CWRU by_fault_type refit | 🔴 | ✅ |
| S15-18 | Notebook CWRU (by_severity + by_fault_type) | 🟡 | ✅ |
| S15-19 | Tests `get_cwru_dataloaders_anomaly_detection()` | 🟡 | ✅ |
| S15-20 | Notebook summary cross-dataset (Sprints 13–15) | 🔴 | ✅ |

**Couverture : 20/20 tâches ✅**

---

## 2. Expériences lancées

| Exp ID | Modèle | Dataset | Scénario | Stratégie |
|--------|--------|---------|----------|-----------|
| exp_137–142 | HDC, TinyOL AE, KMeans, Mahalanobis, DBSCAN, EWC OC | Pronostia | by_condition | refit |
| exp_137b–142b | idem | Pronostia | by_condition | accumulate |
| exp_143–148 | idem | CWRU | by_severity | refit |
| exp_143b–148b | idem | CWRU | by_severity | accumulate |
| exp_149–154 | idem | CWRU | by_fault_type | refit |

> **Artefact à noter** : les `config_snapshot.yaml` de exp_137–154 contiennent des exp_ids erronés (artefact de ré-exécution depuis des configs anciennes). **Référence fiable : `metrics_anomaly.json`** dans chaque dossier d'expérience.

---

## 3. Résultats — Métriques clés

### Pronostia (by_condition, input_dim=13)

| Modèle | avg_AUROC refit | avg_AUROC accum | Δ accum−refit | AF (refit) | RAM | ≤ 64 Ko ? |
|--------|:---------------:|:---------------:|:-------------:|:----------:|:---:|:---------:|
| KMeans | **0.7402** | 0.7243 | −0.0159 | +0.027 | 5 698 B (5.6 Ko) | ✅ |
| TinyOL AE | 0.7268 | 0.7243 | −0.0025 | +0.183 | 1 992 B (1.9 Ko) | ✅ |
| HDC | 0.7231 | 0.7231 | 0.0000 | −0.018 | 15 272 B (14.9 Ko) | ✅ |
| EWC one-class | 0.7165 | 0.7224 | +0.0059 | +0.185 | 1 480 B (1.4 Ko) | ✅ |
| DBSCAN | 0.7034 | 0.7107 | +0.0073 | +0.106 | 201 746 B (**197 Ko**) | ❌ |
| Mahalanobis | 0.6673 | 0.7101 | **+0.0429** | +0.205 | 1 756 B (1.7 Ko) | ✅ |

> AF > 0 = oubli entre tâches. L'AF élevé en refit (TinyOL AE +0.183, Mahalanobis +0.205, EWC OC +0.185) est intrinsèque à la stratégie refit — non un défaut du modèle. HDC seul montre un transfert positif (AF = −0.018).

### CWRU by_severity (input_dim=9, ~62 normaux/tâche)

| Modèle | avg_AUROC refit | avg_AUROC accum | Δ | AF | RAM | ≤ 64 Ko ? |
|--------|:---------------:|:---------------:|:-:|:---:|:---:|:---------:|
| TinyOL AE | **1.0000** | 1.0000 | 0.0000 | 0.000 | 1 992 B (1.9 Ko) | ✅ |
| KMeans | **1.0000** | 1.0000 | 0.0000 | 0.000 | 5 432 B (5.3 Ko) | ✅ |
| Mahalanobis | **1.0000** | 1.0000 | 0.0000 | 0.000 | 1 644 B (1.6 Ko) | ✅ |
| DBSCAN | **1.0000** | 1.0000 | 0.0000 | 0.000 | 10 674 B (10.4 Ko) | ✅ |
| EWC one-class | **1.0000** | 1.0000 | 0.0000 | 0.000 | 1 480 B (1.4 Ko) | ✅ |
| HDC | 0.9906 | 0.9906 | 0.0000 | −0.003 | 8 104 B (7.9 Ko) | ✅ |

### CWRU by_fault_type (refit uniquement)

| Modèle | avg_AUROC | AF | RAM | Latence |
|--------|:---------:|:---:|:---:|:-------:|
| TinyOL AE | **1.0000** | 0.000 | 1 992 B | 0.098 ms |
| KMeans | **1.0000** | 0.000 | 5 340 B | 0.501 ms |
| Mahalanobis | **1.0000** | 0.000 | 1 644 B | 0.008 ms |
| DBSCAN | **1.0000** | 0.000 | 10 674 B | 0.220 ms |
| EWC one-class | **1.0000** | 0.000 | 1 480 B | 0.084 ms |
| HDC | 0.9934 | −0.004 | 8 104 B | 0.088 ms |

### Impact stratégie refit vs accumulate — synthèse cross-dataset

| Dataset | Modèles avec Δ ≠ 0 | Amplitude max | Conclusion |
|---------|--------------------|:-------------:|------------|
| Monitoring (S14) | EWC OC (+0.0052) | +0.5 pp | Accumulate marginalement utile pour EWC OC |
| Pronostia (S15) | Mahalanobis (+0.0429) | +4.3 pp | Accumulate aide Mahalanobis — mais AUROC reste le plus bas |
| CWRU (S15) | Aucun | 0.0 pp | Accumulate totalement indifférent |

---

## 4. Points saillants

**1. Résultat contre-intuitif : ratio normal ≠ difficulté one-class.**
CWRU (~10–20% normal) atteint AUROC=1.000 pour 5/6 modèles. Pronostia (~90% normal) plafonne à AUROC~0.72. La discrimination ne dépend pas du ratio de données normales, mais de la qualité des features : les 9 features spectrales CWRU encodent directement les fréquences caractéristiques des défauts, rendant la séparation triviale. Les 13 features agrégées de Pronostia lissent les signatures précoces, rendant la tâche intrinsèquement plus difficile.

**2. Stratégie accumulate : confirmation de l'inutilité généralisée.**
Sur l'ensemble des datasets testés (Monitoring S14, Pronostia, CWRU), l'accumulate n'apporte de gain mesurable que pour EWC OC sur Monitoring (+0.5 pp) et Mahalanobis sur Pronostia (+4.3 pp — depuis l'AUROC le plus bas). La conclusion opérationnelle est acquise : **refit seul suffit pour tous les modèles candidats au portage MCU**, ce qui élimine le besoin d'un buffer cumulatif embarqué.

**3. RAM DBSCAN : exclusion MCU confirmée sur Pronostia, conditionnelle sur CWRU.**
La RAM DBSCAN est proportionnelle au volume d'entraînement (N_train × input_dim × 4 B), pas à input_dim seul. Sur Pronostia (~3 000 points × 13D) : 197 Ko ❌. Sur CWRU (~62 points × 9D) : 10.4 Ko ✅. **DBSCAN ne peut pas être déployé sur STM32N6 dans un contexte de dataset volumineux.** Ce chiffre est mesuré par tracemalloc — citable dans le manuscrit.

**4. HDC systématiquement en retrait.**
Sur les 3 datasets, HDC est le seul modèle à ne jamais atteindre AUROC=1.000 (CWRU : 0.9906/0.9934, Pronostia : 0.7231, rang 3/6). L'architecture hyperdimenionnelle montre une limite intrinsèque sur ces datasets spectraux, malgré une RAM acceptable (7.9–14.9 Ko selon input_dim).

**5. Mahalanobis : instabilité en haute dimension avec peu de normaux.**
AF = +0.205 (le plus élevé) et AUROC = 0.6673 (le plus bas) sur Pronostia 13D. L'instabilité de la matrice de covariance en end_of_life (peu de normaux disponibles) est à mentionner dans le manuscrit. `REG_COVAR=1e-5` a été appliqué — si l'instabilité persiste, c'est un signal architectural, pas un bug de config.

---

## 5. Bilan technique

Le sprint 15 clôture la **Phase Anomaly Detection** du projet (Sprints 13–15). Les 20 tâches sont livrées, 36 expériences sont reproductibles. Les livrables concrets : deux loaders standardisés (`get_pronostia_dataloaders_anomaly_detection()`, `get_cwru_dataloaders_anomaly_detection()`), leurs configs YAML avec overrides par dataset (INPUT_DIM, EPSILON, N_CLUSTERS), trois notebooks d'analyse (Pronostia, CWRU, summary cross-dataset), deux suites de tests.

**Chiffres-clés pour le manuscrit :**
- Pronostia : KMeans 0.74 avg_AUROC (meilleur), Mahalanobis 0.67 (pire) — 5/6 modèles sous 64 Ko RAM, DBSCAN exclu (197 Ko)
- CWRU : 5/6 modèles à AUROC=1.000 sur les deux scénarios — tous les 6 sous 64 Ko y compris DBSCAN (10.4 Ko)
- Latences : toutes sous 1 ms (0.008 ms pour Mahalanobis, 0.501 ms pour KMeans) — contrainte 100 ms largement respectée

---

## 6. Analyse par gaps

- **Gap 1 ✅ (données industrielles réelles)** — Couverture complète sur 3 datasets industriels (Monitoring, Pronostia, CWRU), 36 expériences reproductibles. Le résultat ratio_normal ≠ difficulté est directement publiable dans la section "Anomaly Detection". Le scatter AUROC vs ratio_normal (section 2 du notebook summary) est un résultat original.

- **Gap 2 ✅ (< 100 Ko RAM mesurés)** — 5/6 modèles respectent la contrainte sur Pronostia (input_dim=13), 6/6 sur CWRU. Les valeurs mesurées par tracemalloc sont dans les `metrics_anomaly.json` de chaque expérience. EWC one-class (1.4 Ko) et TinyOL AE (1.9 Ko) sont les plus frugaux.

- **Gap 3 ❌ (quantification INT8)** — Non adressé. Les annotations `# MEM:` préparent le terrain, mais aucune expérience INT8 n'a été conduite. **À traiter en priorité en Sprint 16.**

---

## 7. Recommandations pour Sprint 16

**1. Candidats MCU à retenir pour le portage.**
KMeans (5.6 Ko, AUROC 0.74/1.00), Mahalanobis (1.7 Ko, AUROC 0.67/1.00), EWC one-class (1.4 Ko, AUROC 0.72/1.00) et TinyOL AE (1.9 Ko, AUROC 0.73/1.00) passent tous la contrainte RAM sur les deux datasets. Pour le portage STM32N6, **EWC one-class et TinyOL AE sont prioritaires** : implémentations paramétriques compatibles avec une backpropagation sur Cortex-M55 (Objectif MCU du projet).

**2. Artefact exp_ids à corriger.**
Les `config_snapshot.yaml` de exp_137–154 contiennent des exp_ids erronés. Avant de consolider les tables de résultats du manuscrit, corriger le `experiments_tracker.md` pour aligner les exp_ids réels avec les `metrics_anomaly.json`.

**3. Quantification INT8 — urgence Gap 3.**
Trois sprints d'anomaly detection ont préparé le terrain, mais Gap 3 reste ouvert. Sprint 16 doit initier au moins une expérience de quantification post-entraînement sur EWC one-class ou TinyOL AE. Compte tenu de la deadline manuscrit (15 avril 2026, déjà dépassée au 13 mai 2026), ce point est à traiter en urgence absolue.

**4. Décision DBSCAN définitive.**
DBSCAN est exclu du portage MCU pour les datasets à volume réaliste. La décision peut être actée dans le manuscrit avec les chiffres mesurés (197 Ko sur Pronostia). Pas besoin de relancer des expériences DBSCAN en Sprint 16.
