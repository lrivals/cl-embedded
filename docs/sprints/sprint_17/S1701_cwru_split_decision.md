# S17-01 — Décision scénario CL CWRU anomaly detection

| Champ | Valeur |
|-------|--------|
| **ID** | S17-01 |
| **Sprint** | Sprint 17 |
| **Priorité** | 🔴 Critique (bloquante pour S17-02) |
| **Durée estimée** | 0.5h |
| **Dépendances** | Réponse `TODO(arnaud)` avant le 18 mai 2026 |
| **Fichier cible** | `docs/context/datasets.md` |

---

## Objectif

Trancher et documenter le scénario CL retenu pour CWRU Anomaly Detection : **by_fault_type** ou **by_severity**. Cette décision est bloquante pour l'implémentation du loader (S17-02).

---

## Analyse des deux options

### Option A — by_fault_type (3 tâches : Ball → Inner Race → Outer Race)

```
Tâche 1 : train = Normal_t1 (sous-ensemble),  test = Normal + Ball_all_severities
Tâche 2 : train = Normal_t2 (sous-ensemble),  test = Normal + InnerRace_all_severities
Tâche 3 : train = Normal_t3 (sous-ensemble),  test = Normal + OuterRace_all_severities
```

**Avantages** :
- Modélise un scénario industriel réaliste : le système rencontre successivement de nouveaux types de défauts
- Aligné avec le scénario by_fault_type déjà utilisé en classification supervisée (exp_074–079)
- Pertinent pour l'articulation avec le manuscrit (même découpage)

**Inconvénients** :
- En anomaly detection, les 3 types de défauts sont structurellement différents — le détecteur doit apprendre "tout sauf X" pour chaque tâche
- Peu de données normales d'entraînement par tâche (~77 échantillons normaux / 3)

### Option B — by_severity (3 tâches : 0.007" → 0.014" → 0.021")

```
Tâche 1 : train = Normal_t1 (sous-ensemble),  test = Normal + défauts_sévérité_0.007"
Tâche 2 : train = Normal_t2 (sous-ensemble),  test = Normal + défauts_sévérité_0.014"
Tâche 3 : train = Normal_t3 (sous-ensemble),  test = Normal + défauts_sévérité_0.021"
```

**Avantages** :
- Modélise la dégradation progressive d'un défaut — plus naturel pour un détecteur one-class
- Cohérent avec Pronostia (dégradation temporelle en fin de vie)
- La sévérité croissante rend les anomalies de plus en plus détectables → progression naturelle des AUROC

**Inconvénients** :
- Moins représentatif d'un déploiement incrémental (les types de défauts sont mélangés par tâche)
- Moins cohérent avec le scénario supervisé CL by_severity (exp_080–085)

---

## Recommandation provisoire

**Option B (by_severity)** semble plus adapté à l'anomaly detection one-class car :
1. La sévérité croissante crée un gradient de difficulté naturel (AUROC attendu croissant)
2. La cohérence avec la dégradation temporelle Pronostia facilite la comparaison cross-dataset
3. Le drift progressif est plus représentatif d'un vrai système industriel en dégradation

> Confirmer avec Arnaud avant implémentation.

---

## Action après décision

Une fois le scénario confirmé, documenter dans `docs/context/datasets.md` :

```markdown
### CWRU — Scénario Anomaly Detection

Scénario retenu : **[by_fault_type | by_severity]**
Justification : [...]
Nombre de tâches : 3
Données normales : ~77 échantillons par tâche (classe "Time_Normal" répartie en 3)
```

---

## Critères d'acceptation

- [ ] Scénario CL retenu documenté dans `docs/context/datasets.md`
- [ ] Justification du choix rédigée (≥ 2 phrases)
- [ ] `SPLIT_STRATEGY` défini pour `configs/unsupervised_config.yaml` (S17-03)

## Statut

⬜ En attente réponse TODO(arnaud)
