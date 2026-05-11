# S19-01 — Décision scénario CL Pronostia anomaly detection

| Champ | Valeur |
|-------|--------|
| **ID** | S19-01 |
| **Sprint** | Sprint 19 |
| **Priorité** | 🔴 Critique (bloquante pour S19-02) |
| **Durée estimée** | 0.5h |
| **Dépendances** | — |
| **Fichier cible** | `docs/context/datasets.md` |

---

## Objectif

Documenter le scénario CL retenu pour Pronostia Anomaly Detection : **by_bearing_condition** (dégradation par roulement). Ce scénario est naturellement aligné avec la nature des données Pronostia (dégradation temporelle jusqu'à rupture).

---

## Analyse du scénario by_bearing_condition

### Structure des 3 tâches

```
Tâche 0 ("early_life") :
    train = Normal_early (début de vie, ~90% des données Pronostia)
    test  = Normal_early_test + Faulty_early (anomalies précoces)

Tâche 1 ("mid_life") :
    train = Normal_mid (milieu de vie)
    test  = Normal_mid_test + Faulty_mid (dégradation en cours)

Tâche 2 ("end_of_life") :
    train = Normal_eol (fin de vie — peu de données normales à ce stade)
    test  = Normal_eol_test + Faulty_eol (rupture imminente)
```

**Avantages** :
- Scénario industriel naturel : modélise l'évolution d'un roulement dans le temps
- Cohérent avec le cadre RUL (Remaining Useful Life) de Pronostia
- Gradient de difficulté croissant (anomalies plus marquées en fin de vie → AUROC attendu croissant)

**Points d'attention** :
- Ratio ~90% normal : beaucoup de données d'entraînement en early_life, moins en end_of_life
- 13 features (spectrales + temporelles) : espace de dimension élevée → Mahalanobis peut être instable sans régularisation suffisante
- Les features Pronostia sont déjà normalisées dans le loader Sprint 15

### Ratio normal par condition

| Condition | Ratio normal estimé | Normaux/tâche | Difficulté one-class |
|-----------|:---:|:---:|:---:|
| Early life | ~95% | large | Facile (beaucoup de normaux) |
| Mid life | ~80% | moyen | Moyen |
| End of life | ~50% | faible | Plus difficile |

---

## Action après décision

Documenter dans `docs/context/datasets.md` :

```markdown
### Pronostia — Scénario Anomaly Detection

Scénario retenu : **by_bearing_condition**
Tâches : Early life (t0) → Mid life (t1) → End of life (t2)
Ratio normal global : ~90% (début de vie très représenté)
Features : 13D (spectrales + temporelles vibration)
Justification : dégradation temporelle naturelle + gradient de difficulté croissant
```

---

## Critères d'acceptation

- [ ] Scénario `by_bearing_condition` documenté dans `docs/context/datasets.md`
- [ ] Tableau ratio normal/faulty par condition présent
- [ ] `SPLIT_STRATEGY: "by_bearing_condition"` défini pour `configs/unsupervised_config.yaml` (S19-03)

## Statut

⬜ À faire
