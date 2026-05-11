# S18-01 — Décision scénario CL Equipment Monitoring anomaly detection

| Champ | Valeur |
|-------|--------|
| **ID** | S18-01 |
| **Sprint** | Sprint 18 |
| **Priorité** | 🔴 Critique (bloquante pour S18-02) |
| **Durée estimée** | 0.5h |
| **Dépendances** | — |
| **Fichier cible** | `docs/context/datasets.md` |

---

## Objectif

Documenter et valider le scénario CL retenu pour Equipment Monitoring Anomaly Detection : **by_equipment_type** (Pump → Turbine → Compressor). Cette décision est directement alignée avec le scénario supervisé déjà utilisé dans les expériences Equipment Monitoring Phase 1.

---

## Analyse du scénario by_equipment_type

### Structure des 3 tâches

```
Tâche 0 ("pump") :
    train = Normal_pump (~50% des données Pump)
    test  = Normal_pump_test + Faulty_pump

Tâche 1 ("turbine") :
    train = Normal_turbine (~50% des données Turbine)
    test  = Normal_turbine_test + Faulty_turbine

Tâche 2 ("compressor") :
    train = Normal_compressor (~50% des données Compressor)
    test  = Normal_compressor_test + Faulty_compressor
```

**Avantages** :
- Scénario industriel réaliste : le système est d'abord déployé sur des pompes, puis étendu à d'autres types d'équipements
- Cohérent avec les expériences supervisées Equipment Monitoring (domaine incrémental par type)
- Ratio ~50% normal par tâche — conditions favorables pour le one-class learning
- Les 4 features (température, pression, vibration, humidité) sont communes à tous les types → transfert possible

**Points d'attention** :
- Vérifier la distribution des features par type d'équipement — les normaux Pump et Turbine peuvent avoir des distributions différentes (drift de domaine inter-tâches)
- Avec ~50% de normaux, le budget RAM accumulate reste gérable même pour les modèles paramétriques

### Ratio normal par dataset

| Dataset | Ratio normal | Normaux/tâche (estimé) | Contexte |
|---------|:---:|:---:|--------|
| CWRU | ~10% | ~77 | Cas défavorable |
| Equipment Monitoring | ~50% | ~500+ | Cas favorable |
| Pronostia | ~90% | ~large | Cas très favorable (début de vie) |

---

## Action après décision

Documenter dans `docs/context/datasets.md` :

```markdown
### Equipment Monitoring — Scénario Anomaly Detection

Scénario retenu : **by_equipment_type**
Tâches : Pump (t0) → Turbine (t1) → Compressor (t2)
Ratio normal : ~50% (favorable pour one-class learning)
Features : 4D (température, pression, vibration, humidité)
Justification : alignement avec scénario supervisé Phase 1 + réalisme industriel
```

---

## Critères d'acceptation

- [ ] Scénario `by_equipment_type` documenté dans `docs/context/datasets.md`
- [ ] Tableau ratio normal/faulty par type d'équipement présent
- [ ] `SPLIT_STRATEGY: "by_equipment_type"` défini pour `configs/unsupervised_config.yaml` (S18-03)

## Statut

⬜ À faire
