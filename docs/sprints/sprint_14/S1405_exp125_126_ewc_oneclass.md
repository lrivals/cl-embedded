# S14-05 — exp_125–126 : EWC one-class Monitoring by_equipment refit + accumulate

| Champ | Valeur |
|-------|--------|
| **ID** | S14-05 |
| **Sprint** | Sprint 14 |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 0.5h |
| **Dépendances** | S14-01, S14-02 |
| **Fichiers cibles** | `experiments/exp_125/`, `experiments/exp_126/` |

---

## Objectif

Exécuter `EWCOneClassDetector` sur Monitoring by_equipment dans les deux variantes stratégiques (refit = réinitialise les paramètres entre tâches, accumulate = conserve et pénalise via EWC).

**Note** : pour `EWCOneClassDetector`, la stratégie "accumulate" correspond à conserver les poids entre tâches **avec pénalité EWC active** (`lambda_ewc > 0`). La stratégie "refit" correspond à réinitialiser les poids à chaque tâche (`lambda_ewc = 0` ou reset complet du modèle).

---

## Expériences

### exp_125 — EWC one-class Monitoring by_equipment refit

```bash
python scripts/run_anomaly_detection.py \
    --model ewc_oneclass \
    --dataset monitoring \
    --scenario by_equipment \
    --strategy refit \
    --config configs/ewc_oneclass_config.yaml \
    --exp_id exp_125
```

### exp_126 — EWC one-class Monitoring by_equipment accumulate (EWC actif)

```bash
python scripts/run_anomaly_detection.py \
    --model ewc_oneclass \
    --dataset monitoring \
    --scenario by_equipment \
    --strategy accumulate \
    --config configs/ewc_oneclass_config.yaml \
    --exp_id exp_126
```

---

## Sorties attendues

```
experiments/exp_125/
├── config_snapshot.yaml
└── results/
    └── metrics_anomaly.json

experiments/exp_126/
├── config_snapshot.yaml
└── results/
    └── metrics_anomaly.json
```

### Comparaison attendue

| Métrique | exp_125 (refit) | exp_126 (accumulate EWC) |
|----------|-----------------|--------------------------|
| AUROC tâche 1 | — | devrait ≥ exp_125 tâche 1 (pas d'oubli grâce EWC) |
| AUROC moyen | référence | objectif > refit grâce à la rétention |
| avg_forgetting | référence | objectif < refit |

---

## Critères d'acceptation

- [ ] exp_125 et exp_126 : `metrics_anomaly.json` présents
- [ ] `config_snapshot.yaml` présents dans les deux répertoires
- [ ] La pénalité EWC est effective en exp_126 (avg_forgetting < exp_125 ou log loss EWC > 0)
- [ ] `ram_peak_bytes` reporté dans les deux expériences

## Statut

⬜ À faire
