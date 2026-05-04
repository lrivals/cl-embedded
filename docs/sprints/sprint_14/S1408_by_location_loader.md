# S14-08 — Extension loader Monitoring pour scénario by_location

| Champ | Valeur |
|-------|--------|
| **ID** | S14-08 |
| **Sprint** | Sprint 14 |
| **Priorité** | 🟡 Important |
| **Durée estimée** | 1.5h |
| **Dépendances** | — (`get_cl_dataloaders_by_location()` déjà implémenté) |
| **Fichier cible** | `src/data/monitoring_dataset.py` |

---

## Objectif

Étendre `get_cl_dataloaders_anomaly_detection()` pour accepter `scenario="by_location"` en plus de `scenario="by_equipment"`. La fonction doit retourner des DataLoaders avec split train_normal / test_all sur les 5 tâches de localisation.

---

## Interface attendue

```python
# Appel existant (by_equipment, 3 tâches)
loaders = get_cl_dataloaders_anomaly_detection(
    data_path="data/processed/equipment_monitoring/",
    scenario="by_equipment",
    test_size=0.2,
    batch_size=32,
    seed=42,
)

# Nouvel appel (by_location, 5 tâches)
loaders = get_cl_dataloaders_anomaly_detection(
    data_path="data/processed/equipment_monitoring/",
    scenario="by_location",   # nouveau paramètre accepté
    test_size=0.2,
    batch_size=32,
    seed=42,
)

# Retour dans les deux cas
for task_id, task_name, train_loader_normal, test_loader_all in loaders:
    # train_loader_normal : uniquement label=0 (normal) pour entraînement one-class
    # test_loader_all     : normal + faulty pour évaluation AUROC
    ...
```

---

## Implémentation

La logique de split by_location existe déjà dans `get_cl_dataloaders_by_location()`. Il s'agit de :

1. Appeler `get_cl_dataloaders_by_location()` pour obtenir les tâches brutes
2. Filtrer `X_train[y_train == 0]` pour le loader d'entraînement one-class
3. Conserver `X_test` complet (normal + faulty) pour l'évaluation

```python
def get_cl_dataloaders_anomaly_detection(
    data_path: str | Path,
    scenario: Literal["by_equipment", "by_location"] = "by_equipment",
    test_size: float = 0.2,
    batch_size: int = 32,
    seed: int = 42,
) -> list[tuple[int, str, DataLoader, DataLoader]]:
    ...
```

---

## Scénario by_location — 5 tâches

| Tâche | Location | Description |
|-------|----------|-------------|
| 0 | `HVAC` | Systèmes de climatisation |
| 1 | `Conveyor` | Convoyeurs industriels |
| 2 | `Pump` | Pompes |
| 3 | `Compressor` | Compresseurs |
| 4 | `Generator` | Générateurs |

> Vérifier les noms exacts de locations dans le dataset Monitoring traité.

---

## Critères d'acceptation

- [ ] `get_cl_dataloaders_anomaly_detection(scenario="by_location")` retourne exactement 5 tâches
- [ ] `train_loader_normal` ne contient que des échantillons avec label=0
- [ ] `test_loader_all` contient normal + faulty (ratio conforme au dataset)
- [ ] L'appel `scenario="by_equipment"` reste inchangé (non-régression)
- [ ] Pas de modification de `get_cl_dataloaders_by_location()` (wrapper, pas refactoring)

## Statut

⬜ À faire
