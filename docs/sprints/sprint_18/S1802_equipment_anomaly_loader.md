# S18-02 — Loader `get_equipment_monitoring_dataloaders_anomaly_detection()`

| Champ | Valeur |
|-------|--------|
| **ID** | S18-02 |
| **Sprint** | Sprint 18 |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 2h |
| **Dépendances** | S18-01 (scénario décidé) |
| **Fichier cible** | `src/data/equipment_monitoring_dataset.py` |

---

## Objectif

Implémenter `get_equipment_monitoring_dataloaders_anomaly_detection()` : wrapper sur les loaders Equipment Monitoring existants qui produit des DataLoaders train_normal / test_all pour le scénario d'anomaly detection one-class by_equipment_type.

---

## Contexte Equipment Monitoring anomaly detection

**Features** : 4D (température, pression, vibration, humidité)
**Types d'équipement** : Pump, Turbine, Compressor (3 tâches)
**Ratio normal** : ~50% — condition favorable par rapport à CWRU (~10%)
**Avantage** : suffisamment de données normales par tâche pour un entraînement stable

---

## Interface

```python
def get_equipment_monitoring_dataloaders_anomaly_detection(
    data_path: str | Path,
    scenario: Literal["by_equipment_type"] = "by_equipment_type",
    test_size: float = 0.2,
    batch_size: int = 32,
    seed: int = 42,
) -> list[tuple[int, str, DataLoader, DataLoader]]:
    """
    Retourne 3 tâches Equipment Monitoring pour anomaly detection one-class.

    Returns
    -------
    list of (task_id, task_name, train_loader_normal, test_loader_all)
        train_loader_normal : uniquement les échantillons normaux du type d'équipement courant
        test_loader_all     : Normal (subset) + faulty du type d'équipement courant
    """
```

---

## Logique de split

### Scénario by_equipment_type

```
Tâche 0 ("pump") :
    train = Normal_pump (train split, ~80% des normaux Pump)
    test  = Normal_pump_test (~20%) + Faulty_pump (tous)

Tâche 1 ("turbine") :
    train = Normal_turbine (train split)
    test  = Normal_turbine_test + Faulty_turbine

Tâche 2 ("compressor") :
    train = Normal_compressor (train split)
    test  = Normal_compressor_test + Faulty_compressor
```

> Chaque tâche est indépendante (refit) ou cumulative (accumulate selon le paramètre de `run_anomaly_detection.py`).

---

## Critères d'acceptation

- [ ] Retourne 3 tuples `(task_id, task_name, train_loader, test_loader)`
- [ ] `train_loader` ne contient que des échantillons avec label=0 (Normal)
- [ ] `test_loader` contient Normal + Faulty (vérifiable avec `unique(y)`)
- [ ] `X.shape[1] == 4` pour toutes les tâches
- [ ] Les noms de tâches correspondent aux types d'équipement (`"pump"`, `"turbine"`, `"compressor"`)
- [ ] Reproductible avec `seed=42`

## Statut

⬜ En attente S18-01
