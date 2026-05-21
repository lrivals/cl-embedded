# S15-01 — Loader `get_pronostia_dataloaders_anomaly_detection()`

| Champ | Valeur |
|-------|--------|
| **ID** | S15-01 |
| **Sprint** | Sprint 15 |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 2.5h |
| **Dépendances** | — (`get_pronostia_dataloaders()` by_condition déjà implémenté) |
| **Fichier cible** | `src/data/pronostia_dataset.py` |

---

## Objectif

Implémenter `get_pronostia_dataloaders_anomaly_detection()` : wrapper sur le loader Pronostia existant qui produit, pour chaque tâche by_condition, un DataLoader train contenant **uniquement les données normales** (début de vie du roulement) et un DataLoader test contenant **toutes les données** (normal + fin de vie marquée faulty).

---

## Contexte dataset Pronostia

| Paramètre | Valeur |
|-----------|--------|
| Dataset | FEMTO Bearing (Pronostia) |
| Features | 13 (statistiques temporelles + fréquentielles) |
| Label original | TTF / run-to-failure (pas de label binaire direct) |
| Conditions | 3 (C1: 1800 tr/min 4 kN, C2: 1650 tr/min 4.2 kN, C3: 1500 tr/min 5 kN) |
| Roulements par condition | 2 (train), 7 (test) — variable selon condition |
| Proportion normale | ~90% (dégradation rapide en fin de vie) |

---

## Stratégie de labelling anomaly detection

La dégradation Pronostia est temporelle : les roulements sont normaux en début de vie et dégradent en fin de vie. Le label binaire est construit comme suit :

```
label = 1 (faulty)  si  t ≥ (1 - FAILURE_RATIO) × T_total
label = 0 (normal)  si  t < (1 - FAILURE_RATIO) × T_total
```

Avec `FAILURE_RATIO = 0.10` par défaut (configurable dans `configs/unsupervised_config.yaml`).

---

## Interface

```python
def get_pronostia_dataloaders_anomaly_detection(
    data_path: str | Path,
    failure_ratio: float = 0.10,
    test_size: float = 0.2,
    batch_size: int = 32,
    seed: int = 42,
) -> list[tuple[int, str, DataLoader, DataLoader]]:
    """
    Retourne 3 tâches (by_condition) pour anomaly detection one-class.

    Returns
    -------
    list of (task_id, condition_name, train_loader_normal, test_loader_all)
        train_loader_normal : uniquement label=0 (données normales, début de vie)
        test_loader_all     : label=0 + label=1 (pour évaluation AUROC)
    """
```

### Scénario by_condition — 3 tâches

| Tâche | Condition | Vitesse | Charge |
|-------|-----------|---------|--------|
| 0 | `condition_1` | 1800 tr/min | 4 kN |
| 1 | `condition_2` | 1650 tr/min | 4.2 kN |
| 2 | `condition_3` | 1500 tr/min | 5 kN |

---

## Implémentation

```python
def get_pronostia_dataloaders_anomaly_detection(...):
    tasks = get_pronostia_dataloaders(data_path, scenario="by_condition", ...)

    result = []
    for task_id, condition_name, X_all, y_ttf in tasks:
        # Construire label binaire anomaly
        T = len(X_all)
        threshold_idx = int((1 - failure_ratio) * T)
        y_anomaly = np.zeros(T, dtype=np.int8)
        y_anomaly[threshold_idx:] = 1

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_anomaly, test_size=test_size, random_state=seed, shuffle=False
        )  # shuffle=False : respecter l'ordre temporel

        # Train loader : uniquement normaux
        X_train_normal = X_train[y_train == 0]
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train_normal)),
            batch_size=batch_size, shuffle=True
        )

        # Test loader : tous (normal + faulty)
        test_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
            batch_size=batch_size, shuffle=False
        )

        result.append((task_id, condition_name, train_loader, test_loader))

    return result
```

> **Note** : `shuffle=False` pour le split train/test est crucial — les données Pronostia sont ordonnées temporellement. Un shuffle aléatoire mélangerait début et fin de vie, rendant le label binaire incohérent.

---

## Critères d'acceptation

- [x] Retourne exactement 3 tuples `(task_id, condition_name, train_loader, test_loader)`
- [x] `train_loader` ne contient que des échantillons avec y=0 (vérifiable : `all(y == 0)`)
- [x] `test_loader` contient les deux classes y=0 et y=1
- [x] Le split est temporel (`shuffle=False`) — les données normales précèdent les données faulty
- [x] `FAILURE_RATIO` est bien pris en compte (test avec 0.05 et 0.20 pour vérification)
- [x] `X.shape[1] == 13` pour toutes les tâches

## Statut

✅ Terminé

## Bilan

`get_pronostia_dataloaders_anomaly_detection()` implémentée à `src/data/pronostia_dataset.py:456`. Les 6 tests unitaires de `tests/test_pronostia_anomaly.py` passent à 100% (voir S15-07). Le label binaire est construit par seuil temporel avec `FAILURE_RATIO=0.10`, le split train/test respecte l'ordre chronologique (`shuffle=False`), et les 3 tâches by_condition sont retournées avec `input_dim=13`.
