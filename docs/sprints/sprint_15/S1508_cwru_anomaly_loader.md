# S15-13 — Loader `get_cwru_dataloaders_anomaly_detection()`

| Champ | Valeur |
|-------|--------|
| **ID** | S15-13 |
| **Sprint** | Sprint 15 |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 2.5h |
| **Dépendances** | — (`CWRUDataset` et `get_cwru_cl_dataloaders_by_severity()` déjà implémentés) |
| **Fichier cible** | `src/data/cwru_dataset.py` |

---

## Objectif

Implémenter `get_cwru_dataloaders_anomaly_detection()` : wrapper sur `CWRUDataset` qui produit, pour chaque tâche (by_severity ou by_fault_type), un DataLoader train contenant **uniquement les données normales** et un DataLoader test contenant **normaux + faulty** pour le calcul de l'AUROC.

---

## Contexte dataset CWRU

| Paramètre | Valeur |
|-----------|--------|
| Dataset | CWRU Bearing Dataset |
| Features | 9 (statistiques temporelles sur fenêtres de 2048 points) |
| Label original | catégorie de défaut (Normal, Ball_007/014/021, IR_007/014/021, OR_007/014/021) |
| Données normales | ~230 fenêtres / 2300 total (~10% normal) |
| Scénarios CL | by_severity (profondeur du défaut) + by_fault_type (type de défaut) |
| Fichier source | `data/raw/CWRU Bearing Dataset/feature_time_48k_2048_load_1.csv` |

---

## Différence fondamentale avec Pronostia

Contrairement à Pronostia, **le label binaire est issu du dataset** (classe `Normal_1` = 0, tout autre = 1). Il n'y a pas de `FAILURE_RATIO` à calculer — la frontière normal/faulty est structurelle, pas temporelle.

La difficulté inverse : **~10% de données normales** (vs ~90% pour Pronostia). Le train loader ne contient donc que ~62 échantillons normaux par tâche, ce qui est critique pour les modèles one-class.

---

## Interface

```python
def get_cwru_dataloaders_anomaly_detection(
    data_path: str | Path,
    scenario: Literal["by_fault_type", "by_severity"] = "by_severity",
    test_size: float = 0.2,
    batch_size: int = 32,
    seed: int = 42,
) -> list[dict]:
    """
    Loader anomaly detection one-class pour CWRU (scénario by_severity ou by_fault_type).

    Returns
    -------
    list[dict]
        Liste de 3 dicts (Tâche 0 → 1 → 2) avec les clés :
        {
            "task_id": int,
            "task_name": str,          # ex. "007" / "ball"
            "domain": str,             # identique à task_name
            "train_loader": DataLoader,      # label=0 uniquement, shuffle=True
            "test_loader_mixed": DataLoader, # label=0 + label=1, shuffle=False
            "n_train": int,
            "n_test": int,
            "n_test_normal": int,
            "n_test_faulty": int,
        }
    """
```

### Scénario by_severity — 3 tâches

| Tâche | Domaine | Défauts inclus |
|-------|---------|----------------|
| 0 | `007` | Ball_007 + IR_007 + OR_007 + Normal[0:77] |
| 1 | `014` | Ball_014 + IR_014 + OR_014 + Normal[77:154] |
| 2 | `021` | Ball_021 + IR_021 + OR_021 + Normal[154:] |

### Scénario by_fault_type — 3 tâches

| Tâche | Domaine | Défauts inclus |
|-------|---------|----------------|
| 0 | `ball` | Ball_007/014/021 + Normal[0:77] |
| 1 | `inner_race` | IR_007/014/021 + Normal[77:154] |
| 2 | `outer_race` | OR_007/014/021 + Normal[154:] |

---

## Implémentation

```python
def get_cwru_dataloaders_anomaly_detection(...):
    ds = CWRUDataset(data_path, random_state=seed)

    # Sélectionner l'ordre des tâches selon le scénario
    if scenario == "by_fault_type":
        task_order = FAULT_TYPE_ORDER
        task_fault_labels = FAULT_TYPE_LABELS
    elif scenario == "by_severity":
        task_order = SEVERITY_ORDER
        task_fault_labels = SEVERITY_LABELS

    # Répartition déterministe des normaux en 3 tiers (np.array_split)
    normal_mask = ds.fault_labels == NORMAL_LABEL
    X_normal = ds.X[normal_mask]
    normal_splits_X = np.array_split(X_normal, N_TASKS)

    scaler = None
    for task_id, task_name in enumerate(task_order):
        X_norm_task = normal_splits_X[task_id]

        # Split normal train/test (20%)
        n_test_norm = max(1, int(len(X_norm_task) * test_size))
        X_norm_train = X_norm_task[n_test_norm:]   # ~62 échantillons
        X_norm_test  = X_norm_task[:n_test_norm]   # ~15 échantillons

        # Toutes les fenêtres faulty de cette tâche vont en test
        faulty_mask = np.isin(ds.fault_labels, task_fault_labels[task_name])
        X_faulty = ds.X[faulty_mask]

        # StandardScaler fitté uniquement sur normaux train de la Tâche 0
        if task_id == 0:
            scaler = StandardScaler().fit(X_norm_train)
        X_norm_train = scaler.transform(X_norm_train).astype(np.float32)
        X_norm_test  = scaler.transform(X_norm_test).astype(np.float32)
        X_faulty     = scaler.transform(X_faulty).astype(np.float32)

        # Train loader : normaux uniquement
        # MEM: X_norm_train [~62, 9] × 4 B @ FP32 / [~62, 9] × 1 B @ INT8
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_norm_train),
                          torch.zeros(len(X_norm_train))),
            batch_size=batch_size, shuffle=True
        )

        # Test loader : normaux + faulty mélangés
        X_test = np.vstack([X_norm_test, X_faulty])
        y_test = np.hstack([np.zeros(len(X_norm_test)), np.ones(len(X_faulty))])
        test_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_test),
                          torch.from_numpy(y_test.astype(np.float32))),
            batch_size=batch_size, shuffle=False
        )
```

> **Note** : `shuffle=True` pour le train loader car les données CWRU ne sont pas ordonnées temporellement (contrairement à Pronostia). Le StandardScaler est fitté sur Tâche 0 uniquement pour éviter le data leakage — il s'applique ensuite à toutes les tâches suivantes.

---

## Warning intégré

```python
if len(X_norm_train) < 100:
    warnings.warn(
        f"Tâche {task_id} ({task_name!r}) : seulement {len(X_norm_train)} échantillons "
        f"normaux d'entraînement (CWRU ~10% normal). Les détecteurs one-class peuvent "
        f"être instables.",
        UserWarning,
    )
```

Ce warning se déclenche systématiquement (~62 < 100) — il est attendu et documenté.

---

## Critères d'acceptation

- [x] Retourne exactement 3 dicts `{task_id, task_name, train_loader, test_loader_mixed, n_train, n_test, n_test_normal, n_test_faulty}`
- [x] `train_loader` ne contient que des échantillons label=0 (~62 par tâche)
- [x] `test_loader_mixed` contient les deux classes (label=0 et label=1)
- [x] StandardScaler fitté sur Tâche 0 uniquement (pas de data leakage)
- [x] `X.shape[1] == 9` pour toutes les tâches
- [x] Fonctionne avec `scenario="by_severity"` et `scenario="by_fault_type"`

## Statut

✅ Terminé

## Bilan

`get_cwru_dataloaders_anomaly_detection()` implémentée à `src/data/cwru_dataset.py:403`. Les deux scénarios `by_severity` et `by_fault_type` sont supportés via le paramètre `scenario`. La répartition des ~230 normaux en 3 tiers déterministes (np.array_split) donne ~77 normaux/tâche → ~62 train, ~15 test. Le StandardScaler est fitté sur le train normal de la Tâche 0 uniquement. Le warning `< 100 normaux` se déclenche systématiquement — comportement attendu et documenté. Les 5 tests unitaires de `tests/test_cwru_anomaly.py` passent à 100% (voir S15-19).
