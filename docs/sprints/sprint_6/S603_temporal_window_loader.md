# S6-02 — Implémenter `get_pump_dataloaders_by_temporal_window()`

| Champ | Valeur |
|-------|--------|
| **ID** | S6-02 |
| **Sprint** | Sprint 6 — Phase 1 Extension (≥ 15 avril 2026) |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 3h |
| **Dépendances** | S3-02 (`pump_dataset.py` existant avec `get_pump_dataloaders_by_id()`) |
| **Fichiers cibles** | `src/data/pump_dataset.py` |
| **Complété le** | — |
| **Statut** | ⬜ À faire |

---

## Objectif

Ajouter `get_pump_dataloaders_by_temporal_window()` dans `src/data/pump_dataset.py` pour
le scénario CL **domain-incremental par fenêtres temporelles** sur le Dataset 1.

Contrairement au scénario `get_pump_dataloaders()` (3 tâches chronologiques), cette fonction
découpe les 20 000 entrées en **4 quartiles de 5 000 entrées** triées par `Operational_Hours`,
modélisant l'évolution temporelle de la machine (rodage → usure → vieillissement → pré-panne).

**Questions scientifiques ciblées** :
- Les quartiles temporels présentent-ils un domain shift suffisant pour provoquer de l'oubli ?
- Ce scénario à 4 tâches est-il plus difficile que le scénario à 3 tâches ? Que le scénario par Pump_ID ?
- `FIXME(gap1)` : les AA très basses sur Dataset 1 (~0.50) persistent-elles sur 4 tâches temporelles ?

**Critère de succès** : la fonction retourne exactement 4 dicts avec la clé `temporal_window`,
et le loader est compatible avec tous les scripts d'entraînement existants.

---

## Constantes à ajouter

Ajouter après la définition de `N_TASKS` dans la section constantes de `pump_dataset.py` :

```python
# Nombre de tâches CL — scénario temporel (4 quartiles)
N_TEMPORAL_TASKS: int = 4

# Entrées par tâche dans le scénario temporel (20 000 / 4)
ENTRIES_PER_TEMPORAL_TASK: int = 5000
```

---

## Signature et docstring

```python
def get_pump_dataloaders_by_temporal_window(
    csv_path: str | Path,
    normalizer_path: str | Path,
    n_tasks: int = N_TEMPORAL_TASKS,
    entries_per_task: int = ENTRIES_PER_TEMPORAL_TASK,
    batch_size: int = 32,
    val_ratio: float = VAL_RATIO,
    seed: int = 42,
    window_size: int = WINDOW_SIZE,
    step_size: int = STEP_SIZE,
) -> list[dict]:
    """
    Crée un scénario CL domain-incremental par fenêtres temporelles.

    Découpe les 20 000 entrées en ``n_tasks`` tranches de ``entries_per_task`` lignes
    chacune, triées par Operational_Hours (ordre chronologique global, tous Pump_ID mélangés).

      T1 : lignes 0–4 999    (Operational_Hours les plus basses)
      T2 : lignes 5 000–9 999
      T3 : lignes 10 000–14 999
      T4 : lignes 15 000–19 999

    Applique le même feature engineering que ``get_pump_dataloaders()`` :
    fenêtrage WINDOW_SIZE=32, STEP_SIZE=16, 6 stats × 4 canaux + temporal_position.

    Normalisation Z-score ajustée sur T1 uniquement (``pump_normalizer.yaml`` préajusté).

    Split train/val **temporel** (chronologique, pas stratifié) pour respecter l'ordre
    causal : les `val_ratio` dernières fenêtres de chaque tâche constituent le set de
    validation (simulation du comportement MCU — pas de shuffle).

    Parameters
    ----------
    csv_path : str | Path
        Chemin complet vers le CSV pump maintenance.
    normalizer_path : str | Path
        Chemin vers ``pump_normalizer.yaml`` (normaliseur ajusté sur Task 1 chronologique).
    n_tasks : int
        Nombre de tâches (quartiles). Default : 4.
    entries_per_task : int
        Nombre de lignes CSV par tâche. Default : 5 000.
    batch_size : int
        Taille de batch pour les DataLoaders.
    val_ratio : float
        Fraction validation (split temporel, pas stratifié). Default : 0.2.
    seed : int
        Seed reproductibilité. Default : 42.
    window_size : int
        Taille fenêtre glissante. Default : WINDOW_SIZE (32).
    step_size : int
        Pas entre fenêtres. Default : STEP_SIZE (16).

    Returns
    -------
    list[dict]
        Liste de ``n_tasks`` dicts (un par quartile temporel) :

        .. code-block:: python

            {
                "task_id": int,            # 1..n_tasks
                "temporal_window": int,    # idem task_id (alias sémantique)
                "train_loader": DataLoader,
                "val_loader": DataLoader,
                "n_train": int,
                "n_val": int,
            }
    """
```

---

## Algorithme d'implémentation

```python
def get_pump_dataloaders_by_temporal_window(...) -> list[dict]:
    set_seed(seed)

    # 1. Chargement + tri chronologique global (PumpMaintenanceDataset.load() trie par operational_hours)
    ds = PumpMaintenanceDataset(Path(csv_path))
    full_df = ds.load()  # déjà trié par operational_hours croissant

    # 2. Chargement du normaliseur (ajusté sur T1 chronologique)
    normalizer = load_pump_normalizer(Path(normalizer_path))
    mean_vec = normalizer["mean"]  # [N_FEATURES]
    std_vec = normalizer["std"]    # [N_FEATURES]

    result: list[dict] = []

    for task_idx in range(n_tasks):
        # 3. Découper la tranche de entries_per_task lignes
        start = task_idx * entries_per_task
        end = start + entries_per_task
        slice_df = full_df.iloc[start:end].copy().reset_index(drop=True)

        # 4. Extraction features via fenêtrage glissant (même pipeline que get_pump_dataloaders)
        original_df = ds._df
        ds._df = slice_df
        X, y = ds.extract_features(window_size=window_size, step_size=step_size)
        ds._df = original_df  # restaurer l'état original

        # 5. Normalisation Z-score avec stats fixes
        # MEM: X [N_windows, 25] × 4B = N_windows × 100 B @ FP32
        X = (X - mean_vec) / std_vec

        # 6. Split train/val temporel (pas de shuffle — ordre causal respecté)
        n = len(X)
        n_val = max(1, int(n * val_ratio))
        train_idx = np.arange(n - n_val)
        val_idx = np.arange(n - n_val, n)

        x_train = torch.from_numpy(X[train_idx].astype(np.float32))
        y_train = torch.from_numpy(y[train_idx].astype(np.float32)).unsqueeze(1)
        x_val   = torch.from_numpy(X[val_idx].astype(np.float32))
        y_val   = torch.from_numpy(y[val_idx].astype(np.float32)).unsqueeze(1)

        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(TensorDataset(x_val, y_val),   batch_size=batch_size, shuffle=False)

        result.append({
            "task_id": task_idx + 1,
            "temporal_window": task_idx + 1,
            "train_loader": train_loader,
            "val_loader": val_loader,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
        })

    return result
```

---

## Critères d'acceptation

- [ ] `get_pump_dataloaders_by_temporal_window()` importable depuis `src.data.pump_dataset`
- [ ] Retourne exactement `n_tasks=4` dicts
- [ ] Chaque dict contient les clés : `task_id`, `temporal_window`, `train_loader`, `val_loader`, `n_train`, `n_val`
- [ ] `temporal_window` ∈ {1, 2, 3, 4} et correspond à `task_id`
- [ ] `n_train + n_val` correspond au nombre total de fenêtres de la tranche (pas de fuite)
- [ ] `ruff check` + `black --check` passent

---

## Artefacts produits

| Fichier | Chemin | Commitable |
|---------|--------|:----------:|
| Loader additionnel | `src/data/pump_dataset.py` (fonction ajoutée après `get_pump_dataloaders_by_id`) | ✅ |
| Tests | `tests/test_pump_dataset.py` (section S6-11) | ✅ |

---

## Commandes de vérification

```bash
python -c "
from src.data.pump_dataset import get_pump_dataloaders_by_temporal_window
tasks = get_pump_dataloaders_by_temporal_window(
    csv_path='data/raw/pump_maintenance/Large_Industrial_Pump_Maintenance_Dataset/Large_Industrial_Pump_Maintenance_Dataset.csv',
    normalizer_path='configs/pump_normalizer.yaml'
)
assert len(tasks) == 4, f'Expected 4 tasks, got {len(tasks)}'
for t in tasks:
    assert 'temporal_window' in t
    print(f'Window {t[\"temporal_window\"]}: {t[\"n_train\"]} train, {t[\"n_val\"]} val')
print('OK')
"
pytest tests/test_pump_dataset.py -v -k temporal_window
```

---

## Questions ouvertes

- `TODO(arnaud)` : Faut-il trier les 20 000 entrées globalement par `Operational_Hours` avant le
  découpage (scénario actuel), ou respecter l'ordre par Pump_ID au sein de chaque quartile ?
  L'interprétation scientifique diffère : global = drift de la flotte, par-pompe = drift individuel.
- `TODO(fred)` : Les quartiles temporels (0–5 000, 5 001–10 000, etc.) correspondent-ils à des
  phases d'exploitation réelles (rodage, fonctionnement nominal, usure avancée) ? Cette information
  changerait le nommage des tâches dans les notebooks.
- `FIXME(gap1)` : Les AA très basses sur Dataset 1 (~0.50) persistent-elles sur le scénario
  temporel à 4 tâches ? Si oui, le dataset est peut-être trop homogène pour distinguer les phases.
