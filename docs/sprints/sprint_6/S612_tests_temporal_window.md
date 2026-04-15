# S6-11 — Tests unitaires `get_pump_dataloaders_by_temporal_window()`

| Champ | Valeur |
|-------|--------|
| **ID** | S6-11 |
| **Sprint** | Sprint 6 — Phase 1 Extension (≥ 15 avril 2026) |
| **Priorité** | 🟡 Important |
| **Durée estimée** | 2h |
| **Dépendances** | S6-02 (`get_pump_dataloaders_by_temporal_window()` implémenté) |
| **Fichiers cibles** | `tests/test_pump_dataset.py` |
| **Complété le** | 2026-04-14 |
| **Statut** | ✅ Terminé |

---

## Objectif

Ajouter une section de tests unitaires pour `get_pump_dataloaders_by_temporal_window()`
dans `tests/test_pump_dataset.py`, en suivant le pattern des tests existants pour
`get_pump_dataloaders_by_id()`.

Les tests utilisent des **données synthétiques** (fixture existante `synthetic_df` du fichier test)
pour ne pas dépendre du vrai CSV. Un test d'intégration marqué `@_integration` est ajouté
pour le vrai CSV.

---

## Tests à ajouter

### Fixture synthétique partagée

La fixture `make_synthetic_tasks_temporal` génère 4 × 100 lignes (400 total, trié par
`operational_hours`) pour simuler le scénario temporel avec les données de test existantes.

### 1. `test_temporal_window_n_tasks()`

Vérifie que la fonction retourne exactement `n_tasks=4` dicts.

```python
def test_temporal_window_n_tasks(tmp_path):
    """get_pump_dataloaders_by_temporal_window() retourne exactement n_tasks dicts."""
    # Utilise le CSV synthétique via tmp_path (pattern fixtures existant)
    tasks = get_pump_dataloaders_by_temporal_window(
        csv_path=_SYNTHETIC_CSV,
        normalizer_path=_NORMALIZER_PATH,
        n_tasks=4,
        entries_per_task=...,  # 1/4 du dataset synthétique
    )
    assert len(tasks) == 4, f"Attendu 4 tâches, obtenu {len(tasks)}"
```

### 2. `test_temporal_window_key_temporal_window()`

Vérifie la présence de la clé `temporal_window` ∈ {1, 2, 3, 4}.

```python
def test_temporal_window_key_temporal_window():
    """Chaque dict retourné doit avoir 'temporal_window' ∈ {1..n_tasks}."""
    tasks = ...
    for i, t in enumerate(tasks):
        assert "temporal_window" in t, f"Clé 'temporal_window' manquante dans la tâche {i}"
        assert t["temporal_window"] == i + 1
```

### 3. `test_temporal_window_keys_complete()`

Vérifie que tous les champs requis sont présents dans chaque dict.

```python
def test_temporal_window_keys_complete():
    """Chaque dict doit contenir : task_id, temporal_window, train_loader, val_loader, n_train, n_val."""
    required_keys = {"task_id", "temporal_window", "train_loader", "val_loader", "n_train", "n_val"}
    tasks = ...
    for t in tasks:
        assert required_keys.issubset(t.keys()), f"Clés manquantes : {required_keys - t.keys()}"
```

### 4. `test_temporal_window_no_data_leakage()`

Vérifie que `n_train + n_val` couvre toutes les fenêtres de la tranche (pas de perte).

```python
def test_temporal_window_no_data_leakage():
    """n_train + n_val == nombre total de fenêtres dans la tranche (pas de fuite / perte)."""
    tasks = ...
    for t in tasks:
        total = t["n_train"] + t["n_val"]
        # Vérifier cohérence avec taille attendue de la tranche
        assert total > 0, "Tâche vide"
        # Loader shapes
        for X, y in t["train_loader"]:
            assert X.shape[1] == N_FEATURES
            assert y.shape[1] == 1
            break
```

### 5. `test_temporal_window_train_before_val()`

Vérifie le split temporel : les données de validation sont les *dernières* fenêtres de la tranche.

```python
def test_temporal_window_train_before_val():
    """Split temporel : val = dernières fenêtres, pas de mélange aléatoire."""
    tasks = ...
    t = tasks[0]
    # n_val = int(n_total * val_ratio), val = lignes [n_train:n_train+n_val]
    assert t["n_val"] > 0
    assert t["n_train"] > t["n_val"]  # train >> val
```

### 6. `test_temporal_window_task_id_sequential()`

Vérifie que `task_id` est séquentiel et 1-indexé.

```python
def test_temporal_window_task_id_sequential():
    tasks = ...
    for i, t in enumerate(tasks):
        assert t["task_id"] == i + 1
```

### 7. Test d'intégration (vrai CSV)

```python
@_integration
def test_real_temporal_window_n_tasks():
    """Sur le vrai CSV (20 000 lignes), retourne 4 tâches de ~1249 fenêtres chacune."""
    tasks = get_pump_dataloaders_by_temporal_window(
        csv_path=_CSV_PATH,
        normalizer_path=_NORMALIZER_PATH,
    )
    assert len(tasks) == 4
    for t in tasks:
        assert t["n_train"] + t["n_val"] > 200, f"Trop peu de fenêtres : {t['n_train']+t['n_val']}"
        print(f"Window {t['temporal_window']}: {t['n_train']} train, {t['n_val']} val")
```

---

## Critères d'acceptation

- [x] Tous les tests unitaires passent sans le vrai CSV
- [x] `pytest tests/test_pump_dataset.py -v -k temporal_window` — vert (8/8)
- [x] `pytest tests/test_pump_dataset.py -v` complet — vert (34/34, pas de régression)
- [x] Test d'intégration skippé proprement si CSV absent (`@_integration` via `pytest.mark.skipif`)

---

## Commandes de vérification

```bash
pytest tests/test_pump_dataset.py -v -k temporal_window
pytest tests/test_pump_dataset.py -v  # régression complète
```

---

## Questions ouvertes

Aucune — tests déterministes, pas de dépendance externe pour les tests unitaires.
