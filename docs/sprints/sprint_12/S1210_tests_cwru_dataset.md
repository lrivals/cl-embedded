# S12-10 — Tests unitaires `cwru_dataset.py`

| Champ | Valeur |
|-------|--------|
| **ID** | S12-10 |
| **Sprint** | Sprint 12 |
| **Priorité** | 🟡 Important |
| **Durée estimée** | 2h |
| **Dépendances** | S12-01 (`cwru_dataset.py` implémenté) |
| **Fichier cible** | `tests/test_cwru_dataset.py` |

---

## Objectif

Valider l'implémentation de `src/data/cwru_dataset.py` via une suite d'au moins 9 tests unitaires sur données synthétiques (pas de dépendance aux données réelles). Les fixtures utilisent `tmp_path` pour créer des CSV temporaires, garantissant que les tests s'exécutent en CI sans accès aux fichiers `data/raw/`.

---

## Liste des tests (≥ 9)

| Test | Classe testée | Assertion |
|------|--------------|-----------|
| `test_dataset_shape` | `CWRUDataset` | `X.shape == (N, 9)`, `y.shape == (N,)` |
| `test_dataset_label_binary` | `CWRUDataset` | `set(y) ⊆ {0, 1}` |
| `test_dataset_dtype` | `CWRUDataset` | `X.dtype == float32`, `y.dtype == int8` |
| `test_dataset_no_nan` | `CWRUDataset` | `np.isnan(X).sum() == 0` |
| `test_fault_stream_n_tasks` | `CWRUFaultTypeStream` | `len(list(stream.iter_tasks())) == 3` |
| `test_fault_stream_task_names` | `CWRUFaultTypeStream` | noms = `["ball", "inner_race", "outer_race"]` dans l'ordre |
| `test_fault_stream_no_overlap` | `CWRUFaultTypeStream` | indices non partagés entre tâches |
| `test_severity_stream_n_tasks` | `CWRUSeverityStream` | `len(list(stream.iter_tasks())) == 3` |
| `test_severity_stream_task_names` | `CWRUSeverityStream` | noms = `["007", "014", "021"]` dans l'ordre |
| `test_severity_stream_no_overlap` | `CWRUSeverityStream` | indices non partagés entre tâches |
| `test_normal_class_in_each_task` | `CWRUFaultTypeStream` | classe 0 présente dans chaque tâche |
| `test_reproducibility` | `CWRUDataset` | deux chargements avec même `random_state` → mêmes splits |

---

## Fixtures CSV synthétiques

```python
@pytest.fixture
def synthetic_cwru_csv(tmp_path) -> Path:
    """CSV minimal : 120 lignes, 9 features + colonnes fault_type + severity + label."""
    rng = np.random.default_rng(42)
    n = 120
    data = {
        "max": rng.random(n), "min": rng.random(n), "mean": rng.random(n),
        "sd": rng.random(n), "rms": rng.random(n), "skewness": rng.random(n),
        "kurtosis": rng.random(n), "crest": rng.random(n), "form": rng.random(n),
        "fault_type": ["ball"] * 40 + ["inner_race"] * 40 + ["outer_race"] * 40,
        "severity": ["007"] * 40 + ["014"] * 40 + ["021"] * 40,
        "label": [0] * 10 + [1] * 30 + [0] * 10 + [1] * 30 + [0] * 10 + [1] * 30,
    }
    csv_path = tmp_path / "cwru_synthetic.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)
    return csv_path
```

---

## Commande de validation

```bash
pytest tests/test_cwru_dataset.py -v
# Attendu : XX passed in Xs (≥ 9 tests)
```

---

## Critères d'acceptation

- [ ] `pytest tests/test_cwru_dataset.py -v` → 100% pass (≥ 9 tests)
- [ ] Aucun test ne lit dans `data/raw/` (tout via `tmp_path`)
- [ ] Pas de dépendances de test non listées dans `requirements.txt` (pytest déjà présent)
- [ ] Temps d'exécution < 5 s sur machine de dev standard

## Statut

⬜ Non démarré
