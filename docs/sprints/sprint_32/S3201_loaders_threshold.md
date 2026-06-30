# S3201 — Paramétrer le seuil RUL→faulty dans les loaders

| Champ | Valeur |
|-------|--------|
| **Sprint** | 32 |
| **Priorité** | 🔴 Critique |
| **Statut** | ✅ Implémenté |
| **Durée estimée** | 4h |
| **Dépendances** | Loaders RUL existants (CMAPSS/Battery/Pronostia) |
| **Résultat** | CMAPSS déjà conforme ✅ ; Battery `rul_failure_threshold` threadé (`load_raw_dataset` + `get_battery_dataloaders` + single-task) ; Pronostia nouveau `label_mode: rul_threshold` (réutilise `_compute_rul_labels`), `failure_ratio` reste défaut. **62/62 tests loaders PASS** (non-régression). |
| **Fichiers cibles** | `src/data/cmapss_loader.py`, `src/data/battery_dataset.py`, `src/data/pronostia_dataset.py` |
| **Références** | `cmapss_loader.py:39,108-109` · `battery_dataset.py:71,118-120` · `pronostia_rul_config.yaml` |

---

## Contexte

Le seuil de binarisation `faulty` est aujourd'hui une **constante en dur** dans chaque loader. Le YAML `cmapss_config.yaml:50` contient déjà `faulty_threshold: 30` mais le code l'ignore. Prérequis bloquant de tout le balayage : rendre le seuil **lisible depuis la config** sans casser le comportement par défaut.

---

## Spec

```python
# cmapss_loader.py — la constante devient une valeur par défaut
def _load_raw(data_dir: Path, subset: str,
              faulty_threshold: int = CMAPSS_FAULTY_THRESHOLD) -> pd.DataFrame:
    ...
    df[LABEL_COL] = (df["RUL"] <= faulty_threshold).astype(int)  # opérateur natif inchangé

# get_cl_dataloaders / single-task : propager depuis la config
faulty_threshold = config["data"].get("faulty_threshold", CMAPSS_FAULTY_THRESHOLD)

# battery_dataset.py — idem (opérateur '<' natif conservé)
rul_thr = config["data"].get("rul_failure_threshold", RUL_FAILURE_THRESHOLD)
df["faulty"] = (df[RUL_COL] < rul_thr).astype(np.float32)

# pronostia_dataset.py — nouveau mode de labélisation RUL
if config["data"].get("label_mode") == "rul_threshold":
    faulty = (rul <= config["data"]["faulty_threshold"]).astype(np.float32)
else:  # défaut : mode ratio temporel existant (failure_ratio)
    ...
```

- **Non-régression** : config sans champ seuil OU seuil = constante → labels **bit-à-bit identiques** à l'existant.
- Conserver l'opérateur natif : CMAPSS/Pronostia `≤` (inclusif), Battery `<` (exclusif).
- Pronostia : mode `failure_ratio` reste le défaut ; `rul_threshold` est opt-in via config.

---

## Vérification

```bash
pytest tests/test_cmapss_loader.py tests/test_battery_dataset.py -v   # non-régression seuils par défaut
python -c "from src.data.cmapss_loader import _load_raw; import inspect; print('faulty_threshold' in inspect.signature(_load_raw).parameters)"
```
