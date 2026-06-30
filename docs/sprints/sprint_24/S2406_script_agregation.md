# S2406 — Script d'agrégation historique `compare_all_sprints.py`

| Champ | Valeur |
|-------|--------|
| **Sprint** | 24 |
| **Priorité** | 🟡 Important |
| **Statut** | ✅ Terminé — scripts/compare_all_sprints.py créé, 95 expériences chargées, JSON + CSV OK |
| **Durée estimée** | 2h |
| **Dépendances** | exp_S24_01 à exp_S24_12 ✅ (résultats JSON disponibles) |
| **Fichiers cibles** | `scripts/compare_all_sprints.py`, `experiments/comparison_sprint24.json`, `experiments/comparison_sprint24.csv` |
| **Référence** | `experiments/comparison_sprint21.json` (agrégation Sprint 21), `experiments/comparison_sprint19_20.json` |

---

## Contexte

Les aggrégations précédentes (`comparison_sprint21.json`, `comparison_sprint19_20.json`) ont été produites manuellement ou par scripts ad-hoc. Sprint 24 produit un script réutilisable qui parcourt tous les dossiers `experiments/exp_*/` pour agréger toutes les métriques connues dans un tableau unifié.

Ce tableau est l'entrée principale du notebook comparatif `24_comprehensive_comparison.ipynb`.

---

## Structure du script

### Interface CLI

```bash
# Agréger tout
python scripts/compare_all_sprints.py \
  --exp_dir experiments/ \
  --output_json experiments/comparison_sprint24.json \
  --output_csv experiments/comparison_sprint24.csv

# Filtrer par sprint
python scripts/compare_all_sprints.py \
  --exp_dir experiments/ \
  --sprint_filter S24 \
  --output_json experiments/comparison_sprint24_only.json
```

### Logique principale

```python
def load_experiment(exp_dir: Path) -> dict | None:
    """
    Charge results.json d'un dossier expérience.
    Retourne None si results.json absent ou malformé.
    """
    results_path = exp_dir / "results.json"
    if not results_path.exists():
        return None
    with open(results_path) as f:
        data = json.load(f)
    # Normalisation des champs (compatibilité formats Sprint 1–24)
    return normalize_fields(data, exp_dir.name)

def normalize_fields(data: dict, exp_id: str) -> dict:
    """
    Harmonise les noms de champs entre formats Sprint 1–24.
    Anciens : {"aa": ..., "forgetting": ...}
    Nouveaux : {"acc_final": ..., "avg_forgetting": ...}
    """
    field_map = {
        "aa": "acc_final",
        "forgetting": "avg_forgetting",
        "backward_transfer": "bwt",
        "ram_bytes": "ram_peak_bytes",
    }
    normalized = {"exp_id": exp_id}
    for old, new in field_map.items():
        if old in data and new not in data:
            data[new] = data.pop(old)
    normalized.update(data)
    return normalized
```

### Colonnes du CSV de sortie

```
exp_id, sprint, model, dataset, scenario, uint8_activations,
acc_final, avg_forgetting, bwt, auroc,
ram_peak_bytes, ram_peak_kb, gap2_compliant,
inference_latency_ms, n_params,
compression_ratio, delta_acc_vs_fp32,
reference_exp, notes
```

---

## Sorties attendues

### `experiments/comparison_sprint24.json`

```json
{
  "generated_at": "...",
  "sprint": 24,
  "n_experiments": "...",
  "experiments": [
    {
      "exp_id": "exp_001_ewc_monitoring_by_equipment",
      "sprint": 1,
      "model": "ewc",
      "dataset": "monitoring",
      "scenario": "by_equipment",
      "acc_final": 0.9824,
      "avg_forgetting": 0.0010,
      "bwt": 0.00001,
      "ram_peak_bytes": 9800,
      "gap2_compliant": true,
      "n_params": 641
    },
    ...
    {
      "exp_id": "exp_S24_01",
      "sprint": 24,
      "model": "ewc",
      "dataset": "monitoring",
      "uint8_activations": true,
      "acc_final": "...",
      "compression_ratio": "..."
    }
  ]
}
```

### Vérification

```bash
python scripts/compare_all_sprints.py --exp_dir experiments/ \
  --output_json experiments/comparison_sprint24.json \
  --output_csv experiments/comparison_sprint24.csv

# Vérifier que le CSV est bien formé
python -c "
import pandas as pd
df = pd.read_csv('experiments/comparison_sprint24.csv')
print(f'Expériences chargées : {len(df)}')
print(f'Modèles : {df.model.unique()}')
print(f'Datasets : {df.dataset.unique()}')
assert df.acc_final.notna().sum() > 0, 'acc_final vide!'
print('compare_all_sprints OK')
"
```

---

## Compatibilité backwards

Le script doit gérer les formats de résultats des Sprints 1–21 sans erreur, même si certains champs obligatoires récents (ex. `gap2_compliant`, `compression_ratio`) sont absents des anciens `results.json`. Dans ce cas : remplir avec `null` dans le JSON/CSV et ne pas planter.
