# S6-03 — Créer `configs/pump_by_temporal_window_config.yaml`

| Champ | Valeur |
|-------|--------|
| **ID** | S6-03 |
| **Sprint** | Sprint 6 — Phase 1 Extension (≥ 15 avril 2026) |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 1h |
| **Dépendances** | S6-02 (`get_pump_dataloaders_by_temporal_window()` implémenté) |
| **Fichiers cibles** | `configs/pump_by_temporal_window_config.yaml` |
| **Complété le** | — |
| **Statut** | ⬜ À faire |

---

## Objectif

Créer le fichier de configuration YAML pour le scénario CL **pump_by_temporal_window** :
4 tâches, chacune correspondant à un quartile de 5 000 entrées triées par `Operational_Hours`.

Ce fichier est passé en `--data_config` aux scripts d'entraînement pour les expériences
exp_024–029 (Sprint 6, tâche S6-10). Il doit être compatible avec :
- `scripts/train_tinyol.py`
- `scripts/train_ewc.py`
- `scripts/train_hdc.py`
- `scripts/train_unsupervised.py`

**Critère de succès** : les scripts lisent le fichier sans erreur et appellent
`get_pump_dataloaders_by_temporal_window()` avec les bons paramètres.

---

## Contenu du fichier à créer

```yaml
# pump_by_temporal_window_config.yaml — Scénario CL par fenêtres temporelles (4 tâches)
# T1: lignes 0–4999   (Operational_Hours les + basses — rodage / début d'exploitation)
# T2: lignes 5000–9999
# T3: lignes 10000–14999
# T4: lignes 15000–19999 (Operational_Hours les + hautes — vieillissement / pré-panne)

data:
  csv_path: "data/raw/pump_maintenance/Large_Industrial_Pump_Maintenance_Dataset/Large_Industrial_Pump_Maintenance_Dataset.csv"
  normalizer_path: "configs/pump_normalizer.yaml"
  task_split: by_temporal_window
  n_tasks: 4
  entries_per_task: 5000
  temporal_col: Operational_Hours
  label_col: Maintenance_Flag
  feature_cols: [temperature, vibration, pressure, rpm]
  window_size: 32
  step_size: 16
  val_ratio: 0.2
  dataset: pump_maintenance
  label_column: maintenance_required
  feature_columns: [temperature, vibration, pressure, rpm]
```

---

## Critères d'acceptation

- [ ] `yaml.safe_load(open("configs/pump_by_temporal_window_config.yaml"))` — aucune erreur
- [ ] Clé `data.task_split == "by_temporal_window"`
- [ ] `data.n_tasks == 4`, `data.entries_per_task == 5000`
- [ ] Chemin CSV cohérent avec `configs/pump_by_id_config.yaml`
- [ ] Lisible par `train_ewc.py --data_config configs/pump_by_temporal_window_config.yaml`

---

## Commandes de vérification

```bash
python -c "
import yaml
cfg = yaml.safe_load(open('configs/pump_by_temporal_window_config.yaml'))
assert cfg['data']['task_split'] == 'by_temporal_window'
assert cfg['data']['n_tasks'] == 4
assert cfg['data']['entries_per_task'] == 5000
print('Config OK')
"
```

---

## Questions ouvertes

- `TODO(arnaud)` : Nommer les tâches `Q1`–`Q4` ou `T1`–`T4` dans les notebooks ? Les quartiles
  impliquent une équirépartition stricte ; `T1`–`T4` est plus neutre si les phases ne sont pas
  équilibrées en termes de taux de panne.
