# S10-02 — Validation configs YAML Pronostia

| Champ | Valeur |
|-------|--------|
| **ID** | S10-02 |
| **Sprint** | Sprint 10 — Phase 1 Extension |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 1h |
| **Dépendances** | S10-01 (`PronostiaDataset` importable) |
| **Fichiers cibles** | `configs/pronostia_config.yaml`, `configs/pronostia_single_task_config.yaml` |

---

## Objectif

Les deux configs YAML Pronostia ont déjà été créées (présentes dans git status). Cette tâche consiste à :
1. Valider leur cohérence avec `PronostiaDataset` (S10-01)
2. Vérifier que les scripts d'entraînement les chargent sans erreur
3. Créer les 12 configs individuelles `exp_044_*_config.yaml` à `exp_055_*_config.yaml`

---

## Configs existantes

### `configs/pronostia_config.yaml` — scénario CL by_condition

Config de référence pour **exp_050** (EWC). À dupliquer pour les 5 autres modèles (exp_051–055).

Paramètres clés à vérifier :
- `data.n_features: 13` ✓
- `data.window_size: 2560` ✓
- `data.failure_ratio: 0.10` ✓
- `data.condition_map` : 3 conditions, 2 roulements chacune ✓
- `memory.target_ram_bytes: 65536` ✓ (contrainte STM32N6)

### `configs/pronostia_single_task_config.yaml` — scénario no_split

Config de référence pour **exp_044** (EWC). À dupliquer pour les 5 autres modèles (exp_045–049).

---

## Sous-tâches

### 1. Créer les 12 configs individuelles

```bash
# Scénario no_split (exp_044–049)
for model in ewc hdc tinyol kmeans mahalanobis dbscan; do
    cp configs/pronostia_single_task_config.yaml configs/${model}_pronostia_single_task_config.yaml
    # Mettre à jour exp_id dans chaque copie
done

# Scénario by_condition (exp_050–055)
for model in ewc hdc tinyol kmeans mahalanobis dbscan; do
    cp configs/pronostia_config.yaml configs/${model}_pronostia_by_condition_config.yaml
    # Mettre à jour exp_id dans chaque copie
done
```

Convention de nommage des `exp_id` :
| Exp | exp_id |
|-----|--------|
| exp_044 | `exp_044_ewc_pronostia_no_split` |
| exp_045 | `exp_045_hdc_pronostia_no_split` |
| exp_046 | `exp_046_tinyol_pronostia_no_split` |
| exp_047 | `exp_047_kmeans_pronostia_no_split` |
| exp_048 | `exp_048_mahalanobis_pronostia_no_split` |
| exp_049 | `exp_049_dbscan_pronostia_no_split` |
| exp_050 | `exp_050_ewc_pronostia_by_condition` |
| exp_051 | `exp_051_hdc_pronostia_by_condition` |
| exp_052 | `exp_052_tinyol_pronostia_by_condition` |
| exp_053 | `exp_053_kmeans_pronostia_by_condition` |
| exp_054 | `exp_054_mahalanobis_pronostia_by_condition` |
| exp_055 | `exp_055_dbscan_pronostia_by_condition` |

### 2. Validation par chargement Python

```python
import yaml
from src.data.pronostia_dataset import PronostiaDataset

with open("configs/pronostia_single_task_config.yaml") as f:
    cfg = yaml.safe_load(f)

dataset = PronostiaDataset(
    npy_dir=cfg["data"]["npy_dir"],
    bearing_ids=["Bearing1_1", "Bearing1_2", "Bearing2_1", "Bearing2_2",
                  "Bearing3_1", "Bearing3_2"],
    failure_ratio=cfg["data"]["failure_ratio"],
)
assert dataset[0][0].shape == (cfg["data"]["n_features"],)
print(f"✅ pronostia_single_task_config.yaml validé — {len(dataset)} fenêtres")
```

### 3. Vérifier les output_dir

S'assurer que chaque config `exp_05X` pointe vers le bon répertoire :
```yaml
evaluation:
  output_dir: "experiments/exp_050_ewc_pronostia_by_condition/results/"
```

---

## Critères d'acceptation

- [ ] `configs/pronostia_config.yaml` chargé sans erreur avec `yaml.safe_load()`
- [ ] `configs/pronostia_single_task_config.yaml` chargé sans erreur
- [ ] 12 configs individuelles créées (exp_044 à exp_055)
- [ ] Chaque config contient le bon `exp_id` et le bon `output_dir`
- [ ] La validation Python (`dataset[0][0].shape == (13,)`) passe pour les deux configs

---

## Questions ouvertes

- `TODO(arnaud)` : Les hyperparamètres EWC (`lambda: 1000`, `gamma: 0.9`) issus du dataset Monitoring sont-ils transférables à Pronostia, ou faut-il relancer un grid search sur la condition 1 pour calibrer ?

---

**Complété le** : _(à renseigner)_
