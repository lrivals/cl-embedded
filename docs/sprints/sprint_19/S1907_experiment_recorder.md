# S1907 — Experiment recorder Python : capture résultats board → experiments/ unifié Phase 1

| Champ | Valeur |
|-------|--------|
| **Sprint** | 19 |
| **Priorité** | 🔴 Critique |
| **Statut** | ✅ Scripté — dry-run à valider |
| **Durée estimée** | 3h |
| **Dépendances** | S1906 (protocole v3) |
| **Fichiers cibles** | `scripts/board_experiment_recorder.py`, `tests/test_board_recorder.py` |

---

## Contexte

Le projet Phase 1 génère ses résultats expérimentaux au format JSON unifié dans `experiments/exp_XXX/results.json` via `scripts/evaluate_all.py`. Pour que les résultats board (Phase 2) soient comparables, `board_experiment_recorder.py` doit produire **exactement le même format** avec les 6 métriques obligatoires.

---

## Objectif

Valider que `board_experiment_recorder.py --dry-run` produit un `results.json` complet avec les 6 métriques, et que le format correspond à celui attendu par les notebooks d'analyse Phase 1.

---

## État actuel — Script existant ✅

**`scripts/board_experiment_recorder.py`**

### Architecture

```
board_experiment_recorder.py
  ├── _load_stream_module()     — import dynamique de sensor_stream.py
  ├── _run_experiment()         — orchestre streaming + collecte métriques
  │     ├── sensor_stream.py   — envoi UART + parse réponse v3 (21 B)
  │     └── retourne (résultats_bruts, durée)
  ├── _compute_cl_metrics()    — calcule AF, BWT depuis résultats bruts
  ├── _read_ram_from_map()     — parse linker map → ram_peak_bytes (S1913)
  └── main()                   — argparse + save JSON + config_snapshot.yaml
```

### Paramètres CLI

```bash
python scripts/board_experiment_recorder.py \
    --model {mahalanobis,ewc,tinyol} \
    --dataset {cwru,monitoring,pronostia} \
    --port /dev/ttyACM0 \          # optionnel si --dry-run
    --baud 115200 \
    --n-samples 500 \
    --n-tasks 3 \
    --request-update \             # active l'update en ligne sur board
    --output experiments/exp_S19_01 \
    --dry-run                      # sans board, génère JSON avec valeurs fictives
```

### Format JSON produit

```json
{
  "exp_id": "S19_01",
  "model": "mahalanobis",
  "dataset": "cwru",
  "platform": "nucleo_f439zi",
  "date": "2026-06-02",
  "acc_final": 0.94,
  "avg_forgetting": 0.02,
  "backward_transfer": -0.01,
  "ram_peak_bytes": 210,
  "inference_latency_ms": 0.003,
  "n_params": 30,
  "n_tasks": 3,
  "n_samples_total": 500,
  "config_snapshot": "configs/board_mahalanobis.yaml"
}
```

### Constante `_N_PARAMS`

```python
_N_PARAMS = {
    "mahalanobis": 30,    # mean(5) + precision(25)
    "ewc":         1538,  # poids + Fisher + star_w ÷ 3 comptés une fois
    "tinyol":      881,   # encoder-only (decoder exclu pour comptage Sprint 19)
}
```

---

## Ce qu'il faut valider

### 1. Dry-run sans board

```bash
python scripts/board_experiment_recorder.py \
    --model mahalanobis --dataset cwru \
    --dry-run --output experiments/exp_S19_01
```

Vérifier :
- Dossier `experiments/exp_S19_01/` créé
- `results.json` présent avec les 6 métriques (`acc_final`, `avg_forgetting`, `backward_transfer`, `ram_peak_bytes`, `inference_latency_ms`, `n_params`)
- `config_snapshot.yaml` copié depuis `configs/board_mahalanobis.yaml`

### 2. Champs obligatoires

Ajouter assertion Python (dans `tests/test_board_recorder.py`) :

```python
REQUIRED_KEYS = {
    "acc_final", "avg_forgetting", "backward_transfer",
    "ram_peak_bytes", "inference_latency_ms", "n_params"
}
```

### 3. Compatibilité format Phase 1

Vérifier que `pandas.read_json("experiments/exp_S19_01/results.json")` retourne un DataFrame avec les mêmes colonnes que les expériences Phase 1 (exp_100 à exp_142).

---

## Fichiers cibles

| Fichier | Action |
|---------|--------|
| `scripts/board_experiment_recorder.py` | Valider dry-run, corriger si JSON incomplet |
| `scripts/sensor_stream.py` | Doit parser réponse 21 B (protocole v3, S1906) |
| `tests/test_board_recorder.py` | Implémenter tests (S1910) |
| `configs/board_mahalanobis.yaml` | Source config_snapshot |

---

## Vérification

- [ ] `python scripts/board_experiment_recorder.py --model mahalanobis --dataset cwru --dry-run --output experiments/exp_S19_01` → exit 0
- [ ] `experiments/exp_S19_01/results.json` contient les 6 métriques obligatoires
- [ ] `pytest tests/test_board_recorder.py -v` → PASS (S1910)
- [ ] Formatage JSON valide : `python -m json.tool experiments/exp_S19_01/results.json`
