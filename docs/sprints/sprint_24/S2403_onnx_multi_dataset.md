# S2403 — Export ONNX systématique (5 datasets × 4 modèles)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 24 |
| **Priorité** | 🟡 Important |
| **Statut** | ✅ Terminé |
| **Durée estimée** | S2403a : 1h / S2403b : 1h = 2h total |
| **Dépendances** | `scripts/export_onnx.py` ✅ (Sprint 4, 3 modèles base), poids sauvegardés dans `experiments/exp_S24_04` à `exp_S24_12` |
| **Fichiers cibles** | `scripts/export_onnx.py`, `experiments/onnx_sprint24/` |
| **Référence** | Sprint 4 S405 (`experiments/onnx_sprint4/` : 3 fichiers `.onnx` base) |

---

## Contexte

Sprint 4 a exporté les 3 modèles (EWC, HDC, TinyOL) en ONNX sur leurs configurations initiales (Dataset 2 / Monitoring, Pump). Les 5 datasets et 4 modèles des Sprints 5–23 n'ont jamais été exportés. Ce fichier couvre l'extension de `export_onnx.py` et la génération systématique.

---

## S2403a — Extension `scripts/export_onnx.py`

### Modifications requises

Ajouter le support `--all` et `--dataset` pour boucler sur tous les modèles × datasets :

```python
# scripts/export_onnx.py — ajout de l'interface multi-dataset
import argparse
from pathlib import Path

SUPPORTED_DATASETS = ["monitoring", "pump", "cwru", "pronostia", "cmapss", "paderborn"]
SUPPORTED_MODELS   = ["ewc", "hdc", "tinyol", "mahalanobis"]

# Correspondance modèle → fichier de poids + config
MODEL_CONFIG_MAP = {
    "ewc": {
        "monitoring": ("configs/ewc_config.yaml", "experiments/exp_001_ewc_monitoring_by_equipment/"),
        "cwru":       ("configs/ewc_config.yaml",  "experiments/exp_S24_04/"),
        "pronostia":  ("configs/ewc_config.yaml",  "experiments/exp_S24_08/"),
        "pump":       ("configs/ewc_config.yaml",  "experiments/exp_S24_10/"),
        "cmapss":     ("configs/cmapss_config.yaml", "experiments/exp_S22_01/"),
        "paderborn":  ("configs/paderborn_config.yaml", "experiments/exp_S22_05/"),
    },
    # ... idem HDC, TinyOL, Mahalanobis
}
```

### Interface CLI enrichie

```bash
# Export d'un modèle × dataset spécifique
python scripts/export_onnx.py --model ewc --dataset cwru --output_dir experiments/onnx_sprint24/

# Export systématique de tout
python scripts/export_onnx.py --all --output_dir experiments/onnx_sprint24/
```

### Vérification ONNX

```python
import onnx, onnxruntime as ort, numpy as np

def validate_onnx(onnx_path: str, input_dim: int) -> bool:
    """Vérifie que le modèle ONNX est valide et produit un output correct."""
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    session = ort.InferenceSession(onnx_path)
    x = np.random.randn(1, input_dim).astype(np.float32)
    out = session.run(None, {"input": x})
    assert out[0].shape == (1, 1), f"Output shape incorrect: {out[0].shape}"
    return True
```

---

## S2403b — Génération des 20 fichiers ONNX

### Structure de sortie

```
experiments/onnx_sprint24/
├── ewc_monitoring.onnx
├── ewc_cwru.onnx
├── ewc_pronostia.onnx
├── ewc_pump.onnx
├── ewc_cmapss.onnx
├── ewc_paderborn.onnx
├── hdc_monitoring.onnx
├── hdc_cwru.onnx
├── hdc_pump.onnx
├── hdc_cmapss.onnx
├── hdc_paderborn.onnx
├── tinyol_monitoring.onnx
├── tinyol_cwru.onnx
├── tinyol_pump.onnx
├── mahalanobis_monitoring.onnx
├── mahalanobis_cwru.onnx
├── mahalanobis_pronostia.onnx
├── mahalanobis_pump.onnx
├── mahalanobis_cmapss.onnx
├── mahalanobis_paderborn.onnx
└── onnx_manifest.json    ← inventaire avec input_dim, n_params, taille fichier
```

### Commande

```bash
python scripts/export_onnx.py --all --output_dir experiments/onnx_sprint24/
```

### Contenu `onnx_manifest.json`

```json
{
  "generated_at": "...",
  "sprint": 24,
  "models": [
    {
      "filename": "ewc_monitoring.onnx",
      "model": "ewc",
      "dataset": "monitoring",
      "input_dim": 5,
      "n_params": "...",
      "file_size_kb": "...",
      "onnx_opset": 17,
      "validated": true
    },
    ...
  ]
}
```

### Critères de validation

- 20 fichiers `.onnx` présents dans `experiments/onnx_sprint24/` ✓
- `onnx.checker.check_model()` passe pour chaque fichier ✓
- `onnxruntime.InferenceSession` produit un output de forme `(1, 1)` pour chaque fichier ✓

---

## Note sur Mahalanobis ONNX

Mahalanobis n'est pas un réseau de neurones PyTorch standard — l'export ONNX doit encapsuler la distance Mahalanobis comme un graphe ONNX custom (opérations matricielles). Si l'export échoue, fallback : sauvegarder le modèle en `joblib` et documenter l'exception dans `onnx_manifest.json`.

```python
# Fallback Mahalanobis si export ONNX non supporté
import joblib
joblib.dump(model, "experiments/onnx_sprint24/mahalanobis_monitoring.joblib")
```
