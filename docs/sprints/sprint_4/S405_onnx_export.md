# S4-05 — Export ONNX des 3 modèles (vérification portabilité)

| Champ | Valeur |
|-------|--------|
| **ID** | S4-05 |
| **Sprint** | Sprint 4 — Semaine 4 (6–13 mai 2026) |
| **Priorité** | 🟡 Important |
| **Durée estimée** | 3h |
| **Dépendances** | exp_001 (EWC checkpoint), exp_002 (HDC checkpoint), exp_003 (TinyOL checkpoint) |
| **Fichiers cibles** | `scripts/export_onnx.py` · `experiments/exp_004_tinyol_uint8/` (pour UINT8) |
| **Statut** | ✅ Terminé |

---

## Objectif

Exporter les 3 modèles CL entraînés (EWC MLP, HDC prototypes, TinyOL encoder + OtO Head) au format ONNX, puis valider que l'inférence ONNX Runtime produit les mêmes résultats que PyTorch (tolérance ≤ 1e-5).

L'export ONNX est le **pont entre la Phase 1 Python et la Phase 2 MCU** : il permet de vérifier la portabilité des opérations et d'identifier les opérateurs non supportés par CMSIS-NN ou STM32Cube.AI avant le portage C.

> **Note** : `scripts/export_onnx.py` a été créé lors de sprints ultérieurs (Sprint Phase2 S1002). Cette tâche valide sa compatibilité avec les checkpoints de Sprint 4 et documente les résultats pour la Phase 1.

**Critère de succès** : 3 fichiers `.onnx` produits dans `experiments/exp_160/` (ou dossier dédié), validation onnxruntime sans erreur, rapport d'opérateurs exporté.

---

## Sous-tâches

### 1. Export EWC (MLP 4→32→16→1)

```bash
python scripts/export_onnx.py \
    --model ewc \
    --config configs/ewc_config.yaml \
    --checkpoint experiments/exp_001_ewc_dataset2/checkpoints/ewc_task3_final.pt \
    --output experiments/onnx_sprint4/ewc_mlp.onnx
```

**Opérateurs ONNX attendus** : `Gemm` (Linear), `Relu`, `Sigmoid`, `Dropout` (training=False).  
**Opérateurs problématiques** : aucun — architecture entièrement CMSIS-NN compatible.

### 2. Export TinyOL (Encoder 25→8→2 + OtO Head 9→1)

```bash
python scripts/export_onnx.py \
    --model tinyol \
    --config configs/tinyol_config.yaml \
    --checkpoint experiments/exp_003_tinyol_dataset1/checkpoints/tinyol_final.pt \
    --output experiments/onnx_sprint4/tinyol_encoder.onnx
```

**Opérateurs ONNX attendus** : `Gemm`, `Relu`, `Sigmoid`.  
**Note** : exporter encoder + OtO head séparément (backbone figé pour l'inférence MCU).

### 3. Validation HDC

HDC n'est pas un réseau de neurones au sens strict (prototypes + produit scalaire). L'export ONNX n'est pas applicable directement.  
**Action** : documenter les opérations HDC comme pseudo-ONNX (`MatMul` + `ArgMax`) et noter l'incompatibilité dans le rapport.

### 4. Validation onnxruntime

```python
import onnxruntime as ort
import numpy as np

# Charger le modèle ONNX
sess = ort.InferenceSession("experiments/onnx_sprint4/ewc_mlp.onnx")

# Input de test
x_np = np.random.randn(1, 4).astype(np.float32)

# Inférence ONNX
ort_out = sess.run(None, {"input": x_np})[0]

# Inférence PyTorch
import torch
from src.models.ewc import EWCMlpClassifier
model = EWCMlpClassifier()
model.load_state_dict(torch.load("..."))
model.eval()
with torch.no_grad():
    pt_out = model(torch.from_numpy(x_np)).numpy()

# Validation
assert np.allclose(ort_out, pt_out, atol=1e-5), f"Divergence ONNX vs PyTorch : {np.max(np.abs(ort_out - pt_out))}"
```

### 5. Rapport d'opérateurs

Pour chaque modèle exporté, noter :

```json
{
  "model": "ewc_mlp",
  "onnx_opset": 17,
  "operators": ["Gemm", "Relu", "Sigmoid"],
  "cmsis_nn_compatible": true,
  "stm32cubeai_compatible": true,
  "max_deviation_ort_vs_pytorch": 1.2e-7
}
```

---

## Structure de sortie attendue

```
experiments/onnx_sprint4/
├── ewc_mlp.onnx
├── ewc_mlp_int8.onnx          ← quantification PTQ (onnxruntime quantize_dynamic)
├── tinyol_encoder.onnx
├── tinyol_oto_head.onnx
└── onnx_validation_report.json
```

---

## Critères d'acceptation

- [ ] `ewc_mlp.onnx` et `tinyol_encoder.onnx` produits sans erreur
- [ ] Validation onnxruntime : `max_deviation ≤ 1e-5` pour EWC et TinyOL
- [ ] `onnx_validation_report.json` avec liste des opérateurs et flag `cmsis_nn_compatible`
- [ ] Note documentée sur l'incompatibilité HDC avec ONNX
- [ ] `ewc_mlp_int8.onnx` produit via `quantize_dynamic` (PTQ post-training)

---

## Questions ouvertes

- `TODO(dorra)` : STM32Cube.AI supporte-t-il l'opset ONNX 17 ? Quelle version downgrader si nécessaire ?
- `TODO(arnaud)` : Faut-il exporter uniquement la tête OtO (10 params) ou le backbone complet (1 496 params) pour le MCU ?
- `FIXME(gap3)` : Le `ewc_mlp_int8.onnx` (PTQ) peut être utilisé comme preuve d'export INT8 dans le manuscrit — valider avec Arnaud si c'est suffisant pour Gap 3 ou si une vraie QAT est nécessaire.
