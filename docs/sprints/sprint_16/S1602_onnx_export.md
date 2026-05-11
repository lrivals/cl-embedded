# S1002 — Export ONNX + Quantification PTQ INT8

| Champ | Valeur |
|-------|--------|
| **ID** | S1002 |
| **Sprint** | Sprint 16 — Semaine 2 (28 mai – 3 juin 2026) |
| **Priorité** | Critique |
| **Durée estimée** | 8h |
| **Dépendances** | S1001 terminé (toolchain OK) |
| **Fichiers cibles** | `scripts/export_onnx.py`, `experiments/exp_160/` |

---

## Objectif

Exporter les modèles Python (EWC-MLP backbone + TinyOL autoencoder) en ONNX,
puis quantifier en INT8 via STM32Cube.AI pour générer du code C embarqué.

**Pipeline cible** :

```
PyTorch .pt → torch.onnx.export → .onnx → STM32Cube.AI → code C INT8
```

---

## Contexte MCU

Le NPU du STM32N6 (NeuralART Turbo) fait de l'inférence INT8 uniquement.
La backpropagation reste en FP32 sur Cortex-M55.
→ Seul le **backbone** (forward pass) est quantifié INT8 et exporté en C.
→ La **tête entraînable** (EWC Fisher, OtO head) reste en FP32 en RAM.

---

## Sous-tâches

### 1. Créer `scripts/export_onnx.py`

```python
import torch
from src.models.ewc.ewc_mlp import EWCMlpClassifier
from src.models.tinyol.autoencoder import TinyOLEncoder

def export_ewc_backbone(config_path: str, output_path: str) -> None:
    """Export EWC backbone (feature extractor) to ONNX."""
    # Charger poids entraînés
    # Passer en mode eval + torch.no_grad()
    # torch.onnx.export(backbone, dummy_input, output_path, opset_version=17)
    pass

def export_tinyol_encoder(config_path: str, output_path: str) -> None:
    """Export TinyOL encoder (autoencoder backbone) to ONNX."""
    pass
```

**Vérifications ONNX à faire** :
- Pas d'opérateurs custom (vérifier avec `onnx.checker.check_model`)
- Opset version ≤ 17 (compatible STM32Cube.AI)
- Pas de BatchNorm, LayerNorm (déjà exclu par contraintes MCU)
- Pas de dynamic shapes (batch size = 1 fixe)

### 2. Valider l'export ONNX

```bash
pip install onnx onnxruntime
python scripts/export_onnx.py --model ewc --config configs/ewc_config.yaml \
                               --output experiments/exp_160/ewc_backbone.onnx

# Vérifier la cohérence des sorties
python -c "
import onnxruntime as ort, numpy as np
sess = ort.InferenceSession('experiments/exp_160/ewc_backbone.onnx')
x = np.random.randn(1, 5).astype(np.float32)  # 5 features monitoring
out = sess.run(None, {sess.get_inputs()[0].name: x})
print('ONNX output shape:', out[0].shape)
"
```

### 3. Quantification PTQ via STM32Cube.AI

```bash
# Option A : STM32Cube.AI CLI (si installé)
stm32ai generate -m experiments/exp_160/ewc_backbone.onnx \
                 --type onnx \
                 -o experiments/exp_160/stm32ai_output/

# Option B : TFLite Converter (via ONNX → TF → TFLite)
# Voir docs/context/hardware_constraints.md pour le pipeline complet
```

**Sortie attendue** :
- `experiments/exp_160/ewc_backbone_int8.tflite`
- `experiments/exp_160/stm32ai_output/network.c` + `network.h`

### 4. Vérifier la précision post-quantification

Comparer l'accuracy FP32 (PyTorch) vs INT8 (ONNX Runtime quantized) sur le
jeu de validation Monitoring :

```bash
python scripts/eval_onnx_vs_pytorch.py \
    --onnx experiments/exp_160/ewc_backbone.onnx \
    --config configs/ewc_config.yaml \
    --dataset monitoring
```

**Seuil acceptable** : dégradation AUROC < 2 points (ex. 0.95 → 0.93 OK).

---

## Critères d'acceptation

- [x] `ewc_backbone.onnx` passe `onnx.checker.check_model` sans erreur
- [x] `tinyol_encoder.onnx` passe `onnx.checker.check_model` sans erreur
- [x] Sorties ONNX Runtime identiques à PyTorch (tolérance 1e-5) — max|Δ| = 5.96e-08
- [ ] Dégradation accuracy post-PTQ < 2 points AUROC — à évaluer avec checkpoint entraîné (`eval_onnx_vs_pytorch.py --checkpoint`)
- [ ] Code C généré par STM32Cube.AI compile sans erreur sur NUCLEO-F439ZI — bloqué TODO(dorra)

---

## Questions ouvertes

- `TODO(dorra)` : quelle version de STM32Cube.AI utiliser pour la compatibilité STM32N6 ?
- `TODO(dorra)` : opset ONNX recommandé pour NeuralART Turbo ?

**Complété le** : 2026-05-11

### Notes d'implémentation

- `scripts/export_onnx.py` — export EWC backbone + TinyOL encoder, validation onnxruntime, PTQ INT8 via `quantize_dynamic`
- `scripts/eval_onnx_vs_pytorch.py` — comparaison FP32 vs INT8 par tâche (AUROC + accuracy)
- Exporteur legacy (`dynamo=False`) utilisé pour compatibilité `quantize_dynamic` (le nouvel exporteur torch 2.x génère un graphe incompatible avec shape inference onnxruntime)
- `onnxscript>=0.1` ajouté comme dépendance (requis par torch.onnx avec PyTorch ≥ 2.9)
- Critères ✓ : `onnx.checker` OK, max|Δ| FP32 = 5.96e-08 (seuil 1e-5)
- Critère dégradation PTQ : à évaluer avec checkpoint entraîné (`--checkpoint`)
- STM32Cube.AI : CLI non installé (`TODO(dorra)`) — flag `--stm32ai` disponible dans le script
