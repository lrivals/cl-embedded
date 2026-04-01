# Skill : Portabilité Embarquée (PC → STM32N6)

> **Usage** : Vérifier la portabilité d'un modèle, préparer l'export ONNX, anticiper les problèmes de portage.  
> **Déclencheur** : "vérifie la portabilité de [modèle]" / "prépare l'export ONNX" / "quels problèmes de portage ?"

---

## Checklist de portabilité MCU

À exécuter avant de déclarer un modèle "prêt pour le portage" :

### 1. Architecture
- [ ] Pas de couches non supportées par TFLite Micro / CMSIS-NN (liste ci-dessous)
- [ ] Tailles de couches constantes (pas de dimensions dynamiques)
- [ ] Activations ReLU uniquement (ou RELU6 si nécessaire)
- [ ] Pas de BatchNorm avec stats en ligne
- [ ] Pas de Dropout dans le forward d'inférence

### 2. Mémoire
- [ ] `ram_peak_bytes` mesuré (via `memory_profiler.py`)
- [ ] Résultat < 65 536 octets (64 Ko)
- [ ] Décomposition documentée (poids / activations / overhead CL)

### 3. Arithmetic
- [ ] Pas d'opérations en double précision (FP64)
- [ ] Compatibilité FP32 SP (Cortex-M55 FPU)
- [ ] Pas de fonctions math non standards (erf, complex, etc.)

### 4. Export
- [ ] Export ONNX réussi (`torch.onnx.export`)
- [ ] Vérification ONNX checker (`onnx.checker.check_model`)
- [ ] Test inférence ONNX Runtime (résultat identique à PyTorch ± 1e-5)

---

## Couches supportées par CMSIS-NN / TFLite Micro

| Couche | Support CMSIS-NN | Support TFLite Micro | Notes |
|--------|:---:|:---:|-------|
| `Linear` / `FullyConnected` | ✅ | ✅ | Kernel INT8 disponible |
| `ReLU` | ✅ | ✅ | |
| `ReLU6` | ✅ | ✅ | |
| `Sigmoid` | ✅ | ✅ | Approximation LUT |
| `Tanh` | ⚠️ | ✅ | LUT, moins efficace |
| `Softmax` | ✅ | ✅ | |
| `Conv1D` | ✅ | ✅ | |
| `Conv2D` | ✅ | ✅ | |
| `BatchNorm` | ⚠️ folded only | ⚠️ | Doit être "folded" dans les poids |
| `LayerNorm` | ❌ | ⚠️ | À éviter |
| `GELU`, `SiLU` | ❌ | ❌ | Remplacer par ReLU |
| `MultiHeadAttention` | ❌ | ⚠️ | Éviter sur MCU contraint |
| `LSTM`, `GRU` | ⚠️ | ✅ | Coûteux en RAM |
| `Dropout` | N/A (inférence) | N/A | Désactiver à l'export |

---

## Script d'export ONNX standard

```python
# scripts/export_onnx.py

import torch
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path


def export_model_to_onnx(
    model: torch.nn.Module,
    input_shape: tuple,
    output_path: str,
    model_name: str,
) -> bool:
    """
    Exporte un modèle PyTorch en ONNX et valide l'export.
    
    Returns
    -------
    bool : True si l'export et la validation réussissent.
    """
    model.eval()
    dummy_input = torch.zeros(input_shape)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Export ONNX
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=11,          # compatible TFLite Micro
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,          # tailles fixes obligatoires pour MCU
    )
    
    # 2. Vérification ONNX
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    
    # 3. Validation numérique
    ort_session = ort.InferenceSession(str(output_path))
    test_input = np.random.randn(*input_shape).astype(np.float32)
    
    with torch.no_grad():
        torch_out = model(torch.tensor(test_input)).numpy()
    
    ort_out = ort_session.run(None, {"input": test_input})[0]
    max_diff = np.max(np.abs(torch_out - ort_out))
    
    success = max_diff < 1e-5
    print(f"Export {model_name}: {'✅' if success else '❌'} (max diff: {max_diff:.2e})")
    return success
```

---

## Estimation du budget MCU (calcul manuel)

Pour vérifier rapidement avant de coder :

```
RAM modèle (FP32) = Σ (D_in × D_out + D_out) × 4 octets  pour chaque Linear

RAM activations (FP32) = max_D_hidden × 4 octets  (pic lors du forward)

RAM overhead EWC = RAM modèle × 2  (Fisher + snapshot)

RAM buffer UINT8 = BUFFER_SIZE × EMBED_DIM × 1 octet

RAM TOTAL = RAM modèle + RAM activations + RAM overhead CL + RAM buffer
```

**Exemple pour M2 EWC MLP (6→32→16→1)** :
```
Linear(6→32)  = (6×32 + 32) × 4 = 224 × 4 = 896 B
Linear(32→16) = (32×16 + 16) × 4 = 528 × 4 = 2112 B
Linear(16→1)  = (16×1 + 1) × 4 = 17 × 4 = 68 B
Modèle total = 3076 B ≈ 3 Ko

Activations max = 32 × 4 = 128 B

Overhead EWC = 3076 × 2 = 6152 B ≈ 6 Ko

RAM TOTAL ≈ 3076 + 128 + 6152 = 9356 B ≈ 9.1 Ko ✅ << 64 Ko
```

---

## Pipeline de portage Phase 2 (référence)

```
1. PC Python (ce dépôt)
   → Entraînement batch backbone
   → Export checkpoint .pt

2. Export ONNX
   → scripts/export_onnx.py
   → Validation numérique

3. Conversion TFLite INT8 (quantification PTQ)
   → Via tf.lite.TFLiteConverter depuis ONNX
   → Ou directement via STM32Cube.AI

4. STM32Cube.AI
   → Validation du réseau (rapport d'analyse mémoire)
   → Génération code C (tableaux de poids + fonctions d'inférence)

5. Intégration MCU (STM32CubeIDE)
   → Backbone en Flash (code C généré)
   → Tête entraînable + CL en C manuel (src/embedded/)
   → Profiling mémoire HAL + SRAM mesurée

6. Validation finale
   → Test stream CL sur MCU réel
   → Comparaison métriques PC vs MCU
   → Publication chiffres précis (Gap 2)
```

---

## Questions pour Dorra Ben Khalifa

Avant d'entamer la Phase 2, valider ces points :

```
TODO(dorra) — Questions Phase 2 :

1. STM32N6 NPU : quel format d'entrée pour l'inférence ?
   (INT8 quantifié / FP32 non quantifié / les deux ?)

2. STM32Cube.AI version : compatible avec export ONNX opset 11 ?

3. CMSIS-NN : y a-t-il des kernels pour la mise à jour de poids (backprop) ?
   (arm_nn_vec_mat_mult_t_s8 avec gradient ?)

4. RAM applicative réelle : quelle SRAM reste disponible sur N6
   après initialisation HAL + DMA + pile système ?

5. TFLite Micro sur Cortex-M55 avec Helium :
   les instructions vectorielles MVE sont-elles utilisées automatiquement
   ou nécessitent-elles une configuration explicite ?
```
