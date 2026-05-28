# S2003 — Export poids TinyOL → `model_weights.h` + validation forward pass

| Champ | Valeur |
|-------|--------|
| **Sprint** | 20 |
| **Priorité** | 🔴 Critique |
| **Statut** | ✅ Terminé |
| **Durée estimée** | 4h |
| **Dépendances** | S1903 (tinyol.c skeleton ✅), checkpoint Python TinyOL Sprint 14/15 |
| **Fichiers cibles** | `firmware/stm32f4_blink/inc/model_weights.h`, `firmware/stm32f4_blink/src/tinyol.c` |
| **Référence** | `docs/models/tinyol_spec.md`, `src/models/tinyol/tinyol_autoencoder.py` |

---

## Contexte

`tinyol.c` existe en skeleton (S1903) — forward pass structuré mais poids non chargés.
Il faut exporter les poids du checkpoint Python (Sprint 14/15) vers un header C statique en Flash, puis valider la cohérence du forward pass C vs Python sur les données de `mock_data.h`.

Architecture encodeur : `Input(5) → Linear(32) → ReLU → Linear(16) → Output(5)` (autoencoder MSE).

---

## Étapes

### 1. Script d'export (à créer ou compléter)

`scripts/export_weights_tinyol.py` :

```python
import torch, struct, numpy as np
from src.models.tinyol.tinyol_autoencoder import TinyOLAutoencoder

model = TinyOLAutoencoder.load('experiments/exp_S14_XX/checkpoint.pt')
model.eval()

def to_c_array(name: str, tensor) -> str:
    flat = tensor.detach().numpy().flatten().astype(np.float32)
    vals = ', '.join(f'{v:.8f}f' for v in flat)
    return f'static const float {name}[{len(flat)}] = {{{vals}}};\n'

with open('firmware/stm32f4_blink/inc/model_weights.h', 'a') as f:
    f.write('/* TinyOL encoder weights — FP32, Flash const */\n')
    f.write(to_c_array('TINYOL_ENC_W1', model.encoder[0].weight))
    f.write(to_c_array('TINYOL_ENC_B1', model.encoder[0].bias))
    f.write(to_c_array('TINYOL_ENC_W2', model.encoder[2].weight))
    f.write(to_c_array('TINYOL_ENC_B2', model.encoder[2].bias))
```

### 2. Compléter `tinyol.c`

Charger les constantes depuis `model_weights.h` dans `tinyol_init()` :

```c
/* MEM: TinyOLEncoder poids en Flash (const) — ~5.6 Ko @ FP32
 * enc_w1[32][5], enc_b1[32], enc_w2[16][32], enc_b2[16] */
void tinyol_init(TinyOLEncoder *enc) {
    memcpy(enc->w1, TINYOL_ENC_W1, sizeof(enc->w1));
    memcpy(enc->b1, TINYOL_ENC_B1, sizeof(enc->b1));
    memcpy(enc->w2, TINYOL_ENC_W2, sizeof(enc->w2));
    memcpy(enc->b2, TINYOL_ENC_B2, sizeof(enc->b2));
}
```

### 3. Validation delta Python vs C

Sur les samples de `mock_data.h` (5D, 10 samples) :
- Forward Python : `encoder(x).detach().numpy()`
- Forward C : sortie de `tinyol_forward()` via test Unity
- Tolérance : max|Δ| ≤ 1e-5 (FP32 sans quantification)

---

## Budget RAM

| Composant | Mémoire |
|-----------|---------|
| Poids encodeur (const Flash) | ~5.6 Ko Flash |
| `TinyOLEncoder` struct (.bss) | ~40 B (pointeurs + état OtO) |
| Stack `tinyol_forward()` | ~512 B (h1[32] + h2[16]) |
| **Total SRAM** | **~552 B** |

---

## Vérification

- [ ] `scripts/export_weights_tinyol.py` produit `model_weights.h` sans erreur
- [ ] `make -C firmware/stm32f4_blink/ all` compile sans warnings
- [ ] Test Unity `test_tinyol_forward_delta` : max|Δ| ≤ 1e-5 vs référence Python
- [ ] `make test` : N/N PASS incluant les nouveaux tests TinyOL

---

## Questions ouvertes

- `TODO(dorra)` : NeuralART Turbo (NPU STM32N6) accepte ONNX opset 17 ou format propriétaire `.nef` ? Impact sur le forward pass C (court-circuité par NPU ?)
- `TODO(arnaud)` : Inclure le décodeur dans `tinyol.c` (reconstruction + MSE threshold) ou forward encodeur seul suffit pour Gap 2 ?
