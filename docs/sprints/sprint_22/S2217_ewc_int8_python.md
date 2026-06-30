# S2217–S2220 — Gap 3 : INT8 Backprop Python (simulation fake-quant)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 22 |
| **Priorité** | 🔴 Critique |
| **Statut** | ✅ Terminé |
| **Durée estimée** | 4h + 1h + 1h + 2h = 8h |
| **Dépendances** | `src/models/ewc/ewc_mlp.py` ✅, `experiments/exp_S22_01/` ✅ (CMAPSS EWC FP32 terminé) |
| **Fichiers cibles** | `src/models/ewc/ewc_mlp_int8.py`, `experiments/exp_S22_INT8_01/`, `experiments/exp_S22_INT8_02/`, `notebooks/int8_vs_fp32_comparison.ipynb` |
| **Référence** | `src/models/ewc/ewc_mlp.py`, `Ravaglia2021QLRCL`, `Kirkpatrick2017EWC` |

---

## Contexte

**Gap 3** : aucun travail de la littérature ne démontre une quantification INT8 pendant l'entraînement incrémental (backpropagation quantifiée, pas seulement l'inférence). Ce sprint implémente la simulation en Python via **fake-quantization** (weights et activations arrondis à INT8 pendant le forward/backward, mais stockés en FP32).

Le critère de succès Gap 3 est : **AUROC_INT8 ≥ AUROC_FP32 − 0.02** (dégradation tolérable < 2 points).

Le modèle de base à quantifier est `EWCMlpClassifier` de `src/models/ewc/ewc_mlp.py`.  
La validation board (latence INT8 vs FP32 mesurée sur NUCLEO) est remise au Sprint 23.

---

## S2217 — `src/models/ewc/ewc_mlp_int8.py`

### Principe fake-quantization

La fake-quantization simule INT8 en FP32 :
```
x_quant = round(x / scale) * scale   où scale = (max - min) / 255
```

PyTorch fournit `torch.quantization.FakeQuantize` et `torch.quantization.observer` pour automatiser ce calcul pendant l'entraînement.

### Structure du module

```python
"""
ewc_mlp_int8.py — EWC MLP avec simulation fake-quantization INT8.

Implémente Gap 3 : SGD INT8 backprop simulé par fake-quant (weights + activations).
Les gradients circulent en FP32 (straight-through estimator automatique avec PyTorch QAT).

Critère : AUROC_INT8 ≥ AUROC_FP32 - 0.02 (docs/triple_gap.md Gap 3)

Usage :
    from src.models.ewc.ewc_mlp_int8 import EWCMlpInt8Classifier
    model = EWCMlpInt8Classifier(input_dim=5)
    # Entraînement identique à EWCMlpClassifier via ewc_loss()

Référence : Ravaglia2021QLRCL (rejeu latent UINT8), ewc_mlp.py
"""

import torch
import torch.nn as nn
import torch.quantization as quant


class EWCMlpInt8Classifier(nn.Module):
    """
    MLP binaire EWC avec fake-quantization INT8 (Quantization-Aware Training).

    Architecture identique à EWCMlpClassifier :
        Linear(input_dim → 32) + FakeQuant + ReLU
        Linear(32 → 16)        + FakeQuant + ReLU
        Linear(16 → 1)         + Sigmoid

    Quantization :
        Weights : per-channel symmetric INT8 (torch MinMaxObserver)
        Activations : per-tensor affine INT8 (torch HistogramObserver)

    Notes
    -----
    MCU mapping (ewc_head_int8.c) :
        Q7  (int8_t)  — activations
        Q15 (int16_t) — accumulateurs MAC
    """
```

### Implémentation détaillée

```python
def __init__(self, input_dim: int = 5, hidden_dims: list[int] | None = None, dropout: float = 0.2):
    super().__init__()
    if hidden_dims is None:
        hidden_dims = [32, 16]

    self.input_dim = input_dim
    self.hidden_dims = hidden_dims

    # Couches linéaires — identiques à EWCMlpClassifier
    # MEM: Linear(5→32) = (5×32+32)×4 = 704 B @ FP32 / 176 B @ INT8
    self.fc1 = nn.Linear(input_dim, hidden_dims[0])
    # MEM: Linear(32→16) = (32×16+16)×4 = 2112 B @ FP32 / 528 B @ INT8
    self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
    # MEM: Linear(16→1) = (16×1+1)×4 = 68 B @ FP32 / 17 B @ INT8
    self.fc3 = nn.Linear(hidden_dims[1], 1)

    self.drop1 = nn.Dropout(p=dropout)
    self.drop2 = nn.Dropout(p=dropout)

    # Fake-quantizers pour activations (par couche)
    # MEM: observers = négligeable (scalaires FP32)
    act_fq = lambda: quant.FakeQuantize.with_args(
        observer=quant.HistogramObserver,
        quant_min=-128, quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_affine,
    )
    self.fq_input = act_fq()()
    self.fq_h1    = act_fq()()
    self.fq_h2    = act_fq()()

    # Fake-quantizers pour poids (per-channel, symétrique)
    w_fq = quant.FakeQuantize.with_args(
        observer=quant.PerChannelMinMaxObserver,
        quant_min=-128, quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
    )
    self.fq_w1 = w_fq()
    self.fq_w2 = w_fq()
    self.fq_w3 = w_fq()

def forward(self, x: torch.Tensor) -> torch.Tensor:
    x   = self.fq_input(x)
    w1  = self.fq_w1(self.fc1.weight)
    h1  = torch.relu(nn.functional.linear(x, w1, self.fc1.bias))
    h1  = self.fq_h1(self.drop1(h1))

    w2  = self.fq_w2(self.fc2.weight)
    h2  = torch.relu(nn.functional.linear(h1, w2, self.fc2.bias))
    h2  = self.fq_h2(self.drop2(h2))

    w3  = self.fq_w3(self.fc3.weight)
    out = torch.sigmoid(nn.functional.linear(h2, w3, self.fc3.bias))
    return out

def ewc_loss(self, x, y, fisher, theta_star, ewc_lambda):
    """Identique à EWCMlpClassifier.ewc_loss() — perte EWC standard."""
    # (copier-coller depuis ewc_mlp.py — même formule)
    ...

def get_theta_star(self):
    """Snapshot des poids quantifiés (même interface que EWCMlpClassifier)."""
    return {name: param.detach().clone() for name, param in self.named_parameters()}
```

### Vérification

```bash
python -c "
import torch
from src.models.ewc.ewc_mlp_int8 import EWCMlpInt8Classifier
model = EWCMlpInt8Classifier(input_dim=5)
model.train()
x = torch.randn(8, 5)
y = torch.randint(0, 2, (8, 1)).float()
out = model(x)
assert out.shape == (8, 1), 'shape error'
loss = model.ewc_loss(x, y, None, None, 1000.0)
loss.backward()
print(f'EWCMlpInt8Classifier OK — loss={loss.item():.4f}')
"
```

---

## S2218 — exp_S22_INT8_01 : EWC FP32 vs INT8 / CWRU

Comparaison sur le dataset CWRU (référence, résultats déjà disponibles depuis Sprint 18-21).

```bash
# Entraîner le modèle INT8 sur CWRU (réutiliser config existante)
python scripts/train_ewc.py \
    --config configs/cwru_by_fault_config.yaml \
    --model int8 \
    --exp-id exp_S22_INT8_01 \
    --output experiments/exp_S22_INT8_01/
```

**`results.json` — format étendu** :
```json
{
  "exp_id": "exp_S22_INT8_01",
  "dataset": "cwru",
  "fp32": {
    "auroc_final": 0.XX,
    "acc_final": 0.XX,
    "avg_forgetting": 0.XX,
    "ram_peak_bytes": XXXX
  },
  "int8": {
    "auroc_final": 0.XX,
    "acc_final": 0.XX,
    "avg_forgetting": 0.XX,
    "ram_peak_bytes": XXXX
  },
  "delta_auroc": 0.XX,
  "gap3_criterion_met": true
}
```

**Critère Gap 3** : `|fp32.auroc_final - int8.auroc_final| < 0.02`

---

## S2219 — exp_S22_INT8_02 : EWC FP32 vs INT8 / CMAPSS

```bash
python scripts/train_ewc.py \
    --config configs/cmapss_config.yaml \
    --model int8 \
    --exp-id exp_S22_INT8_02 \
    --output experiments/exp_S22_INT8_02/
```

Même format `results.json` que S2218. Même critère Gap 3.

---

## S2220 — Notebook `notebooks/int8_vs_fp32_comparison.ipynb`

### Sections requises

1. **Tableau AUROC FP32 vs INT8** (4 colonnes × 2 datasets) :

| Dataset | AUROC FP32 | AUROC INT8 | Δ AUROC | Gap 3 ✅/❌ |
|---------|:----------:|:----------:|:-------:|:-----------:|
| CWRU | | | | |
| CMAPSS | | | | |

2. **Barplot comparatif** : AUROC FP32 vs INT8 par dataset, avec ligne pointillée `Δ = 0.02`

3. **Courbes AF FP32 vs INT8** : average forgetting par tâche (INT8 oublie-t-il plus ?)

4. **Analyse mémoire** : `ram_peak_bytes FP32` vs `ram_peak_bytes INT8` (réduction attendue ×4)

5. **Cellule conclusion Gap 3** :
```python
gap3_cwru   = results_cwru["delta_auroc"] < 0.02
gap3_cmapss = results_cmapss["delta_auroc"] < 0.02
gap3_met    = gap3_cwru and gap3_cmapss

print(f"Gap 3 CWRU   : {'✅' if gap3_cwru   else '❌'} (Δ={results_cwru['delta_auroc']:.4f})")
print(f"Gap 3 CMAPSS : {'✅' if gap3_cmapss else '❌'} (Δ={results_cmapss['delta_auroc']:.4f})")
print(f"Gap 3 global : {'✅ COMBLÉ' if gap3_met else '❌ NON COMBLÉ'}")
# FIXME(gap3) : si non comblé, investiguer les couches problématiques (fq_w1/fq_h1)
```

---

## Vérification end-to-end

```bash
# Critère final Gap 3
python -c "
import json
r1 = json.load(open('experiments/exp_S22_INT8_01/results.json'))
r2 = json.load(open('experiments/exp_S22_INT8_02/results.json'))
assert r1['gap3_criterion_met'], f\"Gap 3 CWRU échoué : Δ={r1['delta_auroc']:.4f}\"
assert r2['gap3_criterion_met'], f\"Gap 3 CMAPSS échoué : Δ={r2['delta_auroc']:.4f}\"
print('Gap 3 comblé sur CWRU + CMAPSS ✅')
"
```

---

## Questions ouvertes

- `TODO(dorra)` : Confirmer le schéma de quantization : `per_tensor_affine` pour les activations ou `per_channel_affine` ? Sur MCU (ewc_head_int8.c), per-tensor est plus simple à implémenter.
- `TODO(dorra)` : Le straight-through estimator de PyTorch QAT est-il équivalent au gradient Q7 qu'on implémentera en C ? Vérifier cohérence Python ↔ C.
- `FIXME(gap3)` : Si `Δ AUROC > 0.02`, inspecter quel fake-quantizer est le plus destructeur (`fq_input`, `fq_h1`, `fq_h2` ou poids) et désactiver sélectivement.
