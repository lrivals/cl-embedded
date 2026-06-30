# S2402 — Extension UINT8 : EWC + HDC (exp_S24_01, exp_S24_02)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 24 |
| **Priorité** | 🔴 Critique |
| **Statut** | ✅ Complété |
| **Durée estimée** | S2402a : 1h30 / S2402b : 1h / S2402c : 1h / S2402d : 30 min = 4h total |
| **Dépendances** | `src/utils/quantization.py` ✅ (Sprint 4), `src/models/ewc/ewc_mlp.py` ✅, `src/models/hdc/hdc_classifier.py` ✅ |
| **Fichiers cibles** | `src/models/ewc/ewc_mlp.py`, `experiments/exp_S24_01/`, `experiments/exp_S24_02/` |
| **Référence** | `src/models/tinyol/oto_head.py` (pattern UINT8 OtO, Sprint 4), `src/utils/quantization.py` |

---

## Contexte

Sprint 4 a appliqué la quantization UINT8 uniquement à la tête OtO de TinyOL (`src/models/tinyol/oto_head.py`). EWC et HDC n'ont pas reçu de traitement UINT8 équivalent. Ce fichier couvre l'extension de cette amélioration aux deux modèles restants.

**Important** : seule la quantization des **activations en inférence** est visée (forward pass). La backpropagation reste en FP32 (cohérent avec Sprint 4, et contrainte hardware NUCLEO-F439ZI sans NPU).

---

## S2402a — Extension UINT8 activations forward EWC

### Objectif

Ajouter un mode `uint8_activations=True` dans `EWCMlpClassifier.forward()` en utilisant `quantize_tensor()` de `src/utils/quantization.py` sur les activations hidden des couches MLP.

### Pattern de référence (TinyOL OtO head, Sprint 4)

```python
# Dans oto_head.py — pattern à reproduire sur EWC
if self.uint8_activations:
    hidden = quantize_tensor(hidden, self._scale, self._zero_point)
    # MEM: 256 B @ FP32 → 64 B @ UINT8
```

### Modifications dans `src/models/ewc/ewc_mlp.py`

```python
from src.utils.quantization import quantize_tensor, calibrate_layer

class EWCMlpClassifier(BaseCLModel):
    def __init__(self, ..., uint8_activations: bool = False) -> None:
        ...
        self.uint8_activations = uint8_activations
        self._scales: dict[str, float] = {}      # calibré par couche
        self._zero_points: dict[str, int] = {}

    def calibrate_uint8(self, dataloader: DataLoader) -> None:
        """Calibre les paramètres UINT8 sur un batch représentatif."""
        with torch.no_grad():
            for x, _ in dataloader:
                h1 = torch.relu(self.fc1(x))
                self._scales["fc1"], self._zero_points["fc1"] = calibrate_layer(h1)
                # ... idem fc2

    def forward(self, x: Tensor) -> Tensor:
        h1 = torch.relu(self.fc1(x))  # MEM: 128 B @ FP32 / 32 B @ UINT8
        if self.uint8_activations:
            h1 = quantize_tensor(h1, self._scales["fc1"], self._zero_points["fc1"])
        h2 = torch.relu(self.fc2(h1))  # MEM: 64 B @ FP32 / 16 B @ UINT8
        if self.uint8_activations:
            h2 = quantize_tensor(h2, self._scales["fc2"], self._zero_points["fc2"])
        return self.fc3(h2)
```

### Vérification

```bash
python -c "
from src.models.ewc.ewc_mlp import EWCMlpClassifier
import torch
model = EWCMlpClassifier(input_dim=5, hidden_dims=[32, 16], output_dim=1, uint8_activations=True)
x = torch.randn(1, 5)
# calibrate avec dummy data
model.calibrate_uint8([(x, torch.tensor([0]))])
out = model(x)
print('EWC UINT8 forward OK, output shape:', out.shape)
"
```

---

## S2402b — exp_S24_01 : EWC UINT8 vs FP32 / Monitoring

### Commande

```bash
python scripts/train_ewc.py \
  --config configs/ewc_config.yaml \
  --uint8_activations \
  --exp_id exp_S24_01 \
  --output_dir experiments/exp_S24_01/
```

### Structure de sortie attendue

```
experiments/exp_S24_01/
├── config_snapshot.yaml
└── results.json
```

### Contenu `results.json`

```json
{
  "exp_id": "exp_S24_01",
  "model": "ewc",
  "dataset": "monitoring",
  "uint8_activations": true,
  "acc_final": "...",
  "avg_forgetting": "...",
  "bwt": "...",
  "ram_peak_bytes_fp32": "...",
  "ram_peak_bytes_uint8": "...",
  "compression_ratio_activations": "...",
  "delta_acc_vs_fp32": "...",
  "inference_latency_ms": "...",
  "n_params": "...",
  "reference_exp": "exp_001_ewc_monitoring_by_equipment"
}
```

### Critères de validation

- `ram_peak_bytes_uint8 < ram_peak_bytes_fp32` ✓
- `|delta_acc_vs_fp32| ≤ 0.01` (Δ accuracy acceptable pour Gap 3) ✓
- `FIXME(gap3)` : Si Δ acc > 0.01, documenter dans `results.json` et signaler à Arnaud

---

## S2402c — Profil RAM explicite HDC INT8

### Contexte

HDC est par conception une architecture INT : les hypervecteurs sont stockés en `int8` / `int32` (accumulateur AM). Cependant, aucun rapport RAM explicite comparant le HDC en "mode FP32 fictif" vs son mode INT natif n'a été produit. Ce profil sert de baseline embarquée pour le manuscrit.

### Modification dans `src/models/hdc/hdc_classifier.py`

Ajouter un rapport `get_memory_footprint()` qui calcule et retourne :
- RAM des hypervecteurs de base (`D × n_levels × 1 byte`, int8)
- RAM de l'associative memory (`D × n_classes × 4 bytes`, int32)
- Comparaison hypothétique FP32 (`D × (n_levels + n_classes) × 4 bytes`)
- Ratio de compression réel

```python
def get_memory_footprint(self) -> dict[str, int]:
    """Calcule l'empreinte mémoire réelle vs hypothèse FP32."""
    D = self.D
    n_levels = self.n_levels
    n_classes = self.n_classes
    return {
        "base_vectors_int8_bytes": D * n_levels * 1,       # MEM: D*n_levels B @ INT8
        "am_int32_bytes": D * n_classes * 4,               # MEM: D*n_classes*4 B @ INT32
        "total_int_bytes": D * n_levels * 1 + D * n_classes * 4,
        "hypothetical_fp32_bytes": D * (n_levels + n_classes) * 4,
        "compression_ratio": (D * (n_levels + n_classes) * 4) / (D * n_levels * 1 + D * n_classes * 4),
    }
```

---

## S2402d — exp_S24_02 : HDC INT8 profile / Monitoring

### Commande

```bash
python scripts/train_hdc.py \
  --config configs/hdc_config.yaml \
  --profile_int8 \
  --exp_id exp_S24_02 \
  --output_dir experiments/exp_S24_02/
```

### Contenu `results.json`

```json
{
  "exp_id": "exp_S24_02",
  "model": "hdc",
  "dataset": "monitoring",
  "native_int_architecture": true,
  "base_vectors_int8_bytes": "...",
  "am_int32_bytes": "...",
  "total_int_bytes": "...",
  "hypothetical_fp32_bytes": "...",
  "compression_ratio": "...",
  "acc_final": "...",
  "avg_forgetting": 0.0,
  "reference_exp": "exp_002_hdc_monitoring_by_equipment"
}
```

---

## Questions ouvertes

- `TODO(dorra)` : La calibration UINT8 d'EWC doit-elle être refaite à chaque nouvelle tâche CL, ou une calibration initiale sur Task 1 est-elle suffisante ?
- `TODO(arnaud)` : L'architecture HDC étant nativement INT, doit-on parler de "compression UINT8" ou simplement de "profil mémoire natif INT" dans le manuscrit ?
- `FIXME(gap3)` : S2402b fournit la première mesure EWC UINT8. Si Δ acc ≤ 0.01, cela supporte Gap 3 pour EWC. Documenter dans `docs/triple_gap.md`.
