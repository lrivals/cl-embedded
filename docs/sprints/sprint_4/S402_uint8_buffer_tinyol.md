# S4-02 — Extension buffer UINT8 sur TinyOL

| Champ | Valeur |
|-------|--------|
| **ID** | S4-02 |
| **Sprint** | Sprint 4 — Semaine 4 (6–13 mai 2026) |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 3h |
| **Dépendances** | S4-01 (`quantization.py`) · S3-05 (`oto_head.py` + `TinyOLOnlineTrainer`) |
| **Fichiers cibles** | `src/models/tinyol/oto_head.py` (extension) · `configs/tinyol_config.yaml` |
| **Statut** | ✅ Terminé |

---

## Objectif

Étendre `TinyOLOnlineTrainer` avec un **buffer d'activations UINT8** : les embeddings produits par le backbone gelé sont stockés en UINT8 (1 octet/valeur) au lieu de FP32 (4 octets/valeur), réduisant la RAM du buffer de 4×.

Ce buffer permet un **mini-replay des embeddings** pour renforcer les tâches passées sans stocker les données brutes (non disponibles sur MCU de toute façon).

**État actuel** (`oto_head.py`) :
- `OtOHead` : Linear(9→1) + Sigmoid, 40 B @ FP32, 10 B @ INT8
- `TinyOLOnlineTrainer` : boucle SGD online, pas de buffer d'activations

**Après cette tâche** :
- `TinyOLOnlineTrainer` avec `use_uint8_buffer: bool` (config YAML)
- Buffer UINT8 de capacité `buffer_size` (depuis config), en RAM constante
- Dequantification à la volée pour le forward pass (pas de dégradation permanente)

**Critère de succès** : `python -c "from src.models.tinyol.oto_head import TinyOLOnlineTrainer"` passe, et le test `test_uint8_buffer_reduces_ram` est vert.

---

## Sous-tâches

### 1. Paramètres YAML à ajouter dans `configs/tinyol_config.yaml`

```yaml
oto_head:
  # ... params existants ...
  use_uint8_buffer: true      # active le buffer UINT8 (false = FP32 pur, comportement actuel)
  buffer_size: 50             # nombre max d'embeddings stockés  # MEM: 50×9×1 = 450 B @ UINT8 / 50×9×4 = 1800 B @ FP32
  buffer_replay_ratio: 0.2   # fraction de steps avec replay (0.2 = 1 replay / 5 updates)
```

### 2. Extension de `TinyOLOnlineTrainer`

```python
from src.utils.quantization import quantize_buffer, dequantize_uint8

class TinyOLOnlineTrainer:
    """
    [Extension S4-02] Ajout du buffer UINT8 pour mini-replay.

    Nouveaux attributs
    ------------------
    use_uint8_buffer : bool
        Si True, les embeddings sont stockés en UINT8 (compression 4×).
    buffer_size : int
        Capacité maximale du buffer (en nombre d'embeddings).
    _buffer_uint8 : Tensor | None, shape [N, embed_dim], dtype uint8
        Buffer circulaire d'embeddings quantifiés.  # MEM: N×embed_dim×1 B @ UINT8
    _buffer_labels : Tensor | None, shape [N], dtype float32
        Labels associés aux embeddings en buffer.   # MEM: N×4 B @ FP32
    _buffer_scale : float
        Paramètre de quantification (global sur le buffer).
    _buffer_zero_point : int
        Décalage de quantification.
    """

    def __init__(self, autoencoder, oto_head, config: dict) -> None:
        # ... init existant ...
        oto_cfg = config.get("oto_head", {})
        self.use_uint8_buffer: bool = oto_cfg.get("use_uint8_buffer", False)
        self.buffer_size: int = oto_cfg.get("buffer_size", 50)
        self.buffer_replay_ratio: float = oto_cfg.get("buffer_replay_ratio", 0.2)

        self._buffer_fp32: list[torch.Tensor] = []   # buffer temporaire avant quantification
        self._buffer_uint8: torch.Tensor | None = None  # MEM: buffer_size×embed_dim×1 B @ UINT8
        self._buffer_labels: torch.Tensor | None = None  # MEM: buffer_size×4 B @ FP32
        self._buffer_scale: float = 1.0
        self._buffer_zero_point: int = 0
        self._step_counter: int = 0

    def _add_to_buffer(self, embedding: torch.Tensor, label: float) -> None:
        """
        Ajoute un embedding au buffer. Si buffer_size atteint, requantifie tout.

        Stratégie FIFO — l'embedding le plus ancien est supprimé.
        """
        self._buffer_fp32.append(embedding.detach())
        if len(self._buffer_fp32) > self.buffer_size:
            self._buffer_fp32.pop(0)

        if self.use_uint8_buffer and len(self._buffer_fp32) >= 2:
            buf_uint8, scale, zp = quantize_buffer(self._buffer_fp32)
            self._buffer_uint8 = buf_uint8     # MEM: len×embed_dim×1 B @ UINT8
            self._buffer_labels = torch.tensor(
                [l for l in self._buffer_labels_raw], dtype=torch.float32
            )  # MEM: len×4 B @ FP32
            self._buffer_scale = scale
            self._buffer_zero_point = zp

    def _replay_from_buffer(self) -> None:
        """
        Effectue un mini-replay : tire 1 embedding du buffer UINT8, reconstruit en FP32,
        et effectue un step SGD sur la tête OtO.
        """
        if self._buffer_uint8 is None or len(self._buffer_uint8) == 0:
            return
        idx = torch.randint(0, len(self._buffer_uint8), (1,)).item()
        emb_uint8 = self._buffer_uint8[idx]           # MEM: embed_dim×1 B @ UINT8
        emb_fp32 = dequantize_uint8(                   # MEM: embed_dim×4 B @ FP32
            emb_uint8.unsqueeze(0),
            self._buffer_scale,
            self._buffer_zero_point,
        ).squeeze(0)
        label = self._buffer_labels[idx]
        # step SGD identique au step normal
        self.optimizer.zero_grad()
        pred = self.oto_head(emb_fp32)
        loss = F.binary_cross_entropy(pred, label.unsqueeze(0))
        loss.backward()
        self.optimizer.step()

    def update(self, x: torch.Tensor, y: float) -> float:
        """
        [Extension] : après le step SGD normal, ajoute l'embedding au buffer
        et effectue un replay conditionnel.
        """
        # ... update existant (forward backbone, step OtO) ...

        # Nouveau : buffer + replay conditionnel
        with torch.no_grad():
            embedding = self.autoencoder.encode(x)
        self._add_to_buffer(embedding, y)

        self._step_counter += 1
        if (self.use_uint8_buffer
                and self._step_counter % max(1, int(1 / self.buffer_replay_ratio)) == 0):
            self._replay_from_buffer()

        return loss.item()

    def get_buffer_ram_bytes(self) -> dict[str, int]:
        """
        Retourne l'empreinte RAM du buffer (utile pour profile_memory.py).

        Returns
        -------
        dict avec clés : "uint8_bytes", "fp32_equivalent_bytes", "compression_ratio"
        """
        if self._buffer_uint8 is None:
            return {"uint8_bytes": 0, "fp32_equivalent_bytes": 0, "compression_ratio": 1}
        n = self._buffer_uint8.numel()
        return {
            "uint8_bytes": n,                  # MEM: N×embed_dim×1 B @ UINT8
            "fp32_equivalent_bytes": n * 4,    # MEM: N×embed_dim×4 B @ FP32
            "compression_ratio": 4.0,
        }
```

### 3. Tests unitaires — `tests/test_uint8_buffer.py`

```python
import torch
from src.models.tinyol.oto_head import TinyOLOnlineTrainer

MOCK_CONFIG = {
    "oto_head": {
        "lr": 0.01,
        "use_uint8_buffer": True,
        "buffer_size": 10,
        "buffer_replay_ratio": 1.0,  # replay à chaque step
    }
}

def test_buffer_ram_reduces_4x(mock_trainer):
    """Buffer UINT8 doit occuper 4× moins que FP32 équivalent."""
    ram = mock_trainer.get_buffer_ram_bytes()
    assert ram["compression_ratio"] == 4.0
    assert ram["uint8_bytes"] * 4 == ram["fp32_equivalent_bytes"]

def test_buffer_dtype_is_uint8(mock_trainer):
    """_buffer_uint8 doit être de dtype torch.uint8."""
    assert mock_trainer._buffer_uint8.dtype == torch.uint8

def test_update_returns_loss(mock_trainer, sample_input):
    """update() retourne un scalaire de perte positif."""
    x, y = sample_input
    loss = mock_trainer.update(x, y)
    assert loss > 0

def test_buffer_fifo_capacity(mock_trainer, sample_input):
    """Le buffer ne dépasse pas buffer_size éléments."""
    x, y = sample_input
    for _ in range(20):
        mock_trainer.update(x, y)
    assert len(mock_trainer._buffer_fp32) <= 10
```

---

## Bilan mémoire attendu

| Composant | Avant (FP32) | Après (UINT8) | Ratio |
|-----------|:------------:|:-------------:|:-----:|
| Buffer embeddings (50 × 9) | 1 800 B | **450 B** | 4× |
| Labels buffer (50) | 200 B | 200 B | 1× |
| scale + zero_point | 0 B | 8 B | — |
| **Total buffer** | **2 000 B** | **658 B** | **~3×** |

---

## Critères d'acceptation

- [ ] `use_uint8_buffer: false` → comportement identique à avant (pas de régression)
- [ ] `use_uint8_buffer: true` → `_buffer_uint8.dtype == torch.uint8`
- [ ] `get_buffer_ram_bytes()["compression_ratio"] == 4.0`
- [ ] `update()` retourne un float de perte valide après replay
- [ ] La capacité buffer est respectée (FIFO — jamais plus de `buffer_size` éléments)
- [ ] Annotations `# MEM:` sur chaque allocation dans `_add_to_buffer` et `_replay_from_buffer`
- [ ] `pytest tests/test_uint8_buffer.py -v` passe

---

## Questions ouvertes

- `TODO(dorra)` : Lors du replay, doit-on clipper le gradient pour stabiliser l'entraînement (grad_clip MCU) ?
- `FIXME(gap3)` : Le delta AA (FP32 vs UINT8 buffer) sera mesuré dans exp_004 (S4-03). Si delta > 0.005, augmenter buffer_size ou passer à n_bits=16.
- `TODO(arnaud)` : La stratégie FIFO est-elle préférable à un buffer aléatoire (réservoir sampling) pour les données de roulements ?
