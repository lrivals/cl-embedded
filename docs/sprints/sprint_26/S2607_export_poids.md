# S2607–S2608 — Export poids entraînés → C headers

| Champ | Valeur |
|-------|--------|
| **Sprint** | 26 |
| **Priorité** | 🔴 Critique (prérequis de S2601 et S2603) |
| **Statut** | ✅ TERMINÉ |
| **Durée estimée** | S2607 : 2h / S2608 : 1h = 3h total |
| **Dépendances** | Sprint 25 ✅ — `experiments/exp_S25_01/` (EWC RUL CMAPSS FD001, checkpoint `model_ewc_reg.pt`) + `experiments/exp_S25_03/` (EWC CWRU multi-class, checkpoint `model_ewc_mc.pt`) doivent exister |
| **Fichiers cibles** | `scripts/export_weights_ewc_rul.py`, `firmware/stm32f4_blink/inc/model_weights_rul.h`, `scripts/export_weights_ewc_multiclass.py`, `firmware/stm32f4_blink/inc/model_weights_multiclass.h` |
| **Référence** | `scripts/export_weights_c.py` (pattern export EWC binaire), `firmware/stm32f4_blink/inc/model_weights.h` (format C header de référence) |

---

## Contexte

Les têtes C `ewc_head_regression.c` et `ewc_head_multiclass.c` sont initialisées en `pipeline_init()` avec des poids Xavier aléatoires. Pour reproduire les résultats PC sur board, il faut **charger les poids entraînés** depuis les checkpoints Sprint 25. L'export se fait en deux étapes :

1. `export_weights_ewc_rul.py` → lit `model_ewc_reg.pt` → génère `model_weights_rul.h`
2. `export_weights_ewc_multiclass.py` → lit `model_ewc_mc.pt` → génère `model_weights_multiclass.h`

`pipeline_init()` sera ensuite modifié pour charger ces poids via `memcpy` depuis la Flash (pattern identique à TinyOL dans `pipeline.c` existant).

> **Règle absolue** : ne pas modifier `model_weights.h` à la main. Toujours générer via script. Ne pas committer `model_weights_rul.h` avec des poids placeholders.

---

## S2607 — `scripts/export_weights_ewc_rul.py`

### Spec complète

```python
#!/usr/bin/env python3
"""
export_weights_ewc_rul.py — Exporte EWCMlpRegressor entraîné → C header FP32.

Usage :
    python scripts/export_weights_ewc_rul.py \\
        --checkpoint experiments/exp_S25_01/model_ewc_reg.pt \\
        --output firmware/stm32f4_blink/inc/model_weights_rul.h \\
        [--input-dim 5] [--hidden-dims 32 16]

Sortie : model_weights_rul.h avec :
    const float EWC_REG_W1[EWC_REG_H1][EWC_REG_IN]
    const float EWC_REG_B1[EWC_REG_H1]
    const float EWC_REG_W2[EWC_REG_H2][EWC_REG_H1]
    const float EWC_REG_B2[EWC_REG_H2]
    const float EWC_REG_W3[EWC_REG_OUT][EWC_REG_H2]
    const float EWC_REG_B3[EWC_REG_OUT]
"""

from __future__ import annotations
import argparse
import hashlib
import struct
from pathlib import Path

import torch

from src.models.ewc.ewc_mlp_regression import EWCMlpRegressor


def tensor_to_c_array(t: torch.Tensor, name: str, shape_str: str) -> str:
    """Convertit un tenseur PyTorch en déclaration C const float[][]."""
    flat = t.detach().cpu().float().numpy().flatten()
    vals = ", ".join(f"{v:.8f}f" for v in flat)
    return f"static const float {name}{shape_str} = {{\n    {vals}\n}};\n"


def export_ewc_reg(
    checkpoint_path: Path,
    output_path: Path,
    input_dim: int = 5,
    hidden_dims: list[int] | None = None,
) -> None:
    if hidden_dims is None:
        hidden_dims = [32, 16]

    model = EWCMlpRegressor(input_dim=input_dim, hidden_dims=hidden_dims)
    state = torch.load(checkpoint_path, map_location="cpu")

    # Accepte un dict state_dict ou un dict wrappé {"model_state_dict": ...}
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()

    h1, h2 = hidden_dims[0], hidden_dims[1]

    lines: list[str] = [
        f"/* model_weights_rul.h — AUTO-GÉNÉRÉ par export_weights_ewc_rul.py",
        f" * Source : {checkpoint_path}",
        f" * SHA256  : {hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()[:16]}...",
        f" * Ne pas modifier manuellement — régénérer via le script.",
        f" * Référence : ewc_head_regression.h (EWC_REG_IN={input_dim}, H1={h1}, H2={h2})",
        f" */",
        f"#pragma once",
        f"",
        f"/* Inclure ewc_head_regression.h avant ce header pour les #define */",
        f"",
    ]

    w1 = model.fc1.weight.data   # shape [H1, IN]
    b1 = model.fc1.bias.data     # shape [H1]
    w2 = model.fc2.weight.data   # shape [H2, H1]
    b2 = model.fc2.bias.data     # shape [H2]
    w3 = model.fc3.weight.data   # shape [1, H2]
    b3 = model.fc3.bias.data     # shape [1]

    lines.append(tensor_to_c_array(w1, "EWC_REG_W1_INIT", f"[{h1}][{input_dim}]"))
    lines.append(tensor_to_c_array(b1, "EWC_REG_B1_INIT", f"[{h1}]"))
    lines.append(tensor_to_c_array(w2, "EWC_REG_W2_INIT", f"[{h2}][{h1}]"))
    lines.append(tensor_to_c_array(b2, "EWC_REG_B2_INIT", f"[{h2}]"))
    lines.append(tensor_to_c_array(w3, "EWC_REG_W3_INIT", f"[1][{h2}]"))
    lines.append(tensor_to_c_array(b3, "EWC_REG_B3_INIT", f"[1]"))

    output_path.write_text("\n".join(lines))
    print(f"Exporté {output_path} ({output_path.stat().st_size // 1024} Ko)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export EWCMlpRegressor → C header")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output",     type=Path,
                        default=Path("firmware/stm32f4_blink/inc/model_weights_rul.h"))
    parser.add_argument("--input-dim",  type=int, default=5)
    parser.add_argument("--hidden-dims", type=int, nargs=2, default=[32, 16])
    args = parser.parse_args()

    export_ewc_reg(args.checkpoint, args.output, args.input_dim, args.hidden_dims)


if __name__ == "__main__":
    main()
```

### Usage dans `pipeline_init()` (ajout à faire en S2601)

```c
/* Dans pipeline.c — pipeline_init() — après ewc_reg_init(&g_ewc_reg) : */
#include "model_weights_rul.h"

memcpy(g_ewc_reg.w1, EWC_REG_W1_INIT, sizeof(g_ewc_reg.w1));
memcpy(g_ewc_reg.b1, EWC_REG_B1_INIT, sizeof(g_ewc_reg.b1));
memcpy(g_ewc_reg.w2, EWC_REG_W2_INIT, sizeof(g_ewc_reg.w2));
memcpy(g_ewc_reg.b2, EWC_REG_B2_INIT, sizeof(g_ewc_reg.b2));
memcpy(g_ewc_reg.w3, EWC_REG_W3_INIT, sizeof(g_ewc_reg.w3));
memcpy(g_ewc_reg.b3, EWC_REG_B3_INIT, sizeof(g_ewc_reg.b3));
```

### Vérification

```bash
# Lancer l'export
python scripts/export_weights_ewc_rul.py \
    --checkpoint experiments/exp_S25_01/model_ewc_reg.pt \
    --output firmware/stm32f4_blink/inc/model_weights_rul.h

# Vérifier que les valeurs correspondent au modèle PyTorch
python -c "
import torch
from src.models.ewc.ewc_mlp_regression import EWCMlpRegressor
from pathlib import Path

model = EWCMlpRegressor(input_dim=5)
model.load_state_dict(torch.load('experiments/exp_S25_01/model_ewc_reg.pt'))

# Lire une valeur depuis le header et comparer
header = Path('firmware/stm32f4_blink/inc/model_weights_rul.h').read_text()
# Chercher la première valeur de W1
first_w1_py = model.fc1.weight.data[0, 0].item()
print(f'W1[0][0] Python : {first_w1_py:.8f}')
print('Vérifier manuellement dans model_weights_rul.h la première valeur de EWC_REG_W1_INIT')
"

# Compilation avec les poids
make -C firmware/stm32f4_blink all
```

---

## S2608 — `scripts/export_weights_ewc_multiclass.py`

### Spec complète

```python
#!/usr/bin/env python3
"""
export_weights_ewc_multiclass.py — Exporte EWCMlpMulticlass entraîné → C header FP32.

Usage :
    python scripts/export_weights_ewc_multiclass.py \\
        --checkpoint experiments/exp_S25_03/model_ewc_mc.pt \\
        --output firmware/stm32f4_blink/inc/model_weights_multiclass.h \\
        [--input-dim 9] [--n-classes 10] [--hidden-dims 32 16]
"""

from __future__ import annotations
import argparse
import hashlib
from pathlib import Path

import torch

from src.models.ewc.ewc_mlp_multiclass import EWCMlpMulticlass


def tensor_to_c_array(t: torch.Tensor, name: str, shape_str: str) -> str:
    flat = t.detach().cpu().float().numpy().flatten()
    vals = ", ".join(f"{v:.8f}f" for v in flat)
    return f"static const float {name}{shape_str} = {{\n    {vals}\n}};\n"


def export_ewc_mc(
    checkpoint_path: Path,
    output_path: Path,
    input_dim: int = 9,
    n_classes: int = 10,
    hidden_dims: list[int] | None = None,
) -> None:
    if hidden_dims is None:
        hidden_dims = [32, 16]

    model = EWCMlpMulticlass(input_dim=input_dim, n_classes=n_classes, hidden_dims=hidden_dims)
    state = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()

    h1, h2 = hidden_dims[0], hidden_dims[1]

    lines: list[str] = [
        f"/* model_weights_multiclass.h — AUTO-GÉNÉRÉ par export_weights_ewc_multiclass.py",
        f" * Source    : {checkpoint_path}",
        f" * SHA256    : {hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()[:16]}...",
        f" * N_CLASSES : {n_classes} (CWRU=10, Paderborn=3)",
        f" * Ne pas modifier manuellement.",
        f" */",
        f"#pragma once",
        f"",
    ]

    w1 = model.fc1.weight.data   # [H1, IN]
    b1 = model.fc1.bias.data
    w2 = model.fc2.weight.data   # [H2, H1]
    b2 = model.fc2.bias.data
    w3 = model.fc3.weight.data   # [N_CLASSES, H2]
    b3 = model.fc3.bias.data

    lines.append(tensor_to_c_array(w1, "EWC_MC_W1_INIT", f"[{h1}][{input_dim}]"))
    lines.append(tensor_to_c_array(b1, "EWC_MC_B1_INIT", f"[{h1}]"))
    lines.append(tensor_to_c_array(w2, "EWC_MC_W2_INIT", f"[{h2}][{h1}]"))
    lines.append(tensor_to_c_array(b2, "EWC_MC_B2_INIT", f"[{h2}]"))
    lines.append(tensor_to_c_array(w3, "EWC_MC_W3_INIT", f"[{n_classes}][{h2}]"))
    lines.append(tensor_to_c_array(b3, "EWC_MC_B3_INIT", f"[{n_classes}]"))

    output_path.write_text("\n".join(lines))
    print(f"Exporté {output_path} (N_CLASSES={n_classes}, {output_path.stat().st_size // 1024} Ko)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export EWCMlpMulticlass → C header")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output",     type=Path,
                        default=Path("firmware/stm32f4_blink/inc/model_weights_multiclass.h"))
    parser.add_argument("--input-dim",  type=int, default=9)
    parser.add_argument("--n-classes",  type=int, default=10)
    parser.add_argument("--hidden-dims", type=int, nargs=2, default=[32, 16])
    args = parser.parse_args()

    export_ewc_mc(args.checkpoint, args.output, args.input_dim, args.n_classes, args.hidden_dims)


if __name__ == "__main__":
    main()
```

### Vérification

```bash
python scripts/export_weights_ewc_multiclass.py \
    --checkpoint experiments/exp_S25_03/model_ewc_mc.pt \
    --output firmware/stm32f4_blink/inc/model_weights_multiclass.h \
    --n-classes 10

# Vérifier N=10 cohérent avec Makefile -DEWC_MC_N_CLASSES=10
make -C firmware/stm32f4_blink all CFLAGS="-DEWC_MC_N_CLASSES=10"
```

---

## Résultats d'implémentation

| Sous-tâche | Statut | Notes |
|------------|:------:|-------|
| S2607 — `scripts/export_weights_ewc_rul.py` | ✅ | Créé — modèle `EWCMlpRegressor`, arrays `EWC_REG_*_INIT` |
| S2607 — `firmware/.../model_weights_rul.h` généré | ✅ | Généré depuis `exp_S25_01/model_ewc_reg.pt` (SHA `8b4f4277…`, 10 Ko) — export reproductible (diff vide) |
| S2608 — `scripts/export_weights_ewc_multiclass.py` | ✅ | Créé — modèle `EWCMlpMulticlass`, arrays `EWC_MC_*_INIT` |
| S2608 — `firmware/.../model_weights_multiclass.h` généré | ✅ | Généré depuis `exp_S25_03/model_ewc_mc.pt` (SHA `ddfe86d9…`, N_CLASSES=10, 14 Ko) — export reproductible (diff vide) |

---

## Questions ouvertes

- `TODO(arnaud)` : Si `experiments/exp_S25_01/` ne contient pas de checkpoint `.pt` (expérience pas encore lancée), les exports S2607/S2608 bloquent S2601/S2603. Confirmer l'ordre d'exécution : expériences Sprint 25 doivent être exécutées **avant** de lancer Sprint 26.
- `FIXME(gap1)` : Le checkpoint CMAPSS FD001 correspond à la tâche FD001 seulement. Pour le scénario CL 4 tâches (FD001→FD002→FD003→FD004), exporter le checkpoint après la dernière tâche pour le board. Préciser quelle tâche est exportée dans le header SHA256.
- `TODO(dorra)` : Les poids exportés sont en FP32. Pour une variante INT8 future, le script devrait quantifier via `torch.quantize_per_tensor` avant l'export. Ajouter un flag `--quantize` préparatoire (même si non utilisé en Sprint 26).
