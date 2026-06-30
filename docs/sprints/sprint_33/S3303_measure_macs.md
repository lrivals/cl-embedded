# S3303 — `scripts/measure_macs.py` (cross-check torchinfo)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 33 |
| **Priorité** | 🟡 Important |
| **Statut** | ✅ Implémenté (23 juin 2026) |
| **Durée estimée** | 2h |
| **Dépendances** | S3301 (`compute_cost.py` étendu) |
| **Fichiers cibles** | `scripts/measure_macs.py` |
| **Références** | `src/evaluation/compute_cost.py`, `src/models/ewc/ewc_mlp.py`, `src/models/tinyol/*.py` |

---

## Contexte

Les MACs de `compute_cost.py` sont analytiques (somme de produits de dimensions). Avant de
les utiliser dans la campagne énergie/coût, il faut un cross-check indépendant pour les
modèles torch (EWC, TinyOL) via `torchinfo`. HDC et Mahalanobis ne sont pas des modèles
`torch.nn.Module` — pour eux, seule l'estimation analytique existe ; ce script le documente
explicitement plutôt que de forcer un outil inadapté.

---

## Spec

```python
# scripts/measure_macs.py
"""Confronte les MACs analytiques de compute_cost.py aux MACs mesurés par torchinfo
pour les modèles torch (EWC, TinyOL). Produit une table d'écart.
"""

def measure_macs_torchinfo(model: torch.nn.Module, input_shape: tuple) -> int:
    """Utilise torchinfo.summary(model, input_size=input_shape, verbose=0) -> total_mult_adds."""
    ...

def compare_analytical_vs_tool(model_name: str, model: torch.nn.Module | None,
                                input_shape: tuple, **macs_kwargs) -> dict:
    """
    Retourne :
    {
        "model": model_name,
        "macs_analytical": int,
        "macs_torchinfo": int | None,   # None si non-torch (HDC, Mahalanobis)
        "delta_pct": float | None,
        "tool_applicable": bool,
        "justification": str,           # ex. "HDC : opérations binaires non-MAC, analytique seul"
    }
    """
    ...
```

CLI :

```bash
python scripts/measure_macs.py --model ewc --config configs/board_ewc.yaml
python scripts/measure_macs.py --model hdc --config configs/board_hdc.yaml   # tool_applicable=False
```

**Règles** :
- EWC / TinyOL : comparaison réelle analytique ↔ `torchinfo`, écart documenté en %.
- HDC / Mahalanobis : `tool_applicable=False` avec justification écrite (pas de
  `torch.nn.Module` sous-jacent — encodage binaire / distance quadratique, pas de couches
  linéaires au sens torchinfo) — ne pas inventer un wrapper torch artificiel juste pour
  satisfaire l'outil.

---

## Vérification

```bash
python scripts/measure_macs.py --model ewc --config configs/board_ewc.yaml
python scripts/measure_macs.py --model tinyol --config configs/board_tinyol.yaml
pytest tests/test_measure_macs.py -v   # si couvert par S3309, sinon smoke manuel ci-dessus
```
