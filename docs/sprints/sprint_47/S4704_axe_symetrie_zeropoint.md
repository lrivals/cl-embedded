# S4704 — Axe symétrie / zero-point (aux bits critiques)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 47 |
| **Priorité** | 🟠 Importante — teste si le zero-point affine rachète la métrique là où les bas bits cassent. |
| **Statut** | 📝 Doc — spec complète ; implémentation à venir |
| **Durée estimée** | 3h |
| **Dépendances** | S4703 (bits critiques identifiés) · S4702 (`symmetry="affine"`) · `src/utils/quantization.py` |
| **Fichiers cibles** | `configs/quant_depth/ewc_sym_*.yaml`, `experiments/exp_S47_symmetry/` |
| **Références** | `compute_scale_zero_point`/`quantize_uint8` (`src/utils/quantization.py`) |

---

## Contexte

Le schéma actuel de l'émulateur est **signé symétrique** : la grille `[−qmax, +qmax]` est centrée sur 0. Or les
**activations post-ReLU sont ≥ 0** — la moitié négative de la grille est **gaspillée**, d'autant plus coûteux que
les bits sont rares (INT3/INT2). Un **zero-point affine** (`q = round(a/s)+z`, déquant `(q−z)·s`) place toute la
plage utile sur les niveaux disponibles. Cette tâche teste si l'affine **rachète l'AUROC aux bits critiques**
identifiés en S4703, ou si la per-channel suffit.

## Spec

### 1. Grille (ciblée, pas exhaustive)

Aux **bits critiques** repérés en S4703 (typiquement là où `delta_auroc` décroche — p. ex. INT3 et INT2),
comparer, par dataset :

**EWC × {Monitoring, Pronostia} × `weight_bits` ∈ {bits critiques S4703} × `symmetry` {symmetric, affine} × granularité gagnante S4703**

Le zero-point s'applique aux **activations** (post-ReLU asymétriques) ; les poids restent symétriques signés
(distribution centrée). Configs `configs/quant_depth/ewc_sym_<dataset>_<bits>.yaml`.

### 2. Sorties

`experiments/exp_S47_symmetry/exp_S47_ewc_<dataset>_<bits>_<sym>.json` : mêmes champs que S4703 + `symmetry`.
Analyse : `delta_auroc(affine) − delta_auroc(symmetric)` par (dataset, bits) → **gain du zero-point**.

### 3. Table (gabarit — `pending`)

| Dataset | bits critique | symmetric | affine | gain affine |
|---------|:---:|:---:|:---:|:---:|
| Monitoring | (S4703) | pending | pending | pending |
| Pronostia | (S4703) | pending | pending | pending |

## Contraintes

- Réutiliser `compute_scale_zero_point` — ne pas ré-implémenter l'affine.
- Ne balayer que les **bits critiques** (pas toute la grille S4703) — l'affine est sans effet notable à 8 bits.
- `pending`/`null` avant exécution.

## Vérification

```bash
python scripts/run_s47_quant_depth.py --sweep configs/quant_depth/ --filter sym
ls experiments/exp_S47_symmetry/*.json
python -c "import json,glob; assert all('symmetry' in json.load(open(f)) for f in glob.glob('experiments/exp_S47_symmetry/*.json'))"
```

---

## Résolution (implémentée)

_À compléter lors de l'implémentation._
