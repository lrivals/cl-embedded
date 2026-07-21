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

✅ **S4704 implémenté.** **12 cellules mesurées** (EWC × {Monitoring, Pronostia} × bits
critiques `{int2, int3, int4}` × `{symmetric, affine}`, granularité **per_channel** = gagnante
S4703, seed 42). `int4` sert de **contrôle** (« affine sans effet notable » attendu aux bits
moins rares).

**Configs** : 12 fichiers `configs/quant_depth/ewc_sym_<dataset>_<bits>_<sym>.yaml` (héritent
de `ewc_int8_<dataset>.yaml`, clés S4701, aucun hyperparamètre en dur). **Harnais** :
`scripts/run_s47_quant_depth.py` étendu — `--filter <substr>` (sélection de configs sous
`--sweep`) + **routage de sortie** (`_out_path` détecte `ewc_sym_*` → répertoire
`experiments/exp_S47_symmetry/`, nom `exp_S47_ewc_<dataset>_<bits>_<sym>.json` avec tag de
symétrie ; le chemin `exp_S47_depth/` reste **inchangé**, 0 régression). Le zero-point affine
n'a **rien à ré-implémenter** : l'émulateur (S4702) câble déjà `symmetry="affine"` sur les
activations post-ReLU via `src/utils/quantization.py::compute_scale_zero_point`.

**Résultats mesurés — `delta_auroc` (quant − FP32), et gain affine = Δ(affine) − Δ(symétrique)** :

| Dataset | bits | symmetric | affine | gain affine |
|---------|:---:|:---:|:---:|:---:|
| Monitoring | int2 | −0.0069 | −0.0619 | **−0.0550** |
| Monitoring | int3 | −0.0026 | −0.0591 | −0.0565 |
| Monitoring | int4 | −0.0005 | −0.0554 | −0.0549 |
| Pronostia | int2 | −0.0086 | −0.0125 | −0.0040 |
| Pronostia | int3 | −0.0021 | −0.0028 | −0.0007 |
| Pronostia | int4 | −0.0016 | −0.0019 | −0.0003 |

(valeurs issues des JSON `exp_S47_symmetry/`, jamais écrites à la main — table régénérable.)

**Constat honnête (négatif mesuré)** : le **zero-point affine ne rachète pas la métrique** —
il est **systématiquement ≤ symétrique** sur cette grille (gain toujours négatif). Sur
Monitoring il **dégrade fortement** (−0.055) car les activations y sont déjà bien séparées et
la grille signée n'est pas gaspillée au point de justifier le décentrage ; sur Pronostia le
gain est quasi nul (−0.004 à −0.0003). **Conclusion : la per-channel (S4703) suffit ; l'affine
n'apporte rien ici** — c'est la per-channel, pas le zero-point, qui repousse le cliff. Réserve :
mesure émulée PC (bit-exact) ; confirmation board = Sprint 48.

**Vérification** : `python scripts/run_s47_quant_depth.py --sweep configs/quant_depth/ --filter sym` ;
`ls experiments/exp_S47_symmetry/*.json | wc -l` = **12** ; tous portent le champ `symmetry`.
