# S4605 — Contexte HDC + Mahalanobis (N/A honnête sur l'axe des moments)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 46 |
| **Priorité** | 🟠 Important — cadre pourquoi l'axe avant/après **ne s'applique pas** à ces deux modèles ; évite la fausse impression d'une grille 4×3 homogène. |
| **Statut** | 📝 Doc — spec complète ; implémentation à venir |
| **Durée estimée** | 3h |
| **Dépendances** | `src/models/hdc/hdc_int8.py` ✅ · `src/models/unsupervised/mahalanobis_int8.py` ✅ · `configs/mahalanobis_{int8,q15}_{monitoring,pronostia}.yaml` ✅ · S4601 (mapping) |
| **Fichiers cibles** | `experiments/exp_S46_context/` |
| **Références** | S2802/S2803 (HDC/Maha INT8) · S3405–S3406 (Q15 Maha) · S4601 (mapping N/A) |

---

## Contexte

L'utilisateur a demandé « puis les autres modèles sur les 2 datasets ». Pour HDC et Mahalanobis, la
réponse honnête est que **l'axe des moments (avant/après entraînement) ne s'applique pas** — et ce sprint
le documente au lieu de fabriquer des cellules artificielles. Cette tâche produit un **contexte mesuré**
(pas 3-way) : pour HDC, montrer l'équivalence structurelle INT8≡FP32 ; pour Maha, montrer l'axe pertinent
INT8 vs Q15 sur les deux datasets.

## Spec

### 1. HDC — quantification structurelle (pas d'axe moment)

HDC est nativement entier : hypervecteurs int8 (±1), mémoire associative int16. Il n'y a **ni fake-quant
d'entraînement, ni conversion post-hoc** — l'« INT8 » est la structure du modèle. Résultat à reporter :

- `metric_fp32_hypothetique` vs `metric_int8_natif` : **égaux par construction** (à confirmer par run).
- `ram_ratio` structurel (`get_memory_footprint_int8` vs FP32 hypothétique, ≈ ×2–3, int16-AM).
- Cellules `before`/`after`/`both` = **N/A structurel** avec justification, pas un chiffre.

### 2. Mahalanobis — axe INT8 vs Q15 (pas d'axe moment)

Maha n'a pas d'entraînement par gradient (fit statistique) → `before` sans objet. Son axe pertinent est
le **format** de Σ⁻¹ :

- `int8` : Σ⁻¹ INT8 affine → **casse** sur grande dynamique (rappel S28 : ΔAUROC −0.236/−0.238).
- `q15` : Σ⁻¹ int16 Q15 → **récupère** (rappel S34 : ΔAUROC < 0.02 sur ds non dégénérés).

Reporter sur Monitoring + Pronostia : `auroc_fp32`, `auroc_int8`, `auroc_q15`, `delta_int8`, `delta_q15`.
Cellules `before`/`both` = **N/A hors-axe** (PTQ-only).

### 3. Exécution

```bash
# HDC : équivalence structurelle
python scripts/run_s46_quant_moment.py --model hdc --dataset monitoring \
    --moment context --output experiments/exp_S46_context/hdc_monitoring.json
# Maha : INT8 vs Q15 (réutilise configs existantes)
python scripts/run_s46_quant_moment.py --model mahalanobis --dataset pronostia \
    --moment context --output experiments/exp_S46_context/maha_pronostia.json
```

(`--moment context` = mode dédié qui produit la ligne de contexte au lieu de la grille 3-way.)

## Format de sortie

`experiments/exp_S46_context/{hdc,maha}_{monitoring,pronostia}.json` :

```json
{
  "model": "hdc",
  "dataset": "monitoring",
  "axis": "structural",           // "structural" (hdc) | "format_int8_q15" (maha)
  "moments_3way": "N/A",
  "na_reason": "HDC natif entier : quantification structurelle, pas de moment avant/après",
  "metric_name": "auroc",
  "values": { "fp32": null, "int8_native": null },
  "ram_ratio": null
}
```

```json
{
  "model": "mahalanobis",
  "dataset": "pronostia",
  "axis": "format_int8_q15",
  "moments_3way": "N/A",
  "na_reason": "Maha PTQ-only (fit statistique, pas d'entraînement gradient) ; axe = format Sigma^-1",
  "metric_name": "auroc",
  "values": { "fp32": null, "int8": null, "q15": null },
  "delta_int8": null, "delta_q15": null
}
```

## Contraintes

- **Aucune cellule 3-way artificielle** : le champ `moments_3way` vaut littéralement `"N/A"` avec
  `na_reason` explicite.
- HDC : ne pas inventer un FP32 réel s'il est hypothétique — l'étiqueter `fp32_hypothetique` comme en S24.
- Maha : réutiliser les configs `mahalanobis_{int8,q15}_*` existantes (pas de nouveaux hyperparamètres).

## Vérification

```bash
python -c "import json; d=json.load(open('experiments/exp_S46_context/hdc_monitoring.json')); \
assert d['moments_3way']=='N/A' and d['na_reason']"
python -c "import json; d=json.load(open('experiments/exp_S46_context/maha_pronostia.json')); \
assert set(d['values'])=={'fp32','int8','q15'}"
```

---

## Résolution (implémentée)

✅ **Implémenté et exécuté** (PC/émulé, seed 42). Nouveau mode `--moment context` dans
`scripts/run_s46_quant_moment.py` (+ `--model {hdc,mahalanobis}`), réutilisant `HDCAdapter`
et `MahalanobisAdapter` de `scripts/benchmark_int8_fp32.py`. **4 JSON** dans
`experiments/exp_S46_context/` (+ `config_snapshot.yaml`). **Aucune cellule 3-way
artificielle** : `moments_3way` vaut littéralement `"N/A"` avec `na_reason`.

### HDC — quantification structurelle (`axis="structural"`)

| Dataset | fp32 (hypothétique) | int8 natif | RAM ratio | Constat |
|---------|:---:|:---:|:---:|---------|
| Monitoring | 0.7443 | 0.7443 | ×2.33 | INT8 ≡ FP32 **par construction** |
| Pronostia  | 0.7231 | 0.7231 | ×2.33 | idem |

HDC est nativement entier (hypervecteurs int8 ±1, mémoire associative int16) → métrique INT8
native **strictement égale** au FP32 hypothétique (confirmé par run) ; le gain est **RAM
structurel** (×2.33, int16-AM). `fp32_is_hypothetical=True` tracé dans le JSON. Pas de moment
avant/après.

### Mahalanobis — axe format INT8 vs Q15 (`axis="format_int8_q15"`)

| Dataset | fp32 | int8 (Σ⁻¹ affine) | Δint8 | q15 (Σ⁻¹ int16) | Δq15 | Constat |
|---------|:---:|:---:|:---:|:---:|:---:|---------|
| Monitoring | 0.9725 | 0.9726 | +0.0001 | 0.9725 | 0.0000 | faible dynamique → int8 OK |
| Pronostia  | 0.8603 | 0.7469 | **−0.1133** | 0.8729 | **+0.0127** | **int8 casse, Q15 récupère** |

**Reproduit exactement le message S28/S34** : sur Pronostia (grande dynamique de Σ⁻¹) l'INT8
affine s'effondre (ΔAUROC −0.113) tandis que le Q15 int16 **récupère** (ΔAUROC +0.013 < 0.02).
Maha n'ayant pas d'entraînement gradient (fit statistique), l'axe `before`/`both` est **sans
objet** (PTQ-only) → non fabriqué.

### Vérification

```
$ python -c "import json;d=json.load(open('experiments/exp_S46_context/hdc_monitoring.json'));\
  assert d['moments_3way']=='N/A' and d['na_reason']"                # OK
$ python -c "import json;d=json.load(open('experiments/exp_S46_context/maha_pronostia.json'));\
  assert set(d['values'])=={'fp32','int8','q15'}"                    # OK
```
