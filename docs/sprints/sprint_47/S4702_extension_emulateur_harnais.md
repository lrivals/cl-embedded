# S4702 — Extension émulateur sub-INT8 + harnais PC `run_s47_quant_depth.py`

| Champ | Valeur |
|-------|--------|
| **Sprint** | 47 |
| **Priorité** | 🔴 Critique — moteur du sweep ; S4703/S4704 s'y branchent. |
| **Statut** | 📝 Doc — spec complète ; implémentation à venir |
| **Durée estimée** | 8h |
| **Dépendances** | S4701 (clés `weight_bits`/`granularity`/`symmetry`) · Sprint 39 ✅ (`int8_c_emulation.py`) · `src/utils/quantization.py` ✅ (zero-point) |
| **Fichiers cibles** | `src/utils/int8_c_emulation.py` (extension), `scripts/run_s47_quant_depth.py` (nouveau) |
| **Références** | `QuantConfig`, `_weight_scales`, `_quant_weight`, `_act_params`, `_forward_calibrated` (émulateur) ; `compute_scale_zero_point`/`quantize_uint8` (`src/utils/quantization.py`) |

---

## Contexte

L'émulateur bit-exact `src/utils/int8_c_emulation.py` **paramétrise déjà** :
- `n_bits` dans `_weight_scales(w, mode, n_bits)` et `_quant_weight(w, scales, n_bits)` (`qmax = (1<<(n_bits-1))-1`),
- la granularité dans `_weight_scales` (`fixed_128` / `per_tensor` / `per_channel`),
- l'activation dans `_act_params` (`q7_fixed` / `q7_calib` / `q15`).

Mais `QuantConfig` **fige les bits à 8 ou 16** (déduits de `act_repr`) et **ne gère que la symétrie signée**
(pas de zero-point). Cette tâche **expose la profondeur et la symétrie** sans réécrire le moteur, puis fournit
le harnais de sweep. **Le chemin `legacy_c` et les presets existants restent strictement inchangés** (0 régression
sur les tests S39).

## Spec

### 1. Extension de `QuantConfig`

Ajouter des champs (rétro-compatibles, valeurs par défaut = comportement actuel) :

```python
@dataclass(frozen=True)
class QuantConfig:
    weight_scale: WeightScale = "per_channel"
    act_repr: ActRepr = "q15"
    acc_dtype: AccDtype = "int32"
    weight_bits: int = 8            # NOUVEAU : 8|6|4|3|2 ; ternaire/binaire via weight_mode
    weight_mode: str = "linear"     # NOUVEAU : "linear" | "ternary" | "binary"
    symmetry: str = "symmetric"     # NOUVEAU : "symmetric" | "affine" (zero-point)
    name: str = "custom"

    @staticmethod
    def subint8(bits: int, granularity: str = "per_channel",
                symmetry: str = "symmetric", mode: str = "linear") -> "QuantConfig":
        """Preset générique du sweep S47 (profondeur × granularité × symétrie)."""
        ...
```

- **`weight_bits`** : passé tel quel à `_weight_scales`/`_quant_weight` (déjà génériques). `qmax` en découle.
- **`weight_mode`** :
  - `linear` : chemin actuel (`round(w/s)` saturé `[−qmax, qmax]`).
  - `ternary` : `q ∈ {−1,0,+1}` avec seuil `Δ = 0,7·mean|W[j,:]|` par canal (schéma TWN standard), scale par-canal.
  - `binary` : `q ∈ {−1,+1}`, scale par-canal `= mean|W[j,:]|` (BWN) ; **activations restent 8-bit**.
- **`symmetry`** :
  - `symmetric` : inchangé.
  - `affine` : pour les **activations** (poids restent symétriques signés), introduire un zero-point via
    `compute_scale_zero_point` de `src/utils/quantization.py` ; le forward déquantifie `(q − z)·s`. Cible les
    activations post-ReLU (≥ 0) où la moitié négative de la grille signée est gaspillée.

### 2. Câblage dans le forward

- `_weight_scales` / `_quant_weight` : déjà génériques en `n_bits` → passer `cfg.weight_bits`. Ajouter deux
  branches `ternary`/`binary` (petites fonctions dédiées, testées bit-à-bit).
- `_forward_calibrated` : accepter `symmetry="affine"` pour les activations (quantification affine + déquant
  `(q−z)·s`). **Le chemin `symmetric` reste le défaut et est inchangé.**
- **Parité C par construction** : les mêmes primitives serviront à l'export board (S4803), donc l'émulateur
  reste la **source unique** du schéma (comme `int8_v2` réutilise déjà `_weight_scales`/`_quant_weight`).

### 3. RAM théorique

Fonction utilitaire `theoretical_weight_ram(head, cfg) -> int` : `Σ n_params_couche × weight_bits / 8`
(bit-packé), + scales (`float32` par canal) + biais FP32. Retourne aussi le **ratio vs FP32**. C'est une valeur
**théorique** (cf. S4701 §2) ; la RAM `.bss` réelle est mesurée au Sprint 48.

### 4. Harnais `scripts/run_s47_quant_depth.py`

```
Pour chaque config configs/quant_depth/*.yaml :
  1. Charger la tête EWC de référence (checkpoint FP32, voie AUROC S28) pour (model, dataset).
  2. Extraire les poids via EWCHeadWeights.from_state_dict.
  3. Construire QuantConfig.subint8(weight_bits, granularity, symmetry, weight_mode).
  4. Calibrer les activations (calibrate_activations) sur le lot d'enrôlement.
  5. forward_quant(...) sur le jeu de test → logits → AUROC (+ agreement vs fp32).
  6. RAM théorique + proxy latence (nb MAC, informatif).
  7. Écrire experiments/exp_S47_depth/exp_S47_<model>_<dataset>_<bits>_<gran>[_<sym>].json
```

Schéma JSON (aligné S28/S39) :

```json
{
  "model": "ewc", "dataset": "monitoring",
  "weight_bits": 4, "granularity": "per_channel", "symmetry": "symmetric",
  "metric": "auroc",
  "auroc_fp32": null, "auroc_quant": null, "delta_auroc": null,
  "agreement_vs_fp32": null,
  "ram_weight_bytes_theoretical": null, "ram_ratio_vs_fp32": null,
  "seed": 42, "config_snapshot": { ... }
}
```

Tous les champs numériques à `null` tant que le script n'a pas tourné (**aucun chiffre inventé**).

## Contraintes

- **0 régression** : `QuantConfig.legacy_c()` et les presets S39 (`per_channel_int8`, `q15`, …) produisent des
  logits **strictement identiques** (test de non-régression obligatoire, réutilise les golden S39).
- Réutiliser `src/utils/quantization.py` pour le zero-point — ne pas ré-implémenter l'affine.
- Aucun hyperparamètre en dur : bits/granularité/symétrie **viennent des configs** (conforme CLAUDE.md).
- L'émulateur reste **sans dépendance torch** dans le chemin quantifié (NumPy) — portable/testable.

## Vérification

```bash
# Non-régression : presets S39 inchangés
pytest tests/ -k "int8_emulation or ablation" -q
# Le harnais tourne sur une config et produit un JSON au schéma attendu
python scripts/run_s47_quant_depth.py --config configs/quant_depth/ewc_monitoring_int4_perchannel.yaml
python -c "import json,glob; d=json.load(open(sorted(glob.glob('experiments/exp_S47_depth/*.json'))[-1])); assert set(['weight_bits','granularity','symmetry','delta_auroc'])<=d.keys()"
```

---

## Résolution (implémentée)

✅ **S4702 implémenté.** Émulateur étendu + harnais de sweep, **0 régression**.

**Extension `QuantConfig`** ([`src/utils/int8_c_emulation.py`](../../../src/utils/int8_c_emulation.py)) :
3 champs rétro-compatibles `weight_bits: int = 8`, `weight_mode: WeightMode = "linear"`,
`symmetry: Symmetry = "symmetric"` + `@staticmethod subint8(bits, granularity, symmetry, mode, act_repr)`.
**Piège résolu** : `q15()` porte désormais `weight_bits=16` (préserve les poids 16-bit — l'ancienne dérivation
`w_bits = 16 if act_repr=="q15"` a été retirée) ; `mixed_int8w_q15act` reste `weight_bits=8`.

**Câblage forward** (`_forward_calibrated`) : `w_bits = cfg.weight_bits` ; dispatch `_quant_weight_mode`
(linear = `_weight_scales`/`_quant_weight` existants ; `_ternary_weight` = TWN {−1,0,+1} seuil
`0.7·mean|W[j,:]|` par canal ; `_binary_weight` = BWN {−1,+1} scale par-canal) ; `symmetry="affine"` sur les
**activations** post-ReLU via `compute_scale_zero_point` ([`src/utils/quantization.py`](../../../src/utils/quantization.py),
**non ré-implémenté**) → accumulation `(q − z) @ wq.T`. Le chemin `symmetric` par-défaut est **inchangé**.

**RAM théorique** : `theoretical_weight_ram(head, cfg) -> (bytes, ratio)` — poids bit-packés (binaire 1b,
ternaire 2b, sinon `weight_bits`) + scales float32 + biais FP32 ; `ratio = 32/bits_effectifs` (×4/×8/×16/×32,
aligne S4701 §2 ; ternaire 32/1.58 ≈ ×20).

**Harnais** [`scripts/run_s47_quant_depth.py`](../../../scripts/run_s47_quant_depth.py) : réutilise `EWCAdapter`
(train FP32), `_first_task_train_X`, `_mean_auroc_over_tasks`, `_eval_quant_auroc`/`_task_eval_xy`/
`_weights_from_model` (S46), `forward_fp32`/`forward_quant`/`calibrate_activations`/`subint8`/
`theoretical_weight_ram`. Accord binaire = **seuil du logit d'anomalie** (`logit > 0`, la tête EWC a une sortie
unique → `argmax` serait dégénéré). CLI `--config` (une cellule) **et** `--sweep DIR`. JSON schéma S4702
(`auroc_fp32`/`auroc_quant`/`delta_auroc`/`agreement_vs_fp32`/`ram_*`/`config_snapshot`).

**Vérification** : `pytest -k "int8_emulation or ablation or s39 or s47"` → **35 PASS** (presets S39 golden
bit-identiques = 0 régression) ; smoke `--config ewc_monitoring_int4_perchannel.yaml` OK
(`auroc_quant=0.9741`, `ram_ratio=×8`).
