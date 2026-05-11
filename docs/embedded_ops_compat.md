# Compatibilité opérateurs ONNX — STM32Cube.AI

> Référence pour S1005. Mis à jour : 2026-05-11.

---

## 1. Version STM32Cube.AI requise

| Paramètre | Valeur |
|-----------|--------|
| Version minimale | **9.x** (pour STM32N6 / NeuralART Turbo) |
| Opset ONNX max supporté | **17** |
| Téléchargement | https://www.st.com/en/embedded-software/x-cube-ai.html (compte MyST requis) |

```bash
# Vérifier l'installation
stm32ai --version  # doit retourner ≥ 9.x
```

> `TODO(dorra)` : confirmer la version minimale exacte pour NeuralART Turbo (STM32N6).
> `TODO(dorra)` : quelle licence (gratuite / Eval / Pro) pour le pipeline complet INT8 ?
> `TODO(dorra)` : opset ONNX recommandé — 17 ou inférieur ?

**Statut** : STM32Cube.AI non installé en local (2026-05-11). Pipeline de repli TFLite disponible via `scripts/convert_tflite.py`.

---

## 2. Opérateurs ONNX supportés / exclus

### Opérateurs validés (opset ≤ 17, utilisés dans ce projet)

| Opérateur | Supporté STM32Cube.AI | Utilisé dans EWC | Utilisé dans TinyOL |
|-----------|-----------------------|-----------------|---------------------|
| `Gemm` | ✅ Oui | ✅ | ✅ |
| `Relu` | ✅ Oui | ✅ | ✅ |
| `Sigmoid` | ✅ Oui | ✅ | — |
| `Add` | ✅ Oui | — | — |
| `Mul` | ✅ Oui | — | — |
| `Reshape` | ✅ Oui | — | — |
| `Flatten` | ✅ Oui | — | — |
| `MatMul` | ✅ Oui | — | — |
| `Transpose` | ✅ Oui | — | — |
| `Concat` | ✅ Oui | — | — |
| `MaxPool` | ✅ Oui | — | — |
| `AveragePool` | ✅ Oui | — | — |
| `Conv` | ✅ Oui | — | — |

### Opérateurs exclus (non supportés par STM32Cube.AI)

| Opérateur | Raison d'exclusion | Impact sur ce projet |
|-----------|--------------------|----------------------|
| `LSTM` | Non supporté | ❌ — ne pas utiliser pour M1/M3 |
| `GRU` | Non supporté | ❌ |
| `RNN` | Non supporté | ❌ |
| `BatchNormalization` | Non supporté | ❌ — déjà exclu par contraintes MCU |
| `LayerNormalization` | Non supporté | ❌ — déjà exclu par contraintes MCU |
| `GroupNormalization` | Non supporté | ❌ |
| `InstanceNormalization` | Non supporté | ❌ |
| `Einsum` | Non supporté | ❌ |
| `Loop` | Non supporté | ❌ — pas de boucle dynamique en MCU |
| `Scan` | Non supporté | ❌ |
| `If` | Non supporté | ❌ |
| `DynamicQuantizeLinear` | Non supporté | ❌ — quantification via STM32Cube.AI uniquement |

> Source : `scripts/check_onnx_compat.py` constante `UNSUPPORTED_OPS`.
> Référence officielle : UM2878 — Getting Started with NeuralART Turbo.

---

## 3. Résultats `check_onnx_compat.py` — exp_160 (2026-05-11)

### EWC backbone (`experiments/exp_160/ewc_backbone.onnx`)

```
$ python scripts/check_onnx_compat.py --model ewc
Vérification : experiments/exp_160/ewc_backbone.onnx
[OK] Modèle compatible STM32Cube.AI (opset OK, ops supportés, shapes statiques)
```

| Propriété | Valeur |
|-----------|--------|
| Opset | 17 ✅ |
| Opérateurs | `Gemm`, `Relu`, `Sigmoid` ✅ |
| Entrée | `input` — shape [1, 4], dtype FLOAT32 ✅ |
| Sortie | `output` — shape [1, 1] |
| Shapes dynamiques | Aucune ✅ |
| Taille FP32 | 3.5 Ko |
| Taille INT8 (quantize_dynamic) | 4.3 Ko |
| Paramètres | 705 |

### TinyOL encoder (`experiments/exp_160/tinyol_encoder.onnx`)

```
$ python scripts/check_onnx_compat.py --model tinyol
Vérification : experiments/exp_160/tinyol_encoder.onnx
[OK] Modèle compatible STM32Cube.AI (opset OK, ops supportés, shapes statiques)
```

| Propriété | Valeur |
|-----------|--------|
| Opset | 17 ✅ |
| Opérateurs | `Gemm`, `Relu` ✅ |
| Entrée | `input` — shape [1, 25], dtype FLOAT32 ✅ |
| Sortie | `output` — shape [1, 8] |
| Shapes dynamiques | Aucune ✅ |
| Taille FP32 | 6.7 Ko |
| Taille INT8 (quantize_dynamic) | 5.0 Ko |
| Paramètres | 1 496 |

---

## 4. Pipeline STM32Cube.AI (quand installé)

```bash
# Analyse (rapport RAM/latency, pas de génération C)
stm32ai analyze \
    -m experiments/exp_160/ewc_backbone.onnx \
    --type onnx \
    --target stm32n6 \
    -o experiments/exp_160/stm32ai_analysis/

# Génération code C INT8
stm32ai generate \
    -m experiments/exp_160/ewc_backbone.onnx \
    --type onnx \
    --target stm32n6 \
    -o experiments/exp_160/stm32ai_output/
```

**Sorties attendues** :
- `stm32ai_analysis/report.json` — RAM, Flash, latency estimée, opérateurs
- `stm32ai_output/network.c` + `network.h`

> Note : le flag `--stm32ai` dans `scripts/export_onnx.py` tente automatiquement cette étape si `stm32ai` est dans le PATH.

---

## 5. Pipeline de repli TFLite (STM32Cube.AI indisponible)

Quand STM32Cube.AI n'est pas installé, utiliser `scripts/convert_tflite.py` :

```bash
# Installation des dépendances (optionnelles, non incluses dans requirements.txt)
pip install onnx-tf tensorflow

# Conversion ONNX → TFLite INT8
python scripts/convert_tflite.py --model ewc
# → experiments/exp_160/ewc_backbone_int8.tflite

python scripts/convert_tflite.py --model tinyol
# → experiments/exp_160/tinyol_encoder_int8.tflite

python scripts/convert_tflite.py --model all
```

**Pipeline interne** :
```
.onnx → onnx-tf backend → TF SavedModel (tmp) → TFLiteConverter INT8 → .tflite
```

Le fichier `.tflite` peut ensuite être déployé via TF Lite Micro sur Cortex-M (alternative au pipeline STM32Cube.AI).

---

## 6. Recommandations pour les modèles futurs (M1 TinyOL, M3 HDC)

- Utiliser uniquement des opérateurs de la section 2 (colonne ✅)
- Éviter tout module PyTorch avec `BatchNorm` ou `LayerNorm`
- Fixer `batch_size=1` dans `torch.onnx.export` (`dynamic_axes=None`)
- Opset cible : **17** (constante `OPSET_VERSION` dans `export_onnx.py`)
- Valider systématiquement avec `python scripts/check_onnx_compat.py --onnx <path>`
- Annoter les couches avec `# MEM:` pour tracer l'empreinte mémoire embarquée

---

## Références

- `scripts/check_onnx_compat.py` — validateur Python ONNX / STM32Cube.AI
- `scripts/export_onnx.py` — pipeline d'export PyTorch → ONNX + quantification PTQ
- `scripts/convert_tflite.py` — pipeline de repli ONNX → TFLite INT8
- `docs/context/hardware_constraints.md` — contraintes STM32N6 (64 Ko RAM, NPU INT8)
- `docs/sprints/sprint_phase2/S1002_onnx_export.md` — export ONNX
- `docs/sprints/sprint_phase2/S1005_stm32cubeai_setup.md` — ce sprint
