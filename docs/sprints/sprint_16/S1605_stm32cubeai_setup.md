# S1005 — Setup STM32Cube.AI + Validation compatibilité opérateurs ONNX

| Champ | Valeur |
|-------|--------|
| **ID** | S1005 |
| **Sprint** | Sprint 16 — Semaine 1b (20–27 mai 2026) |
| **Priorité** | Critique |
| **Durée estimée** | 5h |
| **Dépendances** | S1001 ✅ (toolchain ARM OK), S1002 (export ONNX en cours) |
| **Fichiers cibles** | `scripts/check_onnx_compat.py`, `docs/embedded_ops_compat.md` |

---

## Objectif

Débloquer le `TODO(dorra)` de S1002 : installer la chaîne STM32Cube.AI et valider
que les modèles ONNX exportés sont compatibles avec le pipeline de quantification INT8.

**Pipeline cible** :
```
.onnx → stm32ai analyze → rapport RAM/latency + liste opérateurs → .c INT8
```

Si STM32Cube.AI est indisponible (licence, plateforme), fournir un pipeline de
repli via TFLite Converter.

---

## Contexte

S1002 suppose STM32Cube.AI déjà installé mais ce n'est pas le cas.
Sans cela, le pipeline ONNX → code C embarqué INT8 est bloqué.
Ce sprint doit être terminé **avant** de valider les critères d'acceptation de S1002.

---

## Sous-tâches

### 1. Installer STM32Cube.AI CLI

```bash
# Télécharger depuis https://www.st.com/en/embedded-software/x-cube-ai.html
# (compte MyST requis)
# Version cible : ≥ 9.x pour compatibilité STM32N6 / NeuralART Turbo

# Vérifier l'installation
stm32ai --version
```

Documenter la version installée dans `docs/embedded_ops_compat.md`.

> `TODO(dorra)` : confirmer la version minimale pour STM32N6 (NeuralART Turbo).
> `TODO(dorra)` : quelle licence (gratuite / Eval / Pro) est nécessaire pour le pipeline complet ?

### 2. Tester le pipeline ONNX → STM32Cube.AI

```bash
# Générer un rapport d'analyse (sans générer le code C)
stm32ai analyze -m experiments/exp_160/ewc_backbone.onnx \
                --type onnx \
                --target stm32f4 \
                -o experiments/exp_160/stm32ai_analysis/

# Générer le code C embarqué
stm32ai generate -m experiments/exp_160/ewc_backbone.onnx \
                 --type onnx \
                 --target stm32f4 \
                 -o experiments/exp_160/stm32ai_output/
```

**Sorties attendues** :
- `stm32ai_analysis/report.json` — RAM, Flash, latency estimée, opérateurs supportés
- `stm32ai_output/network.c` + `network.h`

### 3. Créer `scripts/check_onnx_compat.py`

Voir implémentation dans `scripts/check_onnx_compat.py`.

Usage :
```bash
python scripts/check_onnx_compat.py --model ewc \
    --onnx experiments/exp_160/ewc_backbone.onnx
python scripts/check_onnx_compat.py --model tinyol \
    --onnx experiments/exp_160/tinyol_encoder.onnx
```

### 4. Pipeline de repli TFLite (si STM32Cube.AI indisponible)

```bash
# ONNX → TF SavedModel (via onnx-tf ou onnx2tf)
pip install onnx-tf tensorflow

python -c "
import onnx
from onnx_tf.backend import prepare
model = onnx.load('experiments/exp_160/ewc_backbone.onnx')
tf_rep = prepare(model)
tf_rep.export_graph('experiments/exp_160/ewc_tf_savedmodel/')
"

# TF SavedModel → TFLite INT8
python -c "
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(
    'experiments/exp_160/ewc_tf_savedmodel/'
)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.int8]
tflite_model = converter.convert()
with open('experiments/exp_160/ewc_backbone_int8.tflite', 'wb') as f:
    f.write(tflite_model)
"
```

### 5. Documenter les opérateurs incompatibles

Remplir `docs/embedded_ops_compat.md` avec :
- Liste des opérateurs ONNX validés (supportés par STM32Cube.AI opset ≤ 17)
- Opérateurs à éviter dans les modèles Phase 1 (BatchNorm, LayerNorm, etc.)
- Résultat de `stm32ai analyze` pour EWC backbone + TinyOL encoder

---

## Critères d'acceptation

- [ ] `stm32ai --version` retourne une version ≥ 9.x (ou pipeline TFLite validé) — bloqué TODO(dorra)
- [x] `python scripts/check_onnx_compat.py --model ewc` passe sans erreur bloquante
- [ ] `stm32ai analyze` produit un `report.json` pour EWC backbone — bloqué TODO(dorra)
- [x] `docs/embedded_ops_compat.md` liste les opérateurs supportés/exclus
- [ ] `TODO(dorra)` résolu avec la version et la licence confirmées — en attente

---

## Questions ouvertes

- `TODO(dorra)` : version STM32Cube.AI minimale pour NeuralART Turbo (STM32N6) ?
- `TODO(dorra)` : opset ONNX recommandé — 17 ou inférieur ?
- Si STM32Cube.AI nécessite Windows : documenter workflow via VM ou Docker
