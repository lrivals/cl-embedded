# S5004 — Coût latence INT8 détaillé (breakdown déquant/requant, cycles DWT)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 50 |
| **Priorité** | 🟡 Moyen — tâche ouverte du CR (détailler la différence de latence INT8). |
| **Statut** | ✅ Implémenté — board réelle NUCLEO-F439ZI (2 cellules mesurées, 0 CRC). Instrumentation DWT gardée `-DINT8_SEGMENT_PROFILE` ; MAC≈6785 cyc / requant≈3777 cyc (surcoût dominant) / déquant≈200 cyc ; total INT8 74 µs vs FP32 48–50 µs (+24 à +26 µs = paradoxe FPU S29 chiffré). Maha = N/A honnête. |
| **Durée estimée** | 5h |
| **Dépendances** | Carte NUCLEO · DWT (`profiling.c`) ✅ · kernel INT8 v2 (`ewc_head_int8_v2.c`, S39) ✅ · pipeline board ✅ |
| **Fichiers cibles** | `docs/context/int8_latency_breakdown.md` · `experiments/exp_S50_int8_latency/` |
| **Références** | CR §5 « Point ouvert — latence INT8 » · Sprint 29 (paradoxe latence FPU) |

## Contexte

Le CR pose la question : le processeur étant FP32 (FPU), passer en INT8 ajoute des **étapes de
déquantification/requantification** dans le pipeline. Il faut **détailler ces étapes et quantifier leur coût en
cycles**. C'est la clé du paradoxe latence FPU du Sprint 29 (INT8 ne réduit pas — voire augmente — la latence
sur Cortex-M4 FPU).

## Spec

### 1. Décomposition du pipeline d'inférence INT8

Instrumenter (segments DWT) les étapes ajoutées par la quantification :

| Étape | Description | Attendu vs FP32 |
|-------|-------------|-----------------|
| déquant entrées | INT8 → FP32 des features/poids avant MAC | surcoût |
| MAC | produit scalaire (reste FPU, S29) | ≈ identique |
| requant sortie | FP32 → INT8 / mise à l'échelle | surcoût |
| **total INT8** | somme | ≥ FP32 (paradoxe FPU) |

### 2. Mesure

- DWT P50/P99 par segment, board réelle, par modèle (EWC/Maha en INT8 v2).
- Comparaison directe INT8 vs FP32 (mêmes échantillons).
- Coût en **cycles** (µs × 180 MHz) des étapes déquant+requant.

### 3. Interprétation

Documenter que le gain INT8 est **RAM (÷4), pas latence** sur cette carte FPU (confirme S29) ; le vrai gain
latence exigerait une carte **sans FPU / INT8 natif** (perspective CR) ou SIMD CMSIS-NN.

## Format de sortie

- `experiments/exp_S50_int8_latency/{model}.json` : segments DWT (déquant/MAC/requant), total INT8 vs FP32, cycles.
- `docs/context/int8_latency_breakdown.md` : schéma du pipeline, table des coûts, interprétation FPU.

## Contraintes

- Mesure DWT réelle (pas proxy) ; board requise.
- Aucun cycle inventé avant run.
- Cohérence avec le message S29 (paradoxe FPU).

## Vérification

```bash
test -f docs/context/int8_latency_breakdown.md
python -c "import json;d=json.load(open('experiments/exp_S50_int8_latency/ewc.json'));\
assert set(d['segments']) >= {'dequant','mac','requant'}"
grep -i "FPU\|déquant\|requant\|cycles" docs/context/int8_latency_breakdown.md
```
