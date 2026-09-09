# Consommations RAM par expérience — CL-Embedded

> Document **généré** par `scripts/aggregate_ram.py` depuis `experiments/exp_S49_ram/summary.json` — **aucune valeur saisie à la main**. Régénérer : `python scripts/aggregate_ram.py`.

## 1. Méthode

Formule officielle (CR du 16 juillet 2026) : **`RAM totale = .data + .bss + pic de pile`**. La mesure précédente ne remontait que `.bss` → **sous-estimée** (la pile vit hors `.bss`). Détails et mécanisme (stack painting) : [`ram_measurement.md`](ram_measurement.md) et [`../presentation_ram_measurement.md`](../presentation_ram_measurement.md).

- **`.data`** : globales initialisées. **`.bss`** : globales à zéro / mémoire non constante (état CL). **Pic de pile** : high-water mark mesuré **après chaque phase** (`idle` → `inférence` → `mise à jour CL`).
- Board : `.bss`+pile réels (NUCLEO-F439ZI, `profiling_total_ram_bytes`). PC : pic `tracemalloc` d'un forward (`.data`/`.bss` non applicables → N/A).

## 2. Tableau par modèle × dataset × encodage × plateforme

| Modèle | Dataset | Encodage | Plateforme | `.data` | `.bss` | pic inférence | pic MAJ | total | ratio int8/fp32 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| EWC | monitoring | fp32 | board (NUCLEO-F439ZI) | 460 | 100 152 | 4 416 | 4 688 | 105 300 | — |
| EWC | monitoring | fp32 | PC (tracemalloc) | N/A | N/A | 1 992 | N/A | 1 992 | — |
| EWC | monitoring | int8 | board (NUCLEO-F439ZI) | 460 | 100 152 | 4 328 | 4 712 | 105 324 | 1.0002 |
| EWC | monitoring | int8 | PC (tracemalloc) | N/A | N/A | N/A | N/A | N/A | N/A |
| EWC | pronostia | fp32 | board (NUCLEO-F439ZI) | 460 | 105 036 | 4 424 | 4 696 | 110 192 | — |
| EWC | pronostia | fp32 | PC (tracemalloc) | N/A | N/A | 1 992 | N/A | 1 992 | — |
| EWC | pronostia | int8 | board (NUCLEO-F439ZI) | 460 | 105 036 | 4 336 | 4 720 | 110 216 | 1.0002 |
| EWC | pronostia | int8 | PC (tracemalloc) | N/A | N/A | N/A | N/A | N/A | N/A |
| HDC | monitoring | fp32 | board (NUCLEO-F439ZI) | 460 | 100 152 | 4 328 | 4 328 | 104 940 | — |
| HDC | monitoring | fp32 | PC (tracemalloc) | N/A | N/A | 15 136 | N/A | 15 136 | — |
| HDC | monitoring | int8 | board (NUCLEO-F439ZI) | 460 | 100 152 | 4 328 | 4 328 | 104 940 | 1 |
| HDC | monitoring | int8 | PC (tracemalloc) | N/A | N/A | N/A | N/A | N/A | N/A |
| HDC | pronostia | fp32 | board (NUCLEO-F439ZI) | 460 | 105 036 | 4 336 | 4 336 | 109 832 | — |
| HDC | pronostia | fp32 | PC (tracemalloc) | N/A | N/A | 15 136 | N/A | 15 136 | — |
| HDC | pronostia | int8 | board (NUCLEO-F439ZI) | 460 | 105 036 | 4 336 | 4 336 | 109 832 | 1 |
| HDC | pronostia | int8 | PC (tracemalloc) | N/A | N/A | N/A | N/A | N/A | N/A |
| TinyOL | monitoring | fp32 | board (NUCLEO-F439ZI) | 460 | 100 152 | 4 328 | 4 328 | 104 940 | — |
| TinyOL | monitoring | fp32 | PC (tracemalloc) | N/A | N/A | 2 280 | N/A | 2 280 | — |
| TinyOL | monitoring | int8 | board (NUCLEO-F439ZI) | 460 | 100 152 | 4 368 | 4 368 | 104 980 | 1.0004 |
| TinyOL | monitoring | int8 | PC (tracemalloc) | N/A | N/A | N/A | N/A | N/A | N/A |
| TinyOL | pronostia | fp32 | board (NUCLEO-F439ZI) | 460 | 105 036 | 4 336 | 4 336 | 109 832 | — |
| TinyOL | pronostia | fp32 | PC (tracemalloc) | N/A | N/A | 2 536 | N/A | 2 536 | — |
| TinyOL | pronostia | int8 | board (NUCLEO-F439ZI) | 460 | 105 036 | 4 376 | 4 376 | 109 872 | 1.0004 |
| TinyOL | pronostia | int8 | PC (tracemalloc) | N/A | N/A | N/A | N/A | N/A | N/A |
| Mahalanobis | monitoring | fp32 | board (NUCLEO-F439ZI) | 460 | 100 152 | 4 328 | 4 328 | 104 940 | — |
| Mahalanobis | monitoring | fp32 | PC (tracemalloc) | N/A | N/A | 4 540 | N/A | 4 540 | — |
| Mahalanobis | monitoring | int8 | board (NUCLEO-F439ZI) | N/A | N/A | N/A | N/A | N/A | N/A |
| Mahalanobis | monitoring | int8 | PC (tracemalloc) | N/A | N/A | N/A | N/A | N/A | N/A |
| Mahalanobis | pronostia | fp32 | board (NUCLEO-F439ZI) | 460 | 105 036 | 4 336 | 4 336 | 109 832 | — |
| Mahalanobis | pronostia | fp32 | PC (tracemalloc) | N/A | N/A | 5 308 | N/A | 5 308 | — |
| Mahalanobis | pronostia | int8 | board (NUCLEO-F439ZI) | N/A | N/A | N/A | N/A | N/A | N/A |
| Mahalanobis | pronostia | int8 | PC (tracemalloc) | N/A | N/A | N/A | N/A | N/A | N/A |

_Board et PC en lignes distinctes (jamais fusionnés). Toutes en octets._

## 3. Historique d'évolution du pic de pile

Pic mesuré **après chaque phase**, chronologiquement. La **mise à jour CL** (SGD embarqué) creuse plus la pile que l'inférence seule — pile transitoire, `.bss` inchangé.

### EWC

- **monitoring** (fp32, board) : idle 4 236 B → inférence 4 416 B → mise à jour CL 4 688 B
- **pronostia** (fp32, board) : idle 4 244 B → inférence 4 424 B → mise à jour CL 4 696 B

### HDC

- **monitoring** (fp32, board) : idle 4 236 B → inférence 4 328 B → mise à jour CL 4 328 B
- **pronostia** (fp32, board) : idle 4 244 B → inférence 4 336 B → mise à jour CL 4 336 B

### TinyOL

- **monitoring** (fp32, board) : idle 4 236 B → inférence 4 328 B → mise à jour CL 4 328 B
- **pronostia** (fp32, board) : idle 4 244 B → inférence 4 336 B → mise à jour CL 4 336 B

### Mahalanobis

- **monitoring** (fp32, board) : idle 4 236 B → inférence 4 328 B → mise à jour CL 4 328 B
- **pronostia** (fp32, board) : idle 4 244 B → inférence 4 336 B → mise à jour CL 4 336 B

## 4. Constats

- **`.bss` seul ≠ RAM totale** : rapporter `.bss` oublie la pile ; le total corrige la sous-estimation (cf. tableau §2).
- **Le pic dépend de la phase** : `pic MAJ ≥ pic inférence` sur les cellules board applicables (cf. historique §3).
- **INT8 réduit la mémoire non constante sans changer la pile** : le ratio int8/fp32 de la RAM totale (§2) reflète l'économie de poids, pas de la pile.
- **Pile ~partagée entre modèles** : trame unique `pipeline_run()` (max des branches) → pic de pile board ~homogène (cf. §2).

## 5. Limites & N/A honnêtes

- **EWC · monitoring · int8 · PC (tracemalloc)** : PC = proxy tracemalloc FP32 ; le gain INT8 (poids ÷4) est analytique — la RAM INT8 mesurée est côté board.
- **EWC · pronostia · int8 · PC (tracemalloc)** : PC = proxy tracemalloc FP32 ; le gain INT8 (poids ÷4) est analytique — la RAM INT8 mesurée est côté board.
- **HDC · monitoring · int8 · PC (tracemalloc)** : PC = proxy tracemalloc FP32 ; le gain INT8 (poids ÷4) est analytique — la RAM INT8 mesurée est côté board.
- **HDC · pronostia · int8 · PC (tracemalloc)** : PC = proxy tracemalloc FP32 ; le gain INT8 (poids ÷4) est analytique — la RAM INT8 mesurée est côté board.
- **TinyOL · monitoring · int8 · PC (tracemalloc)** : PC = proxy tracemalloc FP32 ; le gain INT8 (poids ÷4) est analytique — la RAM INT8 mesurée est côté board.
- **TinyOL · pronostia · int8 · PC (tracemalloc)** : PC = proxy tracemalloc FP32 ; le gain INT8 (poids ÷4) est analytique — la RAM INT8 mesurée est côté board.
- **Mahalanobis · monitoring · int8 · board (NUCLEO-F439ZI)** : Maha INT8 = build compilation dédié -DMAHA_INT8 + export ; hors périmètre mesure RAM S49 (gain = poids ÷4, cf. Sprint 29).
- **Mahalanobis · monitoring · int8 · PC (tracemalloc)** : PC = proxy tracemalloc FP32 ; le gain INT8 (poids ÷4) est analytique — la RAM INT8 mesurée est côté board.
- **Mahalanobis · pronostia · int8 · board (NUCLEO-F439ZI)** : Maha INT8 = build compilation dédié -DMAHA_INT8 + export ; hors périmètre mesure RAM S49 (gain = poids ÷4, cf. Sprint 29).
- **Mahalanobis · pronostia · int8 · PC (tracemalloc)** : PC = proxy tracemalloc FP32 ; le gain INT8 (poids ÷4) est analytique — la RAM INT8 mesurée est côté board.
