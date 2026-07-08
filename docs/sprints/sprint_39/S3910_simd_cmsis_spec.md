# S3910 — Spec C optimisé SIMD CMSIS-NN/DSP (différé board)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 39 |
| **Priorité** | 🟢 Nice-to-have — seule voie crédible vers un gain *latence* INT8 ; mesure = board (Partie B) |
| **Statut** | ✅ Spec livrée (1er juillet 2026) — mesure board différée (Partie B S3917, toolchain CMSIS bloquée, cf. S2908) |
| **Durée estimée** | 1h (spec) + Partie B (mesure board) |
| **Dépendances** | S3907 (kernel v2, base à dériver) · toolchain CMSIS (à débloquer, `TODO(dorra)`) |
| **Fichier cible** | `docs/sprints/sprint_39/S3910_simd_cmsis_spec.md` (ce fichier) + futur `ewc_head_int8_v2_simd.c` |
| **Références** | `docs/sprints/sprint_29/S2908_cmsis_dsp.md` (blocage initial) · CMSIS-NN `arm_fully_connected_s8` |

---

## Contexte

Le constat majeur du sprint (S3901) : **l'INT8 ne gagne pas de latence** sur Cortex-M4 FPU car la boucle
déquantifie vers FP32. La seule façon d'inverser cela est un **vrai chemin entier SIMD** exploitant
l'extension DSP du M4 (`SMLAD` : 2 MAC q15 ou 4 MAC q7 par cycle) via CMSIS-NN/DSP. Cette piste était déjà
ouverte en **S2908** mais **bloquée** : `libarm_cortexM4lf_math.a` / `arm_math.h` absents de la toolchain
`arm-none-eabi-gcc`, et l'installation manuelle est proscrite par la spec → `TODO(dorra)`.

## Recherche : kernels INT8 optimisés disponibles

| Bibliothèque | Fonction | Apport |
|--------------|----------|--------|
| CMSIS-DSP | `arm_dot_prod_q7(a, b, n, &acc_q31)` | produit scalaire SIMD q7, accumulateur q31 |
| CMSIS-NN | `arm_fully_connected_s8(...)` | couche dense int8 complète (per-channel, requant int) |
| CMSIS-NN | `arm_nn_vec_mat_mult_t_s8(...)` | mat-vec int8 transposé, cœur des couches FC |

**`arm_fully_connected_s8`** est le candidat idéal : il fait un matmul **entier de bout en bout**
(accumulateur int32, requantification entière par-canal) — exactement ce qui manque au kernel actuel pour
gagner en latence, et il prend nativement des **scales par-canal** (alignés sur le v2 S3907).

## Plan d'implémentation (quand toolchain débloquée)

1. Vérifier la lib : `arm-none-eabi-gcc --print-file-name=libarm_cortexM4lf_math.a`.
2. Dériver `ewc_head_int8_v2_simd.c` du v2 : remplacer la boucle MAC par `arm_fully_connected_s8` (ou
   `arm_dot_prod_q7` si l'on garde la structure manuelle).
3. Gérer `EWC_IN=5`, `EWC_H1=32`, `EWC_H2=16` (non-multiples de 4 pour la couche 1 → padding ou tail).
4. Build conditionnel `-DUSE_CMSIS_NN` + `LDFLAGS += -larm_cortexM4lf_math`.
5. **Mesure latence DWT** (scalaire v2 vs SIMD) → S3917 (board).

## Interprétation attendue (à mesurer, ne pas affirmer)

| ratio SIMD/scalaire | Conclusion manuscrit |
|:-------------------:|----------------------|
| < 1.0 | CMSIS-NN rend l'INT8 **plus rapide** que FP32 → Gap 3 latence comblé |
| ≈ 1.0 | overhead annule le gain sur petits vecteurs (EWC_IN=5) |
| > 1.0 | confirme le résultat négatif S23/S29 → **INT8 = RAM-only** sur ce M4 |

> Tant que la mesure board n'a pas tourné, la conclusion reste celle de S2908 : *« Gap 3 comblé
> exclusivement sur la RAM (×2.7–4.0) ; l'accélération latence INT8 sur Cortex-M4 FPU reste un problème
> ouvert. »* — `FIXME(gap3)`.

## Vérification (Partie B, board)

```bash
arm-none-eabi-gcc --print-file-name=libarm_cortexM4lf_math.a   # PRÉSENT requis
# puis S3917 : bench DWT scalaire v2 vs SIMD → experiments/exp_S39_board/results_simd.json
```
