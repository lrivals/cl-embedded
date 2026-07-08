# S3901 — Audit & critique de l'implémentation INT8 actuelle

| Champ | Valeur |
|-------|--------|
| **Sprint** | 39 |
| **Priorité** | 🔴 Critique — fonde tout le diagnostic du sprint |
| **Statut** | ✅ Implémenté (30 juin 2026) |
| **Durée estimée** | 2h |
| **Dépendances** | Lecture `firmware/stm32f4_blink/src/ewc_head_int8.c` ✅ · `src/models/ewc/ewc_mlp_int8.py` ✅ |
| **Fichier cible** | `docs/sprints/sprint_39/S3901_audit_int8_actuel.md` (ce fichier) |
| **Références** | Sprint 28 (QAT PC), Sprint 29/36 (PTQ board), Sprint 34 (Q15 Mahalanobis), S2908 (SIMD bloqué) |

---

## Contexte

Le Gap 3 (quantification INT8) est comblé **sur la RAM** (×2.33–4.0) mais deux problèmes restent ouverts :
la **perte d'accuracy/F1 en INT8 sur board** (F1 EWC 0.07–0.15 vs FP32 ≈0.92, Sprint 36) et **l'absence de
gain de latence** (INT8 ~1.84× plus lent que FP32, Sprint 23/29). Ce document audite le code C INT8 actuel,
identifie les faiblesses, et établit le « avant » de la comparaison avec le kernel v2 du sprint.

---

## 1. Cartographie des chemins INT8

| Modèle | Fichier C | Schéma | Matmul | Accumulateur | Gain latence ? |
|--------|-----------|--------|--------|:------------:|:--------------:|
| EWC tête | `ewc_head_int8.c` | Q7 fixe `1/128`, par-tenseur | **déquant→FP32** | `int16_t` ⚠️ | ❌ (déquant) |
| Mahalanobis | `mahalanobis_int8.c` | INT8 affine | déquant→FP32 | FP32 | ❌ |
| Mahalanobis Q15 | `mahalanobis_q15.c` | sigma_inv int16 Q15 | déquant→FP32 | FP32 | ❌ (corrige précision, S34) |
| HDC | `hdc_int8.c` | ±1 exact + AM int16 | **entier int32** | `int32_t` ✅ | ⚠️ (entier mais pas SIMD) |
| TinyOL | `tinyol_int8.c` | Q7 poids + UINT8 act | déquant→FP32 | FP32 | ❌ |

**Constat structurel** : seul HDC fait un vrai matmul entier (int32). Tous les modèles neuronaux
(EWC, TinyOL) **déquantifient vers FP32 dans la boucle interne** → la FPU exécute autant d'opérations FP32
qu'un chemin FP32 pur, **plus** les conversions int→float. C'est la raison fondamentale pour laquelle l'INT8
ne gagne pas de latence sur Cortex-M4 FPU (et en perd même).

---

## 2. Faiblesses identifiées (tête EWC INT8)

### F1 — Accumulateur `int16_t` : overflow latent ⚠️ (bug)

[`ewc_head_int8.c:88-95`](../../../firmware/stm32f4_blink/src/ewc_head_int8.c#L88) :

```c
for (int j = 0; j < EWC_H1; j++) {
    int16_t acc = 0;                                   /* ← int16, pas int32 */
    for (int i = 0; i < EWC_IN; i++) {
        acc += (int16_t)h->w1[j][i] * (int16_t)x_q7[i]; /* Q7×Q7 = Q14, jusqu'à 127×127=16129 */
    }
    float val = (float)(acc >> 7) / 128.0f + h->b1[j]; /* >>7 suppose Q14 propre */
    h1[j] = relu_q7(float_to_q7(val));
}
```

- Un produit Q7×Q7 vaut jusqu'à 16129. **Sommer ≥3 termes peut dépasser ±32767** → l'accumulateur `int16_t`
  **déborde et wrap** (comportement non maîtrisé).
- Risque maximal à la **couche 2** (`EWC_H2`, 32 entrées) : même des produits modérés (~1000) somment à
  ~32000, au bord du débordement.
- Le `>> 7` suppose un accumulateur Q14 propre ; après wrap, la valeur est corrompue.
- **Correctif** : accumuler en `int32_t` (ce que fait déjà HDC). Coût RAM nul (registre).
- ⚠️ Effet **data-dépendant** : ne se déclenche pas toujours → à **mesurer** par l'émulateur (S3902/S3904),
  pas à affirmer en aveugle.

### F2 — Échelle fixe `1/128` par-tenseur (pas de calibration)

[`ewc_head_int8.h:19-22`](../../../firmware/stm32f4_blink/inc/ewc_head_int8.h#L19) et
[`ewc_head_int8.c:31-32`](../../../firmware/stm32f4_blink/src/ewc_head_int8.c#L31) :

```c
#define INT8_SCALE_W   (1.0f / 128.0f)   /* identique pour les 3 couches */
#define INT8_SCALE_ACT (1.0f / 128.0f)   /* activations post-ReLU */
```

- Conversion `float_to_q7(v) = (int8_t)(v * 128.0f)` ([`ewc_head_int8.h:61`](../../../firmware/stm32f4_blink/inc/ewc_head_int8.h#L61))
  suppose `|v| < 1`. **Toute activation ReLU > 1 est clampée à 0.99** → perte massive de dynamique.
- Suppose aussi `|poids| < 1` ; au-delà, saturation `SAT8`.
- C'est **différent du QAT PC** ([`ewc_mlp_int8.py:84-94`](../../../src/models/ewc/ewc_mlp_int8.py#L84)) qui
  calibre des scales **par-canal symétriques** (poids, `PerChannelMinMaxObserver`) et **par-tenseur affines
  calibrés** (activations, `HistogramObserver`). Le « board INT8 » n'est donc **pas** le modèle QAT qui
  préservait la métrique (Sprint 28, Δ≤0.006) : c'est une PTQ grossière à échelle fixe.

### F3 — PTQ one-shot ≠ QAT

[`ewc_head_int8.c:39-70`](../../../firmware/stm32f4_blink/src/ewc_head_int8.c#L39) (`ewc_int8_from_fp32`) :
conversion directe des poids FP32 → Q7 par `×128 + SAT8`, **sans recalibration, sans STE, sans données**.
À l'opposé du QAT PC où les gradients circulent en FP32 et les observers calibrent sur données live.
→ explique l'écart Sprint 36 (PTQ board F1 0.07–0.15) vs Sprint 28 (QAT PC Δ≤0.006).

### F4 — Déquantification dans la boucle → aucun gain latence

Le forward déquantifie immédiatement (`(float)(acc >> 7) / 128.0f`, ligne 94) et l'update fait un forward
**entièrement FP32** depuis poids déquantifiés ([`ewc_head_int8.c:140-160`](../../../firmware/stm32f4_blink/src/ewc_head_int8.c#L140)).
→ RAM gagnée (×4) mais latence ≥ FP32. Confirmé Sprint 23 (EWC INT8 0.461 ms vs FP32 0.251 ms).

### F5 — Aucune instruction SIMD / CMSIS-NN

Aucun `SMLAD`/`SMUAD`/`__SSAT`, ni `arm_math.h`. La piste `arm_dot_prod_q7` (S2908) est **bloquée** faute
de `libarm_cortexM4lf_math.a` dans la toolchain (`TODO(dorra)`). C'est la seule voie crédible vers un vrai
gain de latence INT8 sur Cortex-M4 → traitée en Partie B (board) du sprint.

### F6 — Biais FP32 + Mahalanobis grande dynamique (déjà connus)

- Biais restent FP32 ([`ewc_head_int8.h:32`](../../../firmware/stm32f4_blink/inc/ewc_head_int8.h#L32)) —
  impact mémoire faible, cohérence fake-quant ; non bloquant.
- Mahalanobis `sigma_inv` grande dynamique → INT8 ΔAUROC −0.24, **résolu en Q15** (Sprint 34). Sert de
  patron pour les schémas intermédiaires EWC/TinyOL de ce sprint.

---

## 3. Tableau avant / après attendu

| Facteur | Actuel (v1) | Kernel v2 (S3907) | Effet attendu |
|---------|-------------|-------------------|---------------|
| Accumulateur | `int16_t` (overflow) | `int32_t` | Supprime corruption (F1 ↑) |
| Scale poids | `1/128` fixe par-tenseur | par-canal calibré (export) | Récupère dynamique (F1 ↑) |
| Scale activation | `1/128` fixe (clamp >1) | calibré / Q15 | Évite clamp ReLU (F1 ↑) |
| Précision intermédiaire | Q7 (8-bit) | option Q15 (16-bit) | Fidélité 256× (F1 ↑, RAM ×2 au lieu ×4) |
| Latence | déquant→FP32 | idem (SIMD différé board) | RAM only, latence ≈ inchangée |

> Chaque ligne sera **chiffrée** par l'émulateur (S3902) et l'ablation (S3904) — aucune valeur affirmée ici
> sans mesure.

---

## 4. Questions ouvertes

- `TODO(dorra)` : compléter la toolchain avec CMSIS-NN/DSP (`libarm_cortexM4lf_math.a` + `arm_math.h`) pour
  débloquer le bench SIMD INT8 (S3910/S3917).
- `TODO(arnaud)` : pour le manuscrit, faut-il présenter l'INT8 comme « RAM-only » (latence problème ouvert
  sur FPU) ou attendre le résultat SIMD board avant de conclure sur le Gap 3 latence ?

---

## Vérification

Document de synthèse — pas de code exécutable. La validation des affirmations chiffrées (F1, overflow) est
opérée par S3902 (émulateur) et S3903 (parité vs logs board). Les claims F4/F5 (latence, SIMD) sont sourcés
Sprint 23/29 et S2908.
