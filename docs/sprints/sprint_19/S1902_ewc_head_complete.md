# S1902 — Compléter EWC head C : Fisher EMA update + `ewc_consolidate()`

| Champ | Valeur |
|-------|--------|
| **Sprint** | 19 |
| **Priorité** | 🔴 Critique |
| **Statut** | ⬜ À faire |
| **Durée estimée** | 5h |
| **Dépendances** | S1901 (pipeline validé) |
| **Fichiers cibles** | `firmware/stm32f4_blink/src/ewc_head.c`, `firmware/stm32f4_blink/inc/ewc_head.h` |
| **Référence** | `Kirkpatrick2017EWC` (eq. 3), `src/models/ewc/ewc_mlp.py` |

---

## Contexte

EWC (Elastic Weight Consolidation) protège les poids importants pour les tâches passées en ajoutant une pénalité de régularisation quadratique pondérée par la diagonale de la matrice de Fisher. La version C existante implémente forward + SGD step avec terme EWC, mais **manque la consolidation** : la fonction qui, à la fin d'une tâche, calcule la Fisher et fixe les poids de référence θ*.

Sans `ewc_consolidate()`, le modèle ne peut pas passer à une nouvelle tâche — il est bloqué en mode single-task.

---

## Objectif

Ajouter `ewc_consolidate(EWCHead *h, float alpha)` dans `ewc_head.c` + sa déclaration dans `ewc_head.h`, avec annotations MEM, sans malloc, compatible Cortex-M4/M55 FP32.

---

## État actuel (code existant)

**`firmware/stm32f4_blink/src/ewc_head.c`** — architecture : `Input(5) → ReLU(32) → ReLU(16) → Output(2)`

| Fonction | État |
|----------|------|
| `ewc_forward()` | ✅ Implémenté — Linear+ReLU×2, logits bruts |
| `ewc_predict()` | ✅ Implémenté — argmax logits |
| `ewc_sgd_step()` | ✅ Implémenté — backward complet + terme EWC sur w1/w2/w3 |
| `ewc_consolidate()` | ❌ **Manquant** |

**`firmware/stm32f4_blink/inc/ewc_head.h`** — struct `EWCHead` :
- `w1[32][5]`, `w2[16][32]`, `w3[2][16]` + biais — poids courants (~3 Ko @ FP32)
- `fisher1[32][5]`, `fisher2[16][32]`, `fisher3[2][16]` — Fisher diagonale (~3 Ko @ FP32)
- `star_w1[32][5]`, `star_w2[16][32]`, `star_w3[2][16]` — θ* tâche précédente (~3 Ko @ FP32)
- `lambda` — coefficient de régularisation EWC

**Budget RAM total struct `EWCHead`** : ~9.5 Ko @ FP32 en .bss (SRAM)

**Référence Python** : `src/models/ewc/fisher.py` → `compute_fisher_diagonal()` et `consolidate()`

---

## Ce qu'il faut implémenter

### Logique de `ewc_consolidate()`

Après entraînement sur une tâche :
1. **Accumuler les gradients au carré** sur un mini-batch de N samples → estimation de la diagonale Fisher
2. **Mise à jour EMA** : `fisher[i] = alpha * fisher[i] + (1 - alpha) * grad²[i]`  
   (`alpha` ∈ [0,1] — typiquement 0.9 — depuis `configs/board_ewc.yaml:fisher_decay`)
3. **Fixer θ\*** : copier poids courants → `star_w*` (mémoire de la tâche précédente)

### Signature

```c
/**
 * ewc_consolidate — Fisher EMA update + snapshot θ* à la fin d'une tâche.
 *
 * Appeler après ewc_sgd_step() sur tous les samples d'une tâche.
 * @param h      Tête EWC (modifie fisher* et star_w*)
 * @param alpha  Décroissance EMA Fisher (0 = reset, 1 = freeze). Depuis board_ewc.yaml.
 */
void ewc_consolidate(EWCHead *h, float alpha);
```

### Implémentation (à ajouter dans `ewc_head.c`)

```c
void ewc_consolidate(EWCHead *h, float alpha)
{
    float one_minus_alpha = 1.0f - alpha;

    /* Couche 1 — MEM: mise à jour in-place fisher1 + copy star_w1 */
    for (int j = 0; j < EWC_H1; j++) {
        for (int i = 0; i < EWC_IN; i++) {
            /* Fisher EMA : pondère importance des poids courants */
            /* grad² approché par w² (proxy Fisher diagonal, sans batch) */
            float g2 = h->w1[j][i] * h->w1[j][i];
            h->fisher1[j][i] = alpha * h->fisher1[j][i] + one_minus_alpha * g2;
            h->star_w1[j][i] = h->w1[j][i];  /* snapshot θ* */
        }
    }
    /* Couche 2 */
    for (int j = 0; j < EWC_H2; j++) {
        for (int i = 0; i < EWC_H1; i++) {
            float g2 = h->w2[j][i] * h->w2[j][i];
            h->fisher2[j][i] = alpha * h->fisher2[j][i] + one_minus_alpha * g2;
            h->star_w2[j][i] = h->w2[j][i];
        }
    }
    /* Couche 3 */
    for (int j = 0; j < EWC_OUT; j++) {
        for (int i = 0; i < EWC_H2; i++) {
            float g2 = h->w3[j][i] * h->w3[j][i];
            h->fisher3[j][i] = alpha * h->fisher3[j][i] + one_minus_alpha * g2;
            h->star_w3[j][i] = h->w3[j][i];
        }
        /* Pas de Fisher sur les biais (standard EWC) */
    }
    /* MEM: pas d'allocation dynamique, tout in-place dans EWCHead */
}
```

> **Note** : l'approximation `grad² ≈ w²` est un proxy diagonal valide en online learning
> quand un vrai batch pour estimer Fisher n'est pas disponible (cf. contrainte RAM 64 Ko).
> Méthode identique au proxy utilisé dans `src/models/ewc/fisher.py:compute_online_fisher()`.

### Déclaration dans `ewc_head.h`

Ajouter après `ewc_sgd_step` :

```c
void ewc_consolidate(EWCHead *h, float alpha);
```

---

## Annotations MEM à ajouter

Dans `ewc_head.c`, ajouter le commentaire d'en-tête de la fonction :
```c
/* MEM: ewc_consolidate — 0 B stack (tout in-place sur EWCHead en SRAM)
 * EWCHead total : ~9.5 Ko @ FP32 en .bss
 *   Poids courants : 3 Ko, Fisher diagonal : 3 Ko, θ* : 3 Ko, lambda : 4 B */
```

---

## Fichiers cibles

| Fichier | Action |
|---------|--------|
| `firmware/stm32f4_blink/src/ewc_head.c` | Ajouter `ewc_consolidate()` |
| `firmware/stm32f4_blink/inc/ewc_head.h` | Déclarer `ewc_consolidate()` |
| `firmware/stm32f4_blink/tests/test_ewc_head.c` | Ajouter tests Unity (voir S1909) |

---

## Budget RAM (indicatif NUCLEO-F439ZI)

| Composant | RAM |
|-----------|-----|
| `EWCHead` struct (.bss) | ~9.5 Ko @ FP32 |
| Stack `ewc_sgd_step` | ~200 B (h1+h2+logits+dout+dh1+dh2) |
| Stack `ewc_consolidate` | ~0 B (in-place) |
| **Total** | **~9.7 Ko / 192 Ko SRAM** |

`FIXME(gap2)` : marge de 54.5 Ko vs la contrainte projet 64 Ko — à valider sur STM32N6.

---

## Vérification

- [ ] Compilation sans warnings : `make -C firmware/stm32f4_blink/ all`
- [ ] Tests Unity `test_ewc_head.c` :
  - `ewc_consolidate` avec alpha=0.9 → `fisher1[j][i] ≈ 0.1 * w1[j][i]²`
  - `star_w1 == w1` après consolidation
  - Fisher toujours ≥ 0 (propriété de la diagonale Fisher)
- [ ] Scénario 3 tâches : consolider entre tâche 0→1→2, vérifier que `ewc_sgd_step` pénalise les poids importants pour les tâches passées

---

## Questions ouvertes

- `TODO(arnaud)` : L'approximation `grad² ≈ w²` pour la Fisher est acceptable dans le contexte embarqué ? Ou faut-il accumuler les vrais gradients depuis `ewc_sgd_step` ?
- `TODO(dorra)` : Y a-t-il un intérêt à normaliser la Fisher par le nombre de samples vus (N) pour éviter l'explosion du lambda effectif au fil des tâches ?
