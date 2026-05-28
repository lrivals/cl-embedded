# S2001 — `ewc_consolidate()` : Fisher EMA update + snapshot θ*

| Champ | Valeur |
|-------|--------|
| **Sprint** | 20 |
| **Priorité** | 🔴 Critique |
| **Statut** | ✅ Terminé |
| **Durée estimée** | 3h |
| **Dépendances** | S19 `ewc_head.c` existant (forward + sgd_step ✅) |
| **Fichiers cibles** | `firmware/stm32f4_blink/src/ewc_head.c`, `firmware/stm32f4_blink/inc/ewc_head.h` |
| **Référence** | `Kirkpatrick2017EWC` (eq. 3), `src/models/ewc/fisher.py:compute_online_fisher()` |

---

## Contexte

`ewc_forward()`, `ewc_predict()` et `ewc_sgd_step()` sont implémentés (Sprint 19).
La fonction `ewc_consolidate()` — qui fixe θ* et met à jour la Fisher à la fin d'une tâche — **manque**.
Sans elle, le modèle ne peut pas passer en mode multi-tâche CL : le terme EWC dans `ewc_sgd_step()` reste nul (Fisher = 0).

---

## Ce qu'il faut implémenter

### Signature (à ajouter dans `ewc_head.h`)

```c
/**
 * Fisher EMA update + snapshot θ* — appeler à la fin de chaque tâche.
 * @param h      Tête EWC (modifie fisher* et star_w* in-place)
 * @param alpha  Décroissance EMA (depuis board_ewc.yaml:fisher_decay, typ. 0.9)
 */
void ewc_consolidate(EWCHead *h, float alpha);
```

### Logique

1. Pour chaque poids `w[j][i]` dans les 3 couches :
   - Approximation Fisher diagonale : `g2 = w[j][i] * w[j][i]` (proxy `grad² ≈ w²`)
   - Mise à jour EMA : `fisher[j][i] = alpha * fisher[j][i] + (1 - alpha) * g2`
   - Snapshot θ* : `star_w[j][i] = w[j][i]`
2. Pas de malloc — tout in-place dans la struct `EWCHead` déjà en .bss

### Implémentation (à insérer dans `ewc_head.c`)

```c
/* MEM: ewc_consolidate — 0 B stack (tout in-place sur EWCHead .bss ~9.5 Ko FP32) */
void ewc_consolidate(EWCHead *h, float alpha)
{
    float beta = 1.0f - alpha;

    for (int j = 0; j < EWC_H1; j++)
        for (int i = 0; i < EWC_IN; i++) {
            float g2 = h->w1[j][i] * h->w1[j][i];
            h->fisher1[j][i] = alpha * h->fisher1[j][i] + beta * g2;
            h->star_w1[j][i] = h->w1[j][i];
        }

    for (int j = 0; j < EWC_H2; j++)
        for (int i = 0; i < EWC_H1; i++) {
            float g2 = h->w2[j][i] * h->w2[j][i];
            h->fisher2[j][i] = alpha * h->fisher2[j][i] + beta * g2;
            h->star_w2[j][i] = h->w2[j][i];
        }

    for (int j = 0; j < EWC_OUT; j++)
        for (int i = 0; i < EWC_H2; i++) {
            float g2 = h->w3[j][i] * h->w3[j][i];
            h->fisher3[j][i] = alpha * h->fisher3[j][i] + beta * g2;
            h->star_w3[j][i] = h->w3[j][i];
        }
    /* Biais non inclus dans la Fisher (standard EWC Kirkpatrick 2017) */
}
```

---

## Tests Unity à ajouter (dans `test_ewc_head.c`)

| Test | Assertion |
|------|-----------|
| `test_consolidate_fisher_update` | `fisher1[0][0] ≈ 0.1 * w1[0][0]²` (alpha=0.9, Fisher init=0) |
| `test_consolidate_star_copy` | `star_w1[j][i] == w1[j][i]` pour tout j, i après consolidation |
| `test_consolidate_fisher_nonneg` | `fisher*[j][i] >= 0` pour toutes couches (propriété Fisher) |
| `test_ewc_penalty_active` | Après consolidation, `ewc_sgd_step()` applique un gradient de pénalité non nul |

---

## Vérification

- [ ] `make -C firmware/stm32f4_blink/ all` sans warnings
- [ ] `make -C firmware/stm32f4_blink/ test` : tous les tests `test_ewc_head.c` PASS
- [ ] Scénario 2 tâches : consolider tâche 0 → entraîner tâche 1 → vérifier que les poids importants pour tâche 0 bougent moins (pénalité effective)

---

## Questions ouvertes

- `TODO(arnaud)` : Proxy `grad² ≈ w²` acceptable pour publication, ou accumuler vrais gradients depuis `ewc_sgd_step` ?
- `TODO(dorra)` : Normaliser Fisher par le nombre de samples vus (N) pour éviter l'explosion de λ effectif ?
