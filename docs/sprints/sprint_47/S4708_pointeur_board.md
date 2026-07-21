# S4708 — Pointeur board (cadrage du Sprint 48)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 47 |
| **Priorité** | 🟢 Faible — passerelle vers le portage board (Sprint 48). |
| **Statut** | 📝 Doc — spec complète ; implémentation à venir |
| **Durée estimée** | 1h |
| **Dépendances** | S4703/S4704 (configs gagnantes) |
| **Fichiers cibles** | `docs/sprints/sprint_47/S4708_pointeur_board.md` (renvoi), consommé par `docs/sprints/sprint_48/` |
| **Références** | Sprint 48 (portage board) ; kernel `firmware/stm32f4_blink/src/ewc_head_int8_v2.c` |

---

## Contexte

Le sweep PC (S4703/S4704) identifie **les schémas gagnants** — le plus petit `weight_bits` préservant l'AUROC,
la granularité qui repousse le cliff, le bénéfice éventuel du zero-point. Cette tâche **sélectionne** les
configurations à porter sur NUCLEO-F439ZI au Sprint 48, où l'on mesure la **RAM `.bss` réelle** (bit-packée) et
la **latence** — les deux dimensions que l'émulateur ne mesure pas.

## Spec

### Critères de sélection des configs à porter (Sprint 48)

1. **Config « frontière »** : le plus petit `weight_bits` dont `delta_auroc ≥ −0,02` (per-channel), par dataset.
2. **Config « agressive »** : un cran en dessous (mesurer la chute board réelle et le gain RAM `.bss`).
3. **Référence** : INT8 per-channel (`int8_v2`, déjà porté S39) comme point de comparaison.
4. Si le zero-point affine a aidé (S4704), inclure la variante `affine` de la config frontière.

### Sortie

Liste figée (renseignée après S4703/S4704) dans ce doc + reprise par `docs/sprints/sprint_48/S4801_*.md` :

| Rôle | Dataset | weight_bits | granularité | symétrie | À porter S48 |
|------|---------|:---:|:---:|:---:|:---:|
| frontière | Monitoring | (S4703) | per_channel | (S4704) | ✅ |
| agressive | Monitoring | (S4703 −1 cran) | per_channel | symmetric | ✅ |
| frontière | Pronostia | (S4703) | per_channel | (S4704) | ✅ |
| agressive | Pronostia | (S4703 −1 cran) | per_channel | symmetric | ✅ |
| référence | — | 8 | per_channel | symmetric | (déjà S39) |

**Nuance transmise à S48** : le gain RAM sub-INT8 n'est réel qu'avec **kernel bit-packé** (INT4 = 2 poids/octet,
INT2 = 4/octet). S48 mesure le `.bss` **avec et sans packing** pour objectiver l'écart théorique/matérialisé.

## Contraintes

- Ce doc ne contient **aucun chiffre de résultat** avant exécution de S4703/S4704 (`(S4703)` = placeholder).
- La sélection est **traçable** aux JSON `exp_S47_depth/`/`exp_S47_symmetry/`.

## Vérification

```bash
# Après S4703/S4704 : la table de sélection est renseignée et pointe vers des JSON existants
grep -i "weight_bits\|per_channel" docs/sprints/sprint_47/S4708_pointeur_board.md
```

---

## Résolution (implémentée)

_À compléter lors de l'implémentation (après S4703/S4704)._
