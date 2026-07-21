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

Liste figée (renseignée après S4703/S4704) dans ce doc + reprise par `docs/sprints/sprint_48/S4801_*.md`.
Toutes les valeurs sont traçables aux JSON `experiments/exp_S47_depth/` (per_channel symmetric, seed 42) :

| Rôle | Dataset | weight_bits | granularité | symétrie | ΔAUROC (émulé) | RAM ratio théo. | À porter S48 |
|------|---------|:---:|:---:|:---:|:---:|:---:|:---:|
| frontière | Monitoring | ternaire | per_channel | symmetric | −0.0021 | ×20.25 | ✅ |
| agressive | Monitoring | binaire | per_channel | symmetric | −0.0117 | ×32 | ✅ |
| frontière | Pronostia | ternaire | per_channel | symmetric | −0.0153 | ×20.25 | ✅ |
| agressive | Pronostia | binaire | per_channel | symmetric | −0.0275 (casse −0.02) | ×32 | ✅ |
| référence | — | 8 | per_channel | symmetric | ≈0 | ×4 | (déjà S39, kernel v2) |

**Logique de sélection (per_channel symmetric, critère Δ ≥ −0,02)** :

- **Monitoring** dégrade en douceur — `int8` +0.0002 → `int2` −0.0069 → `ternaire` −0.0021 → `binaire` −0.0117 :
  **toutes les profondeurs restent ≥ −0.02**, y compris le binaire. Frontière = **ternaire** (dernier cran
  avant l'extrême, Δ quasi nul) ; agressive = **binaire** (encore préservé, mais on mesure board le coût réel).
- **Pronostia** exhibe le cliff — `int2` −0.0086 → `ternaire` −0.0153 (dernier ≥ −0.02) → `binaire` −0.0275
  (**casse le seuil**). Frontière = **ternaire** ; agressive = **binaire** (mesure la chute board du cliff).
- **Aucune variante affine** portée : S4704 montre que le zero-point **n'aide nulle part** (Monitoring bien
  pire, jusqu'à Δ=−0.062 ; Pronostia neutre-à-pire) — c'est la **granularité per-channel** qui repousse le
  cliff, pas la symétrie. La frontière reste donc `symmetric` (critère 4 de sélection **non déclenché**).

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

**Sélection figée** (décision utilisateur ; traçable à `experiments/exp_S47_depth/`, per_channel symmetric) :

- **Frontière = ternaire** pour Monitoring **et** Pronostia — plus petit `weight_bits` préservant l'AUROC
  (Δ = −0.0021 / −0.0153, tous deux ≥ −0.02 ; ×20.25 théorique). Fichiers :
  `exp_S47_ewc_{monitoring,pronostia}_ternaire_per_channel.json`.
- **Agressive = binaire** pour les deux — un cran sous la frontière (×32 théorique). Sur **Monitoring** le
  binaire reste préservé (Δ = −0.0117) ; sur **Pronostia** il **casse** le seuil (Δ = −0.0275) : c'est
  précisément le cliff que S48 doit **mesurer sur board réelle** (chute + `.bss` bit-packée).
  Fichiers : `exp_S47_ewc_{monitoring,pronostia}_binaire_per_channel.json`.
- **Référence = int8 per_channel** (`int8_v2`), **déjà porté S39** (kernel `ewc_head_int8_v2.c`) — point
  de comparaison RAM/latence board.
- **Aucune variante affine** : S4704 (`exp_S47_symmetry/`) montre un gain du zero-point ≤ 0 sur toutes les
  cellules critiques → le critère 4 (inclure `affine` si le zero-point a aidé) **n'est pas déclenché**.

**Transmis à Sprint 48** : le gain sub-INT8 (×20.25 ternaire, ×32 binaire) n'est **réel qu'avec un kernel
bit-packé** (ternaire 2 bits/poids, binaire 1 bit/poids) ; l'émulateur PC ne mesure ni la `.bss` ni la
latence. **S48 mesure `.bss` avec ET sans packing** pour objectiver l'écart théorique/matérialisé, et confirme
la chute d'AUROC board (paradoxe FPU : gain RAM sans accélération, cf. S29).
