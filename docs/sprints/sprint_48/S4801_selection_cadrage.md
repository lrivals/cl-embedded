# S4801 — Sélection des configs gagnantes + cadrage build sub-INT8

| Champ | Valeur |
|-------|--------|
| **Sprint** | 48 |
| **Priorité** | 🔴 Critique — fige quoi porter et comment (flags de build). |
| **Statut** | 📝 Doc — spec complète ; implémentation à venir |
| **Durée estimée** | 2h |
| **Dépendances** | S4708 (sélection PC) · S4703/S4704 (sweep) |
| **Fichiers cibles** | `docs/sprints/sprint_48/S4801_selection_cadrage.md` |
| **Références** | `-DEWC_INT8_Q15`/`-DEWC_INT8_MIXED` (S39), `-DMAHA_INT8` (S2912) — précédents de sélection par compilation |

---

## Contexte

Le Sprint 47 a identifié les schémas gagnants (S4708). Cette tâche les fige comme **matrice de portage board** et
définit les **flags de build** (le nibble UART est saturé → sélection par compilation, précédent `-DMAHA_INT8`).

## Spec

### 1. Matrice de portage (renseignée depuis S4708)

| Rôle | Dataset | weight_bits | granularité | symétrie | build flag | packing |
|------|---------|:---:|:---:|:---:|-----------|:---:|
| référence | Monitoring/Pronostia | 8 | per_channel | symmetric | `-DEWC_INT8_V2` (S39) | — |
| frontière | Monitoring | (S4708) | per_channel | (S4708) | `-DEWC_INT4` | packé + non-packé |
| frontière | Pronostia | (S4708) | per_channel | (S4708) | `-DEWC_INT4` | packé + non-packé |
| agressive | Monitoring | (S4708 −1) | per_channel | symmetric | `-DEWC_INT2` | packé + non-packé |
| agressive | Pronostia | (S4708 −1) | per_channel | symmetric | `-DEWC_INT2` | packé + non-packé |

Chaque cellule frontière/agressive est buildée **deux fois** (packé / non-packé) pour objectiver l'écart RAM
théorique↔`.bss`.

### 2. Cadrage des flags de build

| Flag | Effet | `EWC_V2_W_QMAX` |
|------|-------|:---:|
| `-DEWC_INT8_V2` (défaut S39) | poids int8 per-channel | 127 |
| `-DEWC_INT4` | poids 4-bit (conteneur int8) | 7 |
| `-DEWC_INT2` | poids 2-bit (conteneur int8) | 3 |
| `-DEWC_INTx_PACKED` (combiné) | active le stockage bit-packé + dépacking au forward | — |

**Wire format V3 (23 B) inchangé**, `sensor_stream.py` intact (sélection au build, pas au protocole).

### 3. Invariants à préserver

- `.bss` **défaut** (build FP32/INT8_V2 standard) **invariant** — les sub-INT8 sont des variantes de build
  (précédent S2912 : `.bss` défaut inchangé, +60 B sous `-DMAHA_INT8`).
- 0 régression sur les builds existants (`make test`).

## Contraintes

- Aucun chiffre de résultat (placeholders `(S4708)`) tant que S47 n'a pas tourné.
- Sélection **traçable** aux JSON `exp_S47_depth/`/`exp_S47_symmetry/`.

## Vérification

```bash
grep -i "EWC_INT4\|EWC_INT2\|PACKED\|per_channel" docs/sprints/sprint_48/S4801_selection_cadrage.md
```

---

## Résolution (implémentée)

_À compléter lors de l'implémentation._
