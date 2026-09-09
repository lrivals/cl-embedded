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

**Gagnants réels S4708** (`docs/sprints/sprint_47/S4708_pointeur_board.md`, tracés `exp_S47_depth/`) :
frontière = **ternaire**, agressive = **binaire** (et non « INT4/INT2 linéaire » comme le supposait le gabarit
pré-S47 de cette doc). Décision utilisateur : porter la famille **générale** — linéaire INT4/INT2 (comparaison)
**et** ternaire/binaire (gagnants). Le firmware est **agnostique au mode** : il consomme les entiers
pré-quantifiés côté PC + les scales par-canal ; c'est l'export qui choisit le schéma.

| Rôle | Dataset | mode / bits eff. | granularité | symétrie | build flag | Δauroc (émul.) | packing |
|------|---------|:---:|:---:|:---:|-----------|:---:|:---:|
| référence | Monitoring/Pronostia | linéaire int8 | per_channel | symmetric | `-DEWC_INT8_V2` (S39) | ≈0 (×4) | — |
| comparaison | Monitoring/Pronostia | linéaire 4 bits | per_channel | symmetric | `-DEWC_INT4` | Mon −0.0005 / Pro −0.0016 | packé + non-packé |
| **frontière** | Monitoring | **ternaire (2 bits)** | per_channel | symmetric | `-DEWC_INT2` | **−0.0021** (×20.25) | packé + non-packé |
| **frontière** | Pronostia | **ternaire (2 bits)** | per_channel | symmetric | `-DEWC_INT2` | **−0.0153** (×20.25) | packé + non-packé |
| **agressive** | Monitoring | **binaire (1 bit)** | per_channel | symmetric | `-DEWC_INT1` | **−0.0117** (×32) | packé + non-packé |
| **agressive** | Pronostia | **binaire (1 bit)** | per_channel | symmetric | `-DEWC_INT1` | **−0.0275** (casse −0.02, ×32) | packé + non-packé |

Chaque cellule frontière/agressive est buildée **deux fois** (packé / non-packé) pour objectiver l'écart RAM
théorique↔`.bss` (nœud d'honnêteté S4800). Un schéma est portable via `-DEWC_INT2` que ses poids soient
linéaires-int2 ou ternaires (même conteneur 2 bits) — c'est l'export PC qui produit les entiers.

### 2. Cadrage des flags de build

| Flag | Mode(s) portés | conteneur | `EWC_V2_W_QMAX` | `EWC_V2_PACK_BITS` |
|------|-------|:---:|:---:|:---:|
| `-DEWC_INT8_V2` (défaut S39) | linéaire int8 per-channel | int8 | 127 | — |
| `-DEWC_INT4` | linéaire 4 bits | int8 | 7 | 4 (2 poids/octet) |
| `-DEWC_INT2` | linéaire 2 bits **ou ternaire {−1,0,+1}** | int8 | 1 | 2 (4 poids/octet) |
| `-DEWC_INT1` | binaire {−1,+1} | int8 | 1 | 1 (8 poids/octet) |
| `-DEWC_INTx_PACKED` (combiné) | active le stockage bit-packé + dépacking au forward | uint8 packé | — | — |

**Correctif doc** : le gabarit indiquait `-DEWC_INT2 → QMAX 3` ; l'émulateur calcule
`QMAX = (1<<(bits−1))−1`, donc **INT2 = QMAX 1** (parité bit-exacte non négociable). **Ajout `-DEWC_INT1`**
(binaire) : le gagnant agressif est binaire (1 bit) → seul `-DEWC_INT1` matérialise le ÷32 réel (`-DEWC_INT2`
plafonnerait à ÷16).

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

Matrice de portage figée (ci-dessus), traçable aux JSON `experiments/exp_S47_depth/exp_S47_ewc_{monitoring,pronostia}_{ternaire,binaire}_per_channel.json` (frontière = ternaire, agressive = binaire) et `exp_S47_symmetry/` (affine écarté partout, `symmetric` conservé). **Décision utilisateur** : porter la famille générale (linéaire INT4/INT2 **+** ternaire/binaire), firmware agnostique au mode.

**Cadrage des flags de build implémenté** (`ewc_head_int8_v2.h`, S4802) :

- `-DEWC_INT4` (QMAX 7, `EWC_V2_PACK_BITS 4`), `-DEWC_INT2` (QMAX **1** — correctif du gabarit qui citait 3, `EWC_V2_PACK_BITS 2`), `-DEWC_INT1` (binaire, `EWC_V2_PACK_BITS 1`) ; conteneur `int8_t` en non-packé, `uint8_t` packé sous `-DEWC_INTx_PACKED` ;
- **ajout `-DEWC_INT1`** (non prévu au gabarit) car le gagnant agressif est **binaire (1 bit)** : seul un packing 1 bit matérialise le ÷32 (un `-DEWC_INT2` plafonnerait à ÷16) ;
- **wire format V3 (23 B) inchangé**, `sensor_stream.py` intact — sélection au build (précédents `-DMAHA_INT8` S2912, `-DEWC_INT8_Q15` S39).

**Invariants vérifiés** : `.bss` **défaut invariant = 105 036 B** (toutes les additions sub-INT8 sont gardées `#if defined(EWC_INTx…)` → chemin par défaut byte-identique) ; **0 régression** `make test` (141 tests, 2 échecs TinyOL préexistants hors périmètre, tous les tests sub-INT8 PASS). Aucun chiffre de résultat board écrit (mesures DWT/`.bss` packé board = S4804).
