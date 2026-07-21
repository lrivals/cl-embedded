# S4805 — Parité board↔PC + agrégation `exp_S48_summary.json`

| Champ | Valeur |
|-------|--------|
| **Sprint** | 48 |
| **Priorité** | 🟠 Importante — consolide les cellules en une vue unique bits × plateforme. |
| **Statut** | 📝 Doc — spec complète ; implémentation à venir |
| **Durée estimée** | 3h |
| **Dépendances** | S4804 (cellules board) · S47 (émulateur = réplique PC) |
| **Fichiers cibles** | `scripts/board_pc_parity48.py`, `scripts/aggregate_sprint48.py`, `experiments/exp_S48_summary.json` |
| **Références** | patrons `board_pc_parity45.py`/`aggregate_sprint45.py` (mesuré-board vs proxy-PC) |

---

## Contexte

Deux livrables de consolidation : (1) une table de parité **par échantillon** board↔PC (le PC = émulateur S47
rejoué sur l'ordre board, précédent `board_pc_parity45.py`), (2) une agrégation indexée qui met en regard la
**RAM `.bss` mesurée** et la **RAM théorique bit-packée** du Sprint 47.

## Spec

### 1. `board_pc_parity48.py`

Par cellule S4804 : rejoue l'émulateur `subint8` (S47) sur la **même séquence** que la board, produit
`experiments/exp_S48_parity_<ds>_<bits>_<gran>[_packed].json` :
- table `[idx, true, pred_board, pred_pc, score_board, score_pc, match]`,
- `parity_pred` (taux d'accord), `mismatches` (indices), `max_score_err`.

Attendu : **parité exacte 1.000** (schéma bit-identique émulateur/kernel), comme S34/S45.

### 2. `aggregate_sprint48.py`

Lit `exp_S48_board/` + `exp_S47_depth/` → `experiments/exp_S48_summary.json` indexé
`[dataset][weight_bits][granularity][platform]` :
- **board** (mesuré) : `auroc_board`, `latency_dwt_p50/p99_us`, `bss_bytes` (packé/non-packé), `ram_ratio_measured`, `crc_errors`, `gap2_ok`.
- **pc** (S47) : `auroc_quant`, `ram_ratio_theoretical`.
- **deltas** : `auroc_board − auroc_pc`, `ram_ratio_measured_packed − ram_ratio_theoretical` (**écart
  théorie↔matériel = résultat clé du sprint**), sans conflater `bss_bytes` packé/non-packé.

Lecture seule (n'exécute rien) ; `null`/`na_reason` propagés honnêtement.

## Contraintes

- Le PC de référence est **l'émulateur** (parité par construction) — ne pas ré-entraîner.
- Deltas explicites, jamais de conflation packé/non-packé ; N/A propagé.
- `pending`/`null` avant que S4804 ait produit les cellules.

## Vérification

```bash
python scripts/board_pc_parity48.py --cell monitoring_int4_perchannel_packed
python scripts/aggregate_sprint48.py
python -c "import json; d=json.load(open('experiments/exp_S48_summary.json')); assert 'monitoring' in d"
```

---

## Résolution (implémentée)

_À compléter lors de l'implémentation._
