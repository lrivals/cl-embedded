# S3807 — Agrégat & table d'économie (livrable central)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 38 |
| **Priorité** | 🔴 Critique — c'est le livrable qui répond chiffré à la question « économie vs précision ». |
| **Statut** | ✅ Implémenté — `scripts/aggregate_sprint38.py` → `exp_S38_summary.json` (+ `economy_table`). |
| **Durée estimée** | 3h |
| **Dépendances** | S3802 ✅ (PC) · S3804/S3805 ✅ (board) · S3806 ✅ (parité) · `scripts/aggregate_sprint36.py` ✅ (modèle) |
| **Fichiers cibles** | `scripts/aggregate_sprint38.py`, `experiments/exp_S38_summary.json` |
| **Références** | S3606 (agrégat Sprint 36 comme modèle — lecture seule) |

---

## Contexte

Fusionne les sorties dispersées (PC S3802, board S3804/S3805, parité S3806) en **un seul** JSON indexé
`[dataset][policy][platform]` avec `platform ∈ {pc, board}`. **Lecture seule** : aucune métrique
recalculée, on reprend les valeurs déjà stockées (à l'image des `exp_S3{2,5,6}_*_summary.json`).
Les champs absents → `null`.

## Spec

Champs par cellule `[dataset][policy][platform]` :
- Précision : `acc_final`, `f1_faulty`, `af` (oubli).
- **Économie** : `n_updates`, `update_rate`, `mean_latency_us`, `inference_latency_us`, `gate_overhead_us`,
  `bss_bytes`, `bss_delta_vs_default`.
- Parité : `prediction_parity_rate`, `verdict_parity_rate`.
- Gap : `gap2_ok`.

**Bloc `economy_table`** (le livrable) : pour chaque dataset, deltas **vs `always`** (P1) :
- `latency_saved_us` = `always.mean_latency_us − policy.mean_latency_us` (gain).
- `ram_added_bytes` = `policy.bss_delta_vs_default` (coût du gate).
- `f1_lost` = `always.f1_faulty − policy.f1_faulty` (précision perdue).
- `updates_saved_pct` = `1 − policy.update_rate / always.update_rate`.

→ permet de lire directement : « P3 économise X µs et Y % de MAJ vs P1, au coût de Z B de RAM et ΔF1 ».

## Vérification

```bash
python scripts/aggregate_sprint38.py     # → experiments/exp_S38_summary.json
```
- `economy_table` cohérent : `frozen` (0 update, F1 plancher), `always` (référence), `gated_*` entre les deux.
- Aucune valeur recalculée ; champs manquants = `null` (pas de chiffre inventé).
