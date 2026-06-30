# S3308 — Notebook de synthèse énergie/coût

| Champ | Valeur |
|-------|--------|
| **Sprint** | 33 |
| **Priorité** | 🟡 Important |
| **Statut** | ✅ Implémenté |
| **Durée estimée** | 3h |
| **Dépendances** | S3301-S3303 (coût) · S3305-S3307 (énergie/autonomie) |
| **Fichiers cibles** | `notebooks/cl_eval/energy_cost/comparison.ipynb` · `notebooks/sprint33_energy_cost.ipynb` (synthèse racine) |
| **Références** | `notebooks/cl_eval/cwru_by_severity/comparison.ipynb` (pattern notebook existant), `notebooks/sprint30_pairs_disagreement.ipynb`, `src/evaluation/plots.py`, `src/evaluation/compute_cost.py` |

---

## Contexte

Synthèse visuelle finale du sprint : relier les métriques de coût matériel-agnostiques
(FLOPs/BOPs/FLOPS-W) aux mesures énergie réelles (µJ/phase, LPM01A) et à l'autonomie, en
répondant explicitement à la question du Gap 3 : l'INT8 réduit-il l'énergie même sans
accélérer la latence FPU (constat Sprint 29) ?

---

## Spec

Sections attendues (réutiliser les patterns de visualisation de
`notebooks/cl_eval/*/comparison.ipynb` existants, pas de nouvelle lib de viz hors
`src/evaluation/plots.py`/`eda_plots.py`) :

1. **Chargement** des JSON `experiments/exp_S33_energy/*.json` + sorties
   `compute_cost.py`/`hw_cost_model.py` (S3301-S3302).
2. **µJ par phase** par modèle, barres groupées FP32 vs INT8 (4 modèles × 4 phases).
3. **Table FP32 vs INT8 énergie** : delta_uj, ratio, et delta_metric (réutilise les champs
   de `benchmark_int8_fp32.py` Sprint 28 pour la cohérence métrique).
4. **Pareto énergie/AUROC** : scatter (µJ total, AUROC) par modèle×encodage.
5. **FLOPS/W** par modèle (issu de `hw_cost_model.flops_per_watt`).
6. **Courbes autonomie vs capacité batterie** (issu de `autonomy.sweep_capacities`).
7. **Synthèse écrite** (cellule markdown) répondant explicitement : *l'INT8 réduit-il
   l'énergie sans accélérer la latence FPU sur Cortex-M4 ?* — avec les chiffres réels, pas
   de conclusion anticipée avant exécution.

**Règles** :
- Notebook **exécutable de bout en bout**, déplacé directement dans `notebooks/` (jamais
  créé ailleurs puis déplacé — règle CLAUDE.md).
- Tout chiffre affiché provient d'une exécution réelle des scripts S3301-S3307 ; si une
  donnée board n'a pas encore été capturée, la cellule l'affiche comme « à mesurer »
  plutôt que de fabriquer une valeur.

---

## Réalisation

Deux notebooks couvrent l'objectif :

- `notebooks/cl_eval/energy_cost/comparison.ipynb` — synthèse thématique dans l'arborescence `cl_eval/`.
- `notebooks/sprint33_energy_cost.ipynb` — **notebook de synthèse à la racine** (convention `sprintXX_*`,
  alignée sur `sprint30_pairs_disagreement.ipynb`), 7 cellules code / 4 figures, exécuté de bout en bout
  sans erreur. Contenu :
  - **Coûts réels** (tracés avec vraies valeurs) : FLOPs / BOPs / paramètres via
    `src/evaluation/compute_cost.py` (ratio **BOPs FP32/INT8 = 16**, gain Gap 3 quantitatif) ; latence
    board inférence vs inférence+update (`exp_S33_board_latency/latency_summary.json`, toutes ≪ Gap 2) ;
    RAM peak + accuracy PC (`exp_S33_PC_*`).
  - **Énergie / autonomie** : matrices d'état explicitant les champs **`"à mesurer"`** (compteur des
    cellules en attente, heatmap « à mesurer »), aucun chiffre énergétique fabriqué ; seul `ram_peak_bytes`
    = 208 B du calcul `autonomy.py` (proxy tracemalloc PC) est tracé comme valeur réelle.
  - **Conclusion Gap 3** : l'INT8 réduit la RAM sans accélérer la latence (FPU Cortex-M4, pas de NPU
    INT8) → question µJ ouverte tant que la sonde LPM01A n'est pas posée.

## Vérification

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/sprint33_energy_cost.ipynb
jupyter nbconvert --to notebook --execute notebooks/cl_eval/energy_cost/comparison.ipynb \
    --output comparison.ipynb
```
