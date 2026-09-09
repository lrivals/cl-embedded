# S4008 — Coût de calcul EWC et agrégat unique des métriques de l'article

| Champ | Valeur |
|-------|--------|
| **Sprint** | 40 (refonte) |
| **Priorité** | 🔴 Critique — chaînon manquant : les métriques EWC vivaient dans dix répertoires disjoints |
| **Statut** | ✅ Implémenté |
| **Plateforme** | PC uniquement (lecture seule ; **firmware non modifié**) |
| **Fichiers** | `scripts/measure_macs.py` (`--head`) · `scripts/aggregate_article_ewc.py` → `experiments/exp_S40_article_metrics/` |
| **Tests** | `tests/test_article_metrics.py` (9 PASS) |

## Contexte

L'article devait citer des chiffres venus de S36, S39, S40, S46, S47, S48, S49, S50, S53 et S54 — soit
dix répertoires, chacun avec son schéma. Sans agrégat, chaque valeur du `.tex` devait être retrouvée à la
main : c'est exactement la situation qui avait laissé passer une ligne « à mesurer » alors que la mesure
existait. Par ailleurs **aucun JSON du dépôt ne portait de coût de calcul absolu** : `compute_cost.py`
savait produire MACs/FLOPs/BOPs et `measure_macs.py` les calculait, mais uniquement vers `stdout`.

## Coût de calcul — un écart d'architecture corrigé au passage

`measure_macs.py --model ewc` construisait `EWCMlpClassifier`, dont la sortie est **binaire** (`fc3 → 1`),
et renvoyait 656 MACs. Or la tête réellement portée sur la carte est `EWCMlpMulticlass`, `k→32→16→2`.
Publier 656 MACs à côté de chiffres board aurait décrit une autre architecture que celle mesurée.

Correctif minimal et additif : une option **`--head {binary,multiclass}`** (défaut `binary`, comportement
historique strictement inchangé — vérifié : 656 MACs). Avec `--head multiclass`, l'outil produit 672 MACs
(Monitoring, k=4) et 704 (Pronostia, k=5), et torchinfo compte **722 / 754 paramètres**.

Ce compte est ensuite **vérifié** par l'agrégat contre `exp_S39_quant_sweep` (`n_params_matches_s39`) :
c'est le garde-fou qui empêche de décrire une architecture différente de celle qui a été quantifiée et
flashée. BOPs FP32/INT8 = 16, soit $(32/8)^2$ — un ratio **théorique**, que la latence mesurée contredit.

## Agrégat `exp_S40_article_metrics/summary.json`

Lecture seule, sur le patron de `aggregate_sprint48.py` / `aggregate_ram.py`. Indexé
`[dataset][axe][cellule]` sur les deux jeux de l'article, 9 axes, **238 cellules par jeu**.

Chaque cellule porte `value`, `source_json`, `platform` ∈ {`mesuré board`, `émulé PC`, `mesuré PC`,
`théorique`} et `na_reason`. Une mesure absente vaut `null` ou le sentinel `"à mesurer"` — jamais 0.

Trois choix de structure méritent d'être justifiés :

* **`energy.estimators` sépare les méthodes** (régression de cadence, delta/WFI, lot `INFER_BATCH_N`,
  balayage de fréquence, énergie par MAJ), chacune avec sa `method_note` recopiée de la source. Les JSON
  S53 portent eux-mêmes l'interdiction de fusionner : leur désaccord — 5,08 · 67,65 · 146,77 µJ — dit ce
  que chaque méthode inclut, et une moyenne détruirait cette information. Un test l'impose.
* **L'oubli (S54) va dans `context`**, pas dans les axes : il est mesuré sur **CWRU**, qui ne recouvre
  aucun des deux jeux de l'article. Le mélanger aurait produit une comparaison fausse.
* **`missing`** énumère les chemins réellement non mesurés (28 à ce jour) et alimente `S4010`.

## Vérification

```bash
python scripts/measure_macs.py --model ewc --config configs/board_ewc.yaml --n-in 4 --head multiclass \
       --out experiments/exp_S40_article_metrics/compute_cost_ewc_monitoring.json
python scripts/aggregate_article_ewc.py
pytest tests/test_article_metrics.py -q
```
