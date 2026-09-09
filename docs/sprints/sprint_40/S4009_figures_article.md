# S4009 — Catalogue de figures régénérable et synchronisation avec l'article

| Champ | Valeur |
|-------|--------|
| **Sprint** | 40 (refonte) |
| **Priorité** | 🟠 Haute — corrige une dérive de copie qui faisait compiler les PDF sur des figures périmées |
| **Statut** | ✅ Implémenté |
| **Plateforme** | PC uniquement (**firmware non modifié**) |
| **Fichiers** | `src/figures/catalogs/article_ewc.py` · `notebooks/cl_eval/article_ewc_int8/synthesis.ipynb` |
| **Tests** | `tests/test_sprint40_article.py::TestFigureSync` · `tests/test_article_metrics.py` · `tests/test_figures_library.py` |

## Le défaut corrigé

Avant la refonte, le notebook traçait les figures dans `docs/figures/sprint40_article/` et une **copie
manuelle** les portait vers `docs/article/ewc_int8_mcu/figures/`. `md5sum` montrait que `fig2`, `fig4` et
`fig5` avaient divergé : les PDF étaient donc compilés sur des figures plus anciennes que les données.
Deux producteurs pour une même figure suffisent à produire cette dérive ; le correctif est d'en garder un.

## Architecture retenue

`src/figures/catalogs/article_ewc.py`, enregistré `@register_catalog("article_ewc_int8")` (patron
`quant_depth_board.py` / `ram_full.py`). Il lit **une seule source**, l'agrégat S4008, et n'ouvre aucun
JSON de sprint directement : les figures ne peuvent donc pas diverger du texte, qui cite le même agrégat.

Dix figures, la numérotation `figN_` restant celle attendue par l'article et les tests :

| # | Fichier | Contenu | Source |
|---|---------|---------|--------|
| A1 | `fig1_parity_fp32_pc_board` | parité FP32 PC↔carte, gelé / en ligne | S36 |
| A2 | `fig2_latency_gap2` | latences Gap 2 + décomposition du kernel INT8 | S36 + S50 |
| A3 | `fig3_ablation_ladder` | échelle d'ablation `legacy_c` → `q15` | S39 |
| A4 | `fig4_int8_recovery_board` | effondrement puis **récupération mesurée carte** | S36 + S40 |
| A5 | `fig5_pareto_ram_f1_latency` | Pareto RAM × F1, étoile = point flashé | S39 + S40 |
| A6 | `fig6_quant_moment` | moment : avant / après / les deux | S46 |
| A7 | `fig7_quant_depth_packing` | profondeur × granularité, **octets gagnés par le packing** | S47 + S48 |
| A8 | `fig8_ram_total` | RAM totale `.data` + `.bss` + pic, % du budget | S49 |
| A9 | `fig9_energy_estimators` | **3 estimateurs côte à côte** + arbitrage fréquence | S53 |
| A10 | `fig10_compute_cost` | BOPs théoriques contre latence mesurée | S4008 + S50 |

Deux décisions de tracé valent d'être notées, parce qu'elles changent ce que la figure démontre :

* **A7 (droite) trace les octets *économisés* par le packing**, pas le `.bss` total. Sur ~105 Ko de
  `.bss`, un gain de 336 à 604 octets est invisible en valeur absolue : la première version de la figure
  montrait deux barres identiques et ne démontrait rien.
* **A9 garde les trois estimateurs séparés** et n'en dérive aucune moyenne (5,08 · 67,65 · 146,77 µJ) :
  l'écart d'un facteur 10 avec le lot `INFER_BATCH_N` mesure précisément le coût de la trame UART.

## Synchronisation automatique

En fin de `build()`, les PNG sont copiés (`shutil.copy2`) vers `docs/article/ewc_int8_mcu/figures/` **et**
`docs/figures/sprint40_article/`. La copie n'a lieu que si `out_root` est le répertoire canonique : une
génération vers un dossier temporaire (tests, essais) n'écrase jamais les figures publiées. Le `Makefile`
de l'article déclare désormais `figures/*.png` en prérequis, pour qu'un PDF ne reste pas compilé sur une
figure périmée. Un test compare les md5 des trois répertoires.

## Notebook

`synthesis.ipynb` est conservé mais **délègue le tracé** au catalogue : il reste la source de
`provenance_table.csv`, qu'il construit désormais depuis l'agrégat. Une cellule mesurée ne peut donc plus
y rester marquée « à mesurer » — le défaut exact que la refonte corrige. Un test interdit au notebook de
retracer une figure (`plt.subplots` absent) : deux producteurs, c'est la dérive qui revient.

## Vérification

```bash
python scripts/generate_figures.py --catalog article_ewc_int8 --style manuscript
md5sum docs/figures/article_ewc_int8/*.png docs/article/ewc_int8_mcu/figures/*.png \
       docs/figures/sprint40_article/*.png | sort
jupyter nbconvert --execute --to notebook --inplace notebooks/cl_eval/article_ewc_int8/synthesis.ipynb
```
