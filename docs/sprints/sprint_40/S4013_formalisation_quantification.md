# S4013 — Formalisation de la quantification en section 3 de l'article

**Sprint** : 40 (article standalone « EWC INT8 sur MCU »)
**Périmètre** : rédaction FR+EN + bibliographie, PC-only. Aucun code, aucun firmware, aucune mesure nouvelle.
**Statut** : ✅ implémenté

## Motivation

Symétrique de [S4012](S4012_formalisation_ewc.md). L'article mesurait trois axes de quantification
(format, moment, profondeur) sans jamais écrire ce qu'est une quantification. L'« axe du format » se
réduisait à une liste à puces de cinq noms de noyaux, sans une seule échelle, sans un seul symbole
`s`/`q`/`z`, et **sans aucune référence de quantification dans la bibliographie** — un article dont le
sujet est la quantification INT8 ne citait ni Jacob, ni Krishnamoorthi, ni les schémas ternaire et
binaire qu'il évalue pourtant.

Conséquence directe : la section 5b renvoyait à la section 3 pour la définition des schémas
(« Les trois placements comparés ici et les paramètres d'un schéma sub-INT8 sont définis en \S3 »),
renvoi que la section 3 n'honorait pas.

## Contenu ajouté

Nouvelle sous-section « Formulation de la quantification » (`sec:quant_formulation`), qui absorbe et
remplace l'ancienne « Axe du format », plus une extension de « Axe de la profondeur ». Six équations :

| Label | Contenu | Source de vérité |
|---|---|---|
| `eq:quant_map` | mapping symétrique signé, `Q = clip(round(W/s), −qmax, qmax)`, `qmax = 2^(b−1)−1` | `int8_c_emulation.py:219-224`, `ewc_head_int8_v2.c:40-44` |
| `eq:scale_gran` | les trois granularités comme **une seule famille** : `1/128` figée, `max|W|/qmax`, `max_i|W[j,i]|/qmax` | `int8_c_emulation.py:205-216`, `ewc_head_int8_v2.c:51-57` |
| `eq:int_layer` | couche entière `acc_j = Σ_i Q_ji q^a_i`, `y_j = acc_j·s_j·s_a + b_j` | `int8_c_emulation.py:361-368`, `ewc_head_int8_v2.c:123-133` |
| `eq:twn` | ternaire, seuil `Δ_j = 0.7·mean|W[j,:]|`, scale = moyenne des poids retenus | `int8_c_emulation.py:226-243` |
| (binaire, en ligne) | `Q = sign(W)`, `α_j = mean|W[j,:]|` | `int8_c_emulation.py:246-255` |
| `eq:affine_act` | activation affine `q = round(a/s)+z`, déquant `(q−z)·s` | `int8_c_emulation.py:347-355`, `src/utils/quantization.py:51-58` |

Plus le tableau `tab:quant_ref` (schéma de référence vs noyau v2 embarqué) et deux paragraphes neufs :
le **fonctionnement en profondeur du par-canal** (l'indice `j` porté par `s_j` dans `eq:int_layer` ;
stockage en tableau de flottants parallèle à chaque matrice ; coût qui ramène le gain mémoire réel des
poids sous le facteur quatre annoncé par le seul changement de type) et l'**échelle d'ablation
reformulée comme une trajectoire** dans la famille `eq:scale_gran` plutôt qu'une liste de noyaux.

## Provenance du schéma, et écarts assumés

Le schéma entier per-canal vient de la formulation de référence `Jacob2018` / `Krishnamoorthi2018`.
Le tableau `tab:quant_ref` nomme sept écarts, tous dictés par la cible et non par une simplification
de commodité :

1. **Retour d'échelle** — la référence requantifie par un multiplicateur entier et un décalage ; le
   noyau v2 déquantifie en flottant sur l'unité matérielle du Cortex-M4F. C'est l'origine directe du
   **paradoxe de latence** documenté en 5c (la requantification pèse un tiers du budget).
2. **Biais** — `int32` à l'échelle du produit dans la référence, flottants exacts jamais quantifiés
   chez nous (`ewc_head_int8_v2.c:73`, `int8_c_emulation.py:366`).
3. **Zéro-point** — la référence en pose un sur les activations et les poids ; nos poids restent
   symétriques sans zéro-point, les activations sont symétriques par défaut et affines en option.
4. **Granularité** — per-tensor dans `Jacob2018`, per-canal en extension ; per-canal de sortie par
   défaut chez nous, c'est le correctif qui récupère la métrique.
5. **Calibration** — histogramme ou centile dans la référence, **maximum absolu par couche sur un
   lot** chez nous (`calibrate_activations`, `int8_c_emulation.py:379-387`).
6. **Fusion de normalisation** — sans objet, le réseau ne contient pas de couche de normalisation.
7. **Accumulateur** — `int32`, élargi en `int64` pour la variante Q15 (`ewc_head_int8_v2.h:35`).

Ternaire et binaire sont repris **tels quels** de `Li2016TWN` et `Rastegari2016XNOR`, attributions qui
vivaient jusqu'ici dans les docstrings du code (`int8_c_emulation.py:229`, `:249`) et dans aucune
bibliographie du dépôt. Le texte note qu'ils portent une échelle par canal par construction, ce qui
explique pourquoi l'axe de la granularité cesse de s'appliquer une fois qu'on les atteint (constat
mesuré S4703).

## Correctif factuel

L'ancienne liste décrivait `legacy_c` comme un « accumulateur `int8` ». C'est faux : l'accumulateur du
noyau v1 est un **`int16` qui reboucle silencieusement** (`ewc_head_int8.c:89`, émulé par `_wrap_int16`,
`int8_c_emulation.py:50`). Le texte énumère désormais les **quatre** écarts cumulés du noyau d'origine
— échelle figée, troncature au lieu d'arrondi, accumulateur `int16`, activations repliées sur une
grille Q7 — au lieu d'un seul, mal nommé.

## Bibliographie

Cinq entrées ajoutées à `docs/article/ewc_int8_mcu/references.bib` (9 → 15 clés avec `Schwarz2018Online`
de S4012) :

| Clé | Rôle | Cité en |
|---|---|---|
| `Jacob2018` | schéma entier de référence | 2 (PTQ/QAT), 3.4 (formulation, `tab:quant_ref`) |
| `Krishnamoorthi2018` | livre blanc PTQ/QAT, per-canal | 2, 3.4 |
| `Li2016TWN` | ternaire | 3.6 (`eq:twn`) |
| `Rastegari2016XNOR` | binaire | 3.6 |
| `Lai2018CMSISNN` | noyaux entiers SIMD Cortex-M | 6 (piste du paradoxe de latence) |

Correction au passage : l'entrée `Ravaglia2021QLRCL` portait un **journal erroné** (« IEEE
Transactions on Circuits and Systems for Video Technology »). Elle est alignée sur celle du manuscrit,
IEEE JETCAS 11(4):789-802, avec DOI.

## Contraintes respectées

- **Miroir strict FR/EN** : mêmes équations, mêmes labels, même tableau, mêmes littéraux décimaux
  (`0.7` du seuil ternaire est présent des deux côtés).
- **Aucun chiffre de résultat introduit.** Les seules constantes sont des paramètres de schéma
  (`0.7`, `127`, `128`, `32767`), tous traçables à une ligne de code.
- `amsmath` suffit (`gathered` pour l'équation ternaire, coupée en deux lignes pour ne pas déborder).

## Vérification

```
make -C docs/article/ewc_int8_mcu clean && make -C docs/article/ewc_int8_mcu all
pytest tests/test_sprint40_article.py tests/test_article_metrics.py tests/test_figures_library.py -q
```

Résultats : **0 erreur LaTeX, 0 lien de glossaire cassé, 0 référence indéfinie**, `main_fr.pdf` 30 p. /
`main_en.pdf` 29 p., **15 bibitems** dans chaque `.bbl`. Tests : **41 passés / 2 skips de banc
préexistants**. Aucun débordement de justification introduit dans la section 3.

Firmware et code Python **non touchés** : `make test` et le `.bss` par défaut restent hors périmètre.
