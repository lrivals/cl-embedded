# S4012 — Formalisation d'EWC en section 3 de l'article

**Sprint** : 40 (article standalone « EWC INT8 sur MCU »)
**Périmètre** : rédaction FR+EN, PC-only. Aucun code, aucun firmware, aucune mesure nouvelle.
**Statut** : ✅ implémenté

## Motivation

L'article mesurait la quantification d'une tête EWC sans jamais définir EWC. La méthode s'en tenait
à une phrase, « entraîné en régime incrémental avec pénalité de Fisher en ligne », et l'article ne
contenait **aucune équation** : aucun environnement `equation`/`align` dans les neuf sections.

Deux manques en découlaient. Un lecteur ne pouvait pas reconstruire le modèle dont l'article mesure
la parité, la RAM, la latence et l'énergie. Et surtout, les **adaptations du modèle aux contraintes
embarquées** — qui sont le cœur du portage — restaient invisibles, alors que la carte n'estime pas
l'information de Fisher : elle utilise le carré du poids comme substitut d'importance.

## Contenu ajouté

Nouvelle sous-section 3.1 « Formulation EWC et adaptation embarquée » (`sec:ewc_formulation`),
placée avant « Modèle EWC et portage C », qui renvoie désormais à elle au lieu de répéter la phrase
sur la pénalité. Quatre équations, les premières numérotées de l'article :

| Label | Contenu | Source de vérité |
|---|---|---|
| `eq:ewc_loss` | perte de tâche + `(λ/2) Σ F_i (θ_i − θ*_i)²` | `src/models/ewc/ewc_mlp.py:180-194`, `ewc_mlp_multiclass.py:103-115` |
| `eq:fisher` | Fisher diagonal empirique, carré du gradient de la perte **moyennée sur un lot**, moyenné sur les lots | `src/models/ewc/fisher.py:70-101` |
| `eq:board_sgd` | pas embarqué à un échantillon, gradient de tâche + rappel élastique, η = 0.01 | `firmware/stm32f4_blink/src/ewc_head.c:161-204` |
| `eq:board_fisher` | consolidation embarquée `F ← α F + (1−α) W²`, `W* ← W`, α = 0.99 | `firmware/stm32f4_blink/src/ewc_head.c:212-242` |

Plus le tableau `tab:ewc_port` (PC / carte) : estimateur d'importance, données requises,
accumulation, portée de la pénalité, régime d'optimisation, λ, déclenchement, état initial de `F`.

## Écarts PC ↔ carte formalisés, et pourquoi ils sont assumés

La section les nomme au lieu de les lisser, conformément à la règle d'honnêteté du sprint (S4000) :

1. **Le Fisher embarqué n'en est pas un.** `ewc_consolidate` n'exécute ni forward, ni backward, et
   ne consomme aucune donnée étiquetée : l'importance est approximée par `W²`. La justification est
   la contrainte elle-même — une carte en flux continu ne conserve pas de lot à repasser. Le texte
   dit explicitement que c'est un substitut de magnitude, et sous quelle hypothèse il tient.
2. **La pénalité embarquée ne couvre pas les biais** (ni `fisher_b`, ni `star_b` dans `EWCHead`), ce
   qui économise un tiers de l'état de régularisation.
3. **`F ≡ 0` à l'initialisation** : tant qu'aucune consolidation n'a eu lieu, la carte fait de la
   descente de gradient pure. La régularisation ne démarre qu'à la première consolidation.
4. **Régime d'optimisation** : un échantillon par trame, sans momentum, une seule passe, contre des
   lots avec momentum sur plusieurs époques côté PC.
5. **λ = 1000 (`configs/ewc_config.yaml`) contre 400 sur carte** (`configs/board_ewc.yaml`,
   `pipeline.c:593`). Les deux valeurs coexistent ; le tableau les affiche côte à côte.
6. **Rétropropagation en place** : la carte propage l'erreur avec le poids **déjà mis à jour**
   (`ewc_head.c:165-166, 185-186`), le cadre PC utilise partout les poids d'avant le pas. Cet écart
   est rattaché explicitement aux mesures de parité : exacte en gelé, approchée en ligne.

Nuance rédactionnelle importante : la règle `F ← γ F + F_new` (γ = 0.9, `fisher.py:138-141`) est
attribuée à la variante en ligne du projet, **pas** à la tête effectivement portée. `EWCMlpMulticlass`
**remplace** son Fisher à chaque `consolidate()` (`ewc_mlp_multiclass.py:165-188`).

## Ajout bibliographique

`Schwarz2018Online` (Progress & Compress, ICML 2018) ajouté à `docs/article/ewc_int8_mcu/references.bib`,
pour légitimer l'entretien d'une importance unique accumulée plutôt qu'une pénalité par tâche.
**Écart assumé** vis-à-vis de la liste fermée de clés fixée en S4004 §« Bibliographie » : la clé est
citée dans les deux langues et la bibliographie passe de 9 à 10 entrées. La référence était déjà
mentionnée dans les docstrings du projet (`src/models/ewc/fisher.py`), elle manquait seulement à la
bibliographie autonome de l'article.

## Contraintes respectées

- **Miroir strict FR/EN** : mêmes équations, mêmes labels, même tableau, mêmes littéraux décimaux
  (`test_fr_en_key_values` compare l'ensemble des décimales de `main_*.tex` + `sections/<lang>/*.tex`).
  Les largeurs de colonnes sont écrites en `mm` pour ne pas polluer cet ensemble.
- **Aucun chiffre de résultat introduit.** Les seules valeurs sont des hyperparamètres traçables à
  un YAML ou à un `#define` (0.01, 0.99, 0.9, 200, 400, 1000) — pas des grandeurs mesurées, donc hors
  du périmètre de `test_figures_match_json`.
- `amsmath` suffit ; `amssymb` n'est pas chargé et n'a pas été requis. Aucune unité ajoutée au bloc
  de repli `siunitx`.

## Vérification

```
make -C docs/article/ewc_int8_mcu clean && make -C docs/article/ewc_int8_mcu all
pytest tests/test_sprint40_article.py tests/test_article_metrics.py tests/test_figures_library.py -q
```

Résultats : compilation **0 erreur LaTeX, 0 lien de glossaire cassé, 0 référence indéfinie**,
`main_fr.pdf` 27 p. / `main_en.pdf` 26 p., 10 bibitems dans chaque `.bbl` (clé `Schwarz2018Online`
présente dans les deux). Tests : **24 passés / 2 skips de banc préexistants** (article + agrégat) et
**17 passés** (bibliothèque de figures). Le tableau `tab:ewc_port` ne déborde plus de la
justification (débordement de 138 pt corrigé par des colonnes `p{}` et `\footnotesize`).

Firmware et code Python **non touchés** : `make test` et le `.bss` par défaut restent hors périmètre.
