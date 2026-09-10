# S4014 — Frontière de profondeur, portée de l'INT8 en ligne, latence d'apprentissage

| Champ | Valeur |
|-------|--------|
| **Sprint** | 40 (correctifs d'exactitude) |
| **Statut** | ✅ Implémenté — 3 corrections, **aucune nouvelle mesure** |
| **Source de vérité** | `experiments/exp_S40_article_metrics/summary.json` (blocs `depth_board`, `recovery_board`) |
| **Régénération** | `python scripts/generate_article_tables.py` puis `make -C docs/article/ewc_int8_mcu all` |

Trois affirmations de l'article étaient soit contredites, soit incomplètes au regard de données
**déjà présentes dans le dépôt**. Aucune campagne n'a été relancée : les trois correctifs consistent
à faire dire au texte ce que les mesures disent déjà.

## 1. La frontière de profondeur dépend de la métrique de sélection

`05b_quant_axes.tex` concluait « la frontière exploitable est le ternaire » sur le seul $\Delta$AUROC
**émulé** (S47). Le bloc `depth_board` de l'agrégat porte pourtant le **F1 mesuré sur carte** (S48) des
mêmes schémas, et il raconte l'inverse :

| Jeu | schéma | AUROC carte | F1 carte |
|---|---|---|---|
| Pronostia | ternaire | 0,99907 | **0,3790** |
| Pronostia | binaire | 0,96616 | **0,0000** |
| Monitoring | binaire | 0,96784 | 0,6878 |

L'ordre des scores tient là où le seuil de décision ne tient plus : c'est le motif « accuracy trompeuse,
passer au F1 » déjà établi au Sprint 35. En F1, le seul schéma sub-INT8 qui préserve la métrique sur les
deux jeux est l'**INT4**.

**Correctif.** Les deux lectures sont rapportées côte à côte plutôt qu'arbitrées : le résultat devient
*ce n'est pas la profondeur seule qui fixe la frontière, c'est le couple profondeur–métrique*. Une
nouvelle table **générée depuis l'agrégat** (`build_depth_board` → `tables/depth_board_{fr,en}.tex`,
label `tab:depth-board`) porte les chiffres ; le texte, l'introduction, la conclusion, l'annexe de
profondeur et la légende de la figure 7 sont alignés en FR et EN.

## 2. Le régime « en ligne » INT8 n'apprend pas en INT8

`ewc_head_int8_v2.h` n'expose qu'une passe avant, et `pipeline.c` applique le pas de gradient à la tête
**FP32 maîtresse** avant de reconstruire la vue entière. Les cellules « en ligne » du tableau de
récupération étaient donc lues comme une démonstration du Gap 3 « quantification pendant l'apprentissage
incrémental », ce qu'elles ne sont pas.

**Correctif.** Un paragraphe de portée en `03_method` (FR+EN) énonce que la division par quatre vaut pour
l'inférence seule, la carte portant en régime d'apprentissage la tête flottante, sa vue entière,
l'importance et l'ancrage. La conclusion parle désormais d'une quantification *compatible* avec
l'apprentissage incrémental, et non *conduite* en arithmétique entière, qui reste un problème ouvert.

## 3. Le surcoût de latence est bien plus lourd en apprentissage qu'en inférence

La discussion ne chiffrait que le mode gelé (74 µs contre 48–50 µs). Le tableau de récupération donne,
sur les mêmes cellules, **577 et 606 µs en ligne contre 239 et 251 en FP32**, soit un facteur 2,4 causé
par la requantification intégrale des matrices à chaque échantillon.

**Correctif.** Un paragraphe dédié dans `05_results`, un rappel dans la discussion et dans la conclusion,
et deux pistes ajoutées aux travaux futurs : requantification restreinte aux lignes modifiées par le pas,
et descente de gradient dans le domaine entier.

## Corrections de forme jointes

* `tab:parity` déclarait 6 colonnes pour 5 entêtes (colonne fantôme) → `lcccc` ;
* l'entête « F1 (PC/carte) » ne disait pas la plateforme → « F1 carte », avec la note expliquant que
  Monitoring n'a que 4 features natives et ne donne donc qu'une ligne ;
* l'introduction citait deux des trois cellules de parité en ligne → intervalle 0,9626 à 0,9887 ;
* `05c_system` : précision que la « RAM des poids » compte les matrices seules (672 et 704 poids), les
  biais restant flottants, d'où l'écart avec les 722 et 754 paramètres de la tête. **Les deux chiffres
  étaient justes** ; seule la lecture était ambiguë.

## Vérification

* `python scripts/generate_article_tables.py` → 12 tables (10 + les 2 nouvelles) ;
* `make -C docs/article/ewc_int8_mcu all` → **0 erreur LaTeX, 0 lien de glossaire cassé, 0 référence
  indéfinie** (FR 32 p., EN 31 p.) ;
* `pytest tests/test_sprint40_article.py tests/test_article_metrics.py tests/test_figures_library.py`
  → **43 PASS / 2 skips banc** (fichiers de parité S40 non produits, registre S4010) ;
* deux gardes ajoutées : `test_depth_reports_both_metrics` (la lecture F1 mesurée carte ne peut plus
  disparaître de l'axe profondeur) et `test_online_int8_scope_is_stated` (la réserve sur la tête FP32
  maîtresse ne peut plus être retirée) ;
* **firmware non touché** (correctifs documentaires) : `.bss` défaut invariant, `make test` inchangé.
