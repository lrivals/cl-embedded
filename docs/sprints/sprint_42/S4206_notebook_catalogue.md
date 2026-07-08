# S4206 — Notebook catalogue : galerie commentée des figures quantification

| Champ | Valeur |
|-------|--------|
| **Sprint** | 42 |
| **Priorité** | 🟠 Haute |
| **Durée estimée** | ~3h |
| **Statut** | ✅ Implémenté (7 juillet 2026) |
| **Dépendances** | S4203 + S4204 + S4205 (catalogues implémentés) |
| **Fichier cible** | `notebooks/cl_eval/quantization_figures/catalog.ipynb` |

## Objectif

Un notebook-galerie qui **assemble figures et texte** : pour chaque figure des trois catalogues, la figure
générée + un **paragraphe d'explication FR prêt à copier** dans une slide ou un chapitre. C'est le point
d'entrée humain (« je prépare une présentation, où sont les figures quantification et quoi dire dessus ? »).

## Structure

1. **§0 Reproduction** — une cellule unique qui appelle les builders via le registre S4201 (le notebook
   *n'implémente aucune figure*, il consomme `src/figures/` — 0 duplication de code de tracé).
2. **§1 Panorama** — le tableau comparatif de S4202 (importé/reformaté) + figure F5 (pipeline comparatif)
   + I1 (métrique par stratégie) : les 3 artefacts « slide unique » du sujet.
3. **§2 Mécanismes** — P1→P6 chacune avec : ce que montre la figure, la phrase-clé pour l'oral, le piège
   de lecture éventuel.
4. **§3 Où dans la chaîne** — F1→F4 avec le paragraphe « pourquoi QAT et PTQ divergent alors que le format
   est identique ».
5. **§4 Impact mesuré** — I2→I6 avec les chiffres commentés (chargés, pas recopiés en dur dans le texte
   markdown quand ils risquent de changer — utiliser des cellules qui affichent les valeurs).
6. **§5 Limites & honnêteté** — plateformes (mesuré/émulé/« à mesurer »), paradoxe latence, travaux futurs.

## Règles

- Respect de la convention dépôt : notebook dans `notebooks/`, exécuté via **nbconvert sans erreur** avant
  livraison.
- Les paragraphes FR sont rédigés pour être **auto-portants** (copiables tels quels) et cohérents avec
  `docs/context/quantization_strategies.md` (S4202) — en cas de doublon, le notebook pointe vers le doc.
- Aucun chiffre en dur dans le markdown pour les valeurs susceptibles d'être re-mesurées (board v2) —
  affichage par cellule de code.

## Critères d'acceptation

1. `jupyter nbconvert --execute` passe sans erreur.
2. Les 17 figures (6 P + 5 F + 6 I) présentes et commentées.
3. Relancer §0 après une mise à jour d'expérience régénère figures **et** valeurs affichées sans édition
   manuelle du notebook.

## Réalisation (7 juillet 2026)

- `notebooks/cl_eval/quantization_figures/catalog.ipynb` (49 cellules) : §0 Reproduction (appelle les 3 builders via le registre, `set_seed(42)` + `apply_style`), §1 Panorama (tableau comparatif + F5 + I1), §2 Mécanismes P1–P6, §3 Où dans la chaîne F1–F4 (+ paragraphe « pourquoi QAT/PTQ divergent »), §4 Impact I2–I6 (valeurs **chargées par cellule de code**, pas en dur), §5 Limites & honnêteté.
- **N'implémente aucune figure** : consomme `src/figures/` (0 duplication de tracé), 17 figures affichées via `IPython.display.Image`.
- **`jupyter nbconvert --execute` passe sans erreur** (0 output d'erreur) ; les cellules §4 impriment les valeurs réelles (ablation, corr Q15, latences, v2 board Pronostia = valeur chargée / Monitoring = « à mesurer »). Racine du dépôt résolue par remontée jusqu'à `pyproject.toml` (robuste sous nbconvert).
