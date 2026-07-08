# S4201 — Infrastructure réutilisable de génération de figures (`src/figures/`)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 42 |
| **Priorité** | 🔴 Critique — socle de tout le sprint et des figures futures |
| **Durée estimée** | ~8h |
| **Statut** | ✅ Implémenté (7 juillet 2026) |
| **Dépendances** | `src/evaluation/plots.py` (helpers existants, à réutiliser sans dupliquer) |

## Objectif

Créer un module **pérenne** de génération de figures de présentation/manuscrit, indépendant de tout sprint,
avec trois propriétés :

1. **Style commun** — toutes les figures du projet partagent police, palette, tailles, format d'export.
2. **Données tracées, jamais inventées** — chargement centralisé des JSON `experiments/` avec traçabilité
   (chaque figure sait de quels fichiers elle provient).
3. **Régénérable en une commande** — `python scripts/generate_figures.py --catalog quantization` reproduit
   tous les PNG d'un catalogue ; un catalogue futur s'ajoute sans modifier l'infra.

## Architecture cible

```
src/figures/
├── __init__.py
├── style.py          # apply_style(), palette, tailles slides vs manuscrit, savefig_png()
├── loaders.py        # load_experiment(exp_id) → dict + provenance ; iter_experiments(glob)
├── registry.py       # register_catalog(name)(fn) ; get_catalog(name) ; list_catalogs()
└── catalogs/
    ├── __init__.py   # importe les catalogues → auto-enregistrement
    ├── quant_pedagogy.py   # S4203
    ├── quant_pipeline.py   # S4204
    └── quant_impact.py     # S4205

scripts/generate_figures.py   # CLI : --catalog <nom> | --all | --list ; --out docs/figures/
```

## Spécifications

### `style.py`

- `apply_style(target: str = "slide")` — deux presets : `"slide"` (grandes polices, fond clair, 16:9-friendly)
  et `"manuscript"` (tailles LaTeX, sobre). Basé sur `matplotlib.rcParams`, pas de dépendance nouvelle.
- Palette nommée du projet (réutiliser les conventions de `src/evaluation/plots.py` ; couleurs stables par
  stratégie : FP32, INT8-QAT, INT8-PTQ-legacy, INT8-v2, Q15, int16-AM — **mêmes couleurs dans toutes les
  figures du catalogue**, c'est ce qui rend un jeu de slides lisible).
- `savefig_png(fig, catalog, name)` — export normalisé `docs/figures/<catalog>/<name>.png` (dpi fixe,
  `bbox_inches="tight"`), crée le dossier, retourne le chemin.
- Langue : **labels FR** ; les chaînes de labels de chaque catalogue sont regroupées dans un dict `LABELS`
  en tête de fichier (rend une future option `--lang en` mécanique, non implémentée ici).

### `loaders.py`

- `load_experiment(path)` — charge un JSON de `experiments/`, retourne `(data, provenance)` où provenance =
  chemin + mtime ; lève une erreur claire si absent (pas de valeur par défaut silencieuse).
- `metric_or_na(data, key)` — retourne la valeur ou le sentinel `"à mesurer"` / `None` si champ absent ou
  `null` (`na_reason` honoré, convention Sprints 29/33) — **jamais** 0 par défaut.
- Helpers d'agrégation lecture seule pour les formes récurrentes du dépôt (grilles modèle×dataset type
  `exp_S28_PC_*`, summaries type `exp_S36_summary.json`).

### `registry.py`

- Décorateur `@register_catalog("quantization/pedagogy")` sur une fonction `build(out_dir) -> list[Path]`
  qui génère les figures et retourne les chemins produits.
- `list_catalogs()` pour la CLI `--list` ; un catalogue = une fonction, zéro classe imposée.

### `scripts/generate_figures.py`

- `--catalog <nom>` (préfixe accepté : `quantization` lance les 3 sous-catalogues), `--all`, `--list`,
  `--style {slide,manuscript}` (défaut `slide`), `--out` (défaut `docs/figures/`).
- Sortie console : liste des PNG (ré)générés + les expériences sources utilisées (provenance).
- Idempotent : relancer produit les mêmes fichiers (seed fixé si un catalogue échantillonne).

## Contraintes

- **Réutiliser** `src/evaluation/plots.py` (heatmaps, etc.) — l'infra les enveloppe, ne les réécrit pas.
- Aucune dépendance nouvelle (matplotlib/numpy/pandas déjà présents). Pas de graphviz : les schémas S4204
  se font en matplotlib (patches/annotate), pour rester régénérables partout.
- Les notebooks existants ne sont **pas migrés** — l'infra est pour les figures transversales à venir ;
  la migration éventuelle des figures de sprints passés est hors périmètre.

## Critères d'acceptation

1. `python scripts/generate_figures.py --list` affiche les 3 catalogues quantification.
2. `--catalog quantization` régénère tous les PNG sous `docs/figures/quantization/` sans erreur.
3. Un catalogue jouet ajouté dans `catalogs/` (test) apparaît dans `--list` sans modification de l'infra.
4. Aucune valeur numérique littérale de résultat dans `src/figures/` (vérifié par test S4207).

## Réalisation (7 juillet 2026)

- `src/figures/{style,loaders,registry}.py` + `catalogs/__init__.py` + CLI `scripts/generate_figures.py` (`--catalog` par préfixe, `--all`, `--list`, `--style`, `--out`, `set_seed(42)`, provenance affichée).
- Palette `STRATEGY_COLORS` définie (aucune n'existait) : fp32 bleu, int8_qat vert, int8_ptq_legacy rouge, int8_v2 orange, q15 violet, int16_am brun — famille Material de `plots.py`.
- `metric_or_na` distingue mesuré / `« à mesurer »` (S33) / `None`+`na_reason` (S29) ; `load_experiment` lève `FileNotFoundError` explicite.
- Critères : `--list` OK ; `--catalog quantization` régénère 6 PNG sans erreur, idempotent ; catalogue jouet enregistrable sans toucher l'infra (vérifié) ; 0 chiffre de résultat en dur (grep). Un seul catalogue enregistré à ce stade (pedagogy) — pipeline/impact arrivent avec S4204/S4205. Tests formels → S4207.
