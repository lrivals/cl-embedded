# S4011 — Annexes générées, renvois croisés et liste des acronymes

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 40 |
| **Statut** | ✅ Implémenté |
| **Fichiers cibles** | `scripts/generate_article_tables.py`, `docs/article/ewc_int8_mcu/tables/`, `main_{fr,en}.tex`, `tests/test_article_tables.py` |
| **Dépendances** | S4008 (agrégat), S4004 (structure + glossaire) |

## Motivation

L'article déclare en section « Disponibilité » qu'**aucune valeur n'y est saisie à la
main**. Une annexe qui détaille des dizaines de cellules ne peut donc pas être tapée :
elle doit être **générée depuis l'agrégat**, faute de quoi la déclaration devient
fausse dès la première retouche.

Par ailleurs le corps citait des points de décision (frontière ternaire, rapport RAM,
trois estimateurs) sans jamais donner la grille dont ils sont extraits, et sans
pointer vers la liste des acronymes.

## Réalisation

### Générateur

`scripts/generate_article_tables.py` lit `experiments/exp_S40_article_metrics/summary.json`
et écrit **10 fichiers** (5 tables × 2 langues) dans `docs/article/ewc_int8_mcu/tables/`.
Aucun calcul : uniquement de la mise en forme de valeurs lues. Une cellule sans mesure
sort en `--`, **jamais en 0**.

| Table | Contenu | Nature |
|-------|---------|--------|
| `annex_depth` | 7 profondeurs × 2 granularités × 2 jeux, ΔAUROC | émulé PC |
| `annex_symmetry` | symétrique vs affine à 3 profondeurs × 2 jeux | émulé PC |
| `annex_ram` | `.data` / `.bss` / pics de pile / total / rapport | mesuré carte |
| `annex_energy` | les 3 estimateurs côte à côte | mesuré carte |
| `annex_missing` | registre des cellules « à mesurer » + raison | — |

### Annexes et renvois

Trois annexes (`app:depth`, `app:budget`, `app:missing`) placées **après la
bibliographie et avant le glossaire**, patron du manuscrit. Renvois posés depuis le
corps : §6.2 → grille de profondeur et de symétrie, §7.1 → composantes RAM, §7.3 →
estimateurs, §4 → registre des manquantes **et** liste des acronymes via
`\pageref{sec:acronyms}`.

Ancre `\label{sec:acronyms}` ajoutée sur `\printglossary`. Macros de repli
`\ampere`, `\hour`, `\si` complétées.

### Constat rendu visible par l'annexe

Le Tableau `tab:annex-ram` montre que le `.bss` est **identique** entre FP32 et INT8
(100 152 o sur Monitoring) : l'écart de total ne vient que du pic de pile. C'est
l'argument de §7.1 rendu vérifiable ligne à ligne.

### Pas de figure ajoutée

Les 10 figures du catalogue couvrent déjà les axes (profondeur en fig7, RAM en fig8,
énergie et arbitrage de fréquence en fig9). Ajouter des figures d'annexe supposerait
de nouveaux constructeurs dans `src/figures/catalogs/article_ewc.py` pour un contenu
redondant ; la valeur de l'annexe est dans le **détail chiffré**, que les figures ne
donnent pas.

## Vérification

```bash
python scripts/generate_article_tables.py
cd docs/article/ewc_int8_mcu && make clean && make all
pytest tests/test_article_tables.py -v
```

`tests/test_article_tables.py` **6/6** : présence FR+EN, en-tête « généré »,
idempotence de la régénération, registre des manquantes sans zéro substitué, formes
FR/EN identiques, totaux RAM conformes à l'agrégat.

État : **25 pages FR / 24 EN**, 0 erreur LaTeX, 0 référence indéfinie, 0 lien de
glossaire cassé, 9 entrées bibliographiques toutes citées ; suite article **30 passés,
2 ignorés**.
