# S4006 — Rédaction de l'article (version anglaise)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 40 |
| **Priorité** | 🔴 Critique |
| **Statut** | ✅ Implémenté — `main_en.tex` + 7 sections EN, miroir strict FR (décimales FR≡EN vérifiées), `make en` OK |
| **Durée estimée** | ~6h |
| **Dépendances** | S4005 (version FR de référence) · S4004 (squelette) · S4003 (figures) |
| **Fichiers cibles** | `docs/article/ewc_int8_mcu/main_en.tex` + `sections/` EN |
| **Références** | S4005 (miroir strict) |

## Contexte

Version anglaise, **miroir strict** de la version française (S4005) : même structure, mêmes figures, mêmes
tables, mêmes chiffres. Aucune divergence numérique ou d'argument — seule la langue change. Cible probable :
workshop/revue TinyML (à confirmer `TODO(arnaud)`).

## Spec

- Traduction fidèle section par section de `main_fr.tex` ; réutiliser **exactement** les mêmes figures
  `docs/figures/sprint40_article/` et tables (mêmes valeurs).
- Terminologie technique cohérente : *continual learning*, *post-training quantization (PTQ)*,
  *quantization-aware training (QAT)*, *per-channel scale*, *dequantization*, *frozen/online*, *board vs PC*.
- Conserver la distinction **« measured on-board »** vs **« bit-exact PC emulation »**.
- Abstract, keywords, légendes en anglais.

## Vérification

```bash
cd docs/article/ewc_int8_mcu && make en    # main_en.pdf compile sans erreur
# cohérence FR≡EN des valeurs clés vérifiée par tests/test_sprint40_article.py (S4007)
```

> **Invariant** : les chiffres clés (parité, F1, latences, ratios RAM) doivent être **identiques** entre
> `main_fr.tex` et `main_en.tex` — vérifié automatiquement en S4007.


## Passe de rédaction locale (septembre 2026)

Miroir anglais strict de la passe décrite dans `S4005` : `01`, `02`, `03`, `04` et
`07` réécrits dans la même passe que le français, abstract aligné sur `0.138` /
`0.1337`, glossaire `glossary_en.tex` (24 acronymes, mêmes clés que le français).

Le miroir numérique reste vérifié par `test_fr_en_key_values`, qui compare
l'ensemble des décimales de `main_*.tex` et `sections/<lang>/*.tex`. Les fichiers de
glossaire vivent à la racine de l'article, donc hors du périmètre de ce test : leur
parité de clés est à contrôler à la main (`diff` des motifs `\newacronym`).
