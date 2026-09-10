# S4005 — Rédaction de l'article (version française)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 40 |
| **Priorité** | 🔴 Critique |
| **Statut** | ✅ Implémenté — `main_fr.tex` + 7 sections FR, tables chiffres canoniques (adossés JSON), 5 figures, distinction mesuré/émulé, board v2 « à mesurer » ; `make fr` OK |
| **Durée estimée** | ~10h |
| **Dépendances** | S4004 (squelette) · S4003 (figures) |
| **Fichiers cibles** | `docs/article/ewc_int8_mcu/main_fr.tex` + `sections/` FR |
| **Références** | S4000 (message scientifique) · exp_S36 · exp_S39 |

## Contexte

Rédiger la version française complète. **Tous les chiffres proviennent du notebook S4003** (donc des JSON).
Le tableau ci-dessous fixe les **valeurs canoniques attendues** (issues de `exp_S36_summary.json` et
`exp_S39_ablation/`) pour relecture — mais le `.tex` doit les référencer via les figures/tables générées,
pas les figer à la main.

## Chiffres canoniques (relecture — ne pas hardcoder, vérifier ≡ JSON)

### Parité & performance FP32 PC↔board (Sprint 36)
| Dataset | Cond. | acc PC | acc board frozen | Δacc | F1 PC/board | parité frozen | parité online |
|---------|:----:|:-----:|:----------------:|:----:|:-----------:|:-------------:|:-------------:|
| Pronostia | 5feat | 0.9887 | 0.9821 | 0.0066 | 0.9164 | 1.000 | 0.975 |
| Pronostia | all | 0.9834 | 0.9831 | 0.0003 | 0.9180 | 1.000 | 0.963 |
| Monitoring | 5feat | — | 0.9846 | — | 0.9194 | 1.000 | 0.989 |

Latences board : frozen 48–65 µs · online 239–340 µs (inf. + MAJ) ≪ 100 ms (**Gap 2**). `.bss` 100–145 Ko < 256 Ko.

### INT8 vs FP32 board — legacy (Sprint 36, effondrement)
| Dataset | Cond. | F1 FP32 | F1 INT8 legacy | accord INT8↔FP32 | RAM ÷ |
|---------|:----:|:-------:|:--------------:|:----------------:|:-----:|
| Pronostia | 5feat | ≈0.916 | **0.138** | 0.736 | ×4 |
| Pronostia | 5feat (online) | — | **0.085** | 0.867 | ×4 |
| Monitoring | 5feat | ≈0.919 | 0.05–0.15 | 0.595 | ×4 |

### Échelle d'ablation (Sprint 39, émulateur — explique la cause)
| Schéma | Pronostia F1 | Monitoring F1 |
|--------|:-----------:|:-------------:|
| FP32 | 0.9616 | 0.9194 |
| legacy_c | 0.0663 | 0.1178 |
| **per_tensor_calib** | **0.9462** (+0.88) | **0.9201** (+0.88) |
| per_channel_int8 | 0.9426 | 0.9187 |
| q15 | 0.9616 | 0.9194 |

### Récupération board v2 (Sprint 40, S4002)
→ **`"à mesurer"`** tant que la carte n'a pas streamé. Ne rien affirmer de chiffré côté board v2 avant
l'exécution réelle.

## Consignes de rédaction

- Ton académique, français, structure S4004.
- **Distinction explicite** dans le texte et les légendes : « mesuré sur carte » (Sprint 36 FP32 + legacy,
  Sprint 40 v2) vs « émulé PC bit-exact » (Sprint 39 ablation/récupération).
- Assumer le **paradoxe latence** INT8 (RAM ÷4 sans accélération FPU) — le présenter comme constat honnête +
  piste SIMD, pas le masquer.
- Insérer les figures depuis `docs/figures/sprint40_article/`.

## Vérification

```bash
cd docs/article/ewc_int8_mcu && make fr    # main_fr.pdf compile sans erreur
```


## Passe de rédaction locale (septembre 2026)

Rédaction reprise en local sur la chaîne LaTeX du rapport de stage. Trois constats
mesurés motivaient la reprise, au-delà du confort d'édition.

1. **`01_intro` et `07_conclusion` étaient en deçà de leurs propres résultats.** La
   contribution (iii) annonçait la récupération INT8 comme *émulée* alors que la
   campagne v2 en mesure quatre cellules per-canal sur carte, et la conclusion
   donnait encore comme « priorité immédiate » de compléter cette campagne. C'est le
   défaut qui avait motivé la refonte S4008–S4010, corrigé dans l'abstract et la
   section 5 mais **pas** dans ces deux sections. Le diagnostic reste émulé, la
   récupération ne l'est plus : le texte le dit désormais.
2. **Les sections 5b et 5c n'existaient dans aucune des sections amont.** Les
   contributions de l'intro, la revue (`02`), la méthode (`03`) et le dispositif
   (`04`) s'arrêtaient à la calibration du noyau. La méthode ne décrivait donc pas
   les mesures que l'article rapporte : moment, profondeur, RAM totale
   (`.data + .bss + pic de pile`), segments DWT, trois estimateurs d'énergie. Trois
   sous-sections ajoutées à `03`, tableau de provenance ajouté à `04`.
3. **Le Gap 2 était énoncé « sous 100 ko de RAM », contredit par la section 5c** qui
   mesure 105 300 o de RAM totale. Reformulé en budget mémoire et latence mesuré,
   256 ko de SRAM et 100 ms par cycle.

Fil conducteur explicité dans l'intro, la discussion et la conclusion : l'INT8 est
justifié par un argument tripartite (mémoire, temps, énergie) que l'article
**dissocie** — la première promesse est tenue mais porte sur un poste minoritaire,
les deux autres ne le sont pas.

Abstract aligné sur les valeurs mesurées carte (`0.138` / `0.1337`) : la fourchette
`0.05`–`0.15` empruntait sa borne basse à l'ablation émulée alors que la phrase dit
« sur la carte ».

Glossaire de 24 acronymes ajouté en fin de document (voir S4004).

État : `make all` → 21 pages FR et EN, 0 erreur LaTeX, 0 référence indéfinie,
0 lien de glossaire cassé ; `pytest tests/test_sprint40_article.py
tests/test_article_metrics.py` → 24 passés, 2 ignorés (cellules de banc).

### Passe de cohérence et fraîcheur (suite)

**Vérification de fraîcheur des résultats, faite avant de toucher au texte.** Aucun
JSON source n'est postérieur à `exp_S40_article_metrics/summary.json` ; l'agrégat
régénéré par `scripts/aggregate_article_ewc.py` est **identique hors `_meta`** à celui
versionné ; les 101 valeurs citées dans le texte français ont été confrontées une à
une à l'agrégat. Les valeurs de la condition `all` du Tableau `tab:parity`
(`0.9180`, `0.9626`) ne sont pas dans l'agrégat, qui est `5feat` par construction :
elles ont été vérifiées directement contre `exp_S36_summary.json`.

**Deux résultats récents n'étaient pas exploités.** La cellule **90 MHz** du balayage
de fréquence (`192.64` µJ) était `N/A` avant son refit (r²=0.999, correctif S4010) :
la section 7.3 ne donnait que les deux extrémités, elle affiche désormais les trois
points et parle de croissance **monotone**. L'**autonomie mesurée** (`72.95` h à
2000 mAh, 1 inférence/s) chiffrait l'affirmation « la marge de latence est
convertible en autonomie », jusque-là qualitative. Unités `\ampere`/`\hour` ajoutées
au bloc de repli siunitx.

**Corrections de cohérence.**
- 5.3 « Diagnostic et récupération (émulé PC) » contredisait 5.4 « Récupération
  mesurée sur carte » : la sous-section 5.3 devient « Diagnostic de l'effondrement »
  et se clôt sur un renvoi explicite vers la mesure. Dernier reste du cadrage
  d'avant la refonte S4008–S4010.
- Les sous-sections de méthode ajoutées dupliquaient les titres des sections de
  résultats (3.4 ≡ 6.1, 3.5 ≈ 6.2). Renommées en « Axe du … » : la méthode nomme la
  définition, les résultats le constat.
- Les sections 6 et 7 ne renvoyaient nulle part à la méthode alors que 3.6 décrit
  leur instrumentation. Renvois `\S\ref{sec:method}` ajoutés en tête de chacune.
- Le fil des « trois promesses » s'arrêtait aux extrémités du texte ; l'accroche de
  la section 7, qui est celle qui y répond, le reprend désormais.

**Parties standard ajoutées** avant la bibliographie, en FR et EN : disponibilité du
code et des données (règle « aucune valeur saisie à la main »), remerciements.

État : `make all` → **22 pages** FR et EN, 0 erreur, 0 référence indéfinie, 0 lien de
glossaire cassé ; 24 tests passés, 2 ignorés.
