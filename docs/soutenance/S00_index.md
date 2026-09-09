# S00 — Préparation de soutenance : index

**Soutenance** : mémoire M2 « Apprentissage Incrémental pour Capteurs Intelligents à
Ressources Limitées » — Léonard Rivals, ISAE-SUPAERO (DISC) / ENAC (LII) / Edge Spectrum.
**Format** : **15 min d'exposé + 15 min de questions.**

> **Le format a changé.** Le dossier a d'abord été construit sur 30 min d'exposé. `S00`, `S02`,
> `S03` et `S05` sont tous réalignés sur **15 min / 15 slides** — aucun document du dossier ne
> décrit plus l'ancien format.
>
> Conséquence de fond, et pas seulement de forme : à 15 min, l'exposé suit **un seul modèle —
> EWC — de bout en bout**. Les trois autres familles n'apparaissent qu'une fois, sur la slide
> de classement. Ce qui sort du fil descend en slide de secours dans `S03`, qui devient de ce
> fait plus sollicité qu'avant : une part de l'argumentation s'y joue désormais.

**Hypothèse structurante de tout ce dossier** : *le jury n'a lu que le manuscrit.* Ni le
dépôt, ni les JSON d'expériences, ni les fiches de sprint, ni les notebooks. Toute question
naîtra d'une phrase, d'un chiffre ou d'une figure du manuscrit, et toute réponse doit tenir
sans renvoyer au code.

---

## Les quatre documents

| Fichier | Contenu | Quand le lire |
|---|---|---|
| [`S01_questions_jury.md`](S01_questions_jury.md) | **60 questions** avec réponses préparées, en 5 sections : les 3 questions critiques, compréhension, points attaquables, positionnement scientifique, industriel. Plus une annexe « chiffres à connaître par cœur ». | Le cœur de la révision |
| [`S02_plan_presentation.md`](S02_plan_presentation.md) | Plan **slide par slide**, **15 slides / 15:00**, fil rouge EWC, avec message unique, figure exacte, notes orateur et minutage bloc par bloc. **Document de référence.** | Pour construire le deck |
| [`S03_slides_backup.md`](S03_slides_backup.md) | **15 slides de secours** (numérotées B1 à B16, sans B15) et la table de routage *question → slide* pour les 15 min de questions. | Pour la phase de questions |
| [`S05_script_oral.md`](S05_script_oral.md) | Le **texte parlé intégral des 15 minutes**, slide par slide (2 630 mots prononcés), plus les six phrases à savoir mot à mot. | Pour répéter à voix haute |

---

## Ce qu'il faut retenir en priorité

### Les trois questions qui peuvent faire basculer la soutenance

1. **Q-01** — « Si la parité PC ↔ carte est exacte à 1,000, comment la carte obtient-elle
   0,947 quand le PC obtient 0,893 ? » Les deux colonnes du tableau 5.1 ne portent pas la même
   population d'évaluation ; la parité mesure la fidélité du portage, pas la performance.
   **Slide de secours B1.**
2. **Q-02** — « Sur combien d'échantillons ? Quelle barre d'erreur ? » Les cellules de la
   grille sont mesurées sur une centaine d'échantillons ; les conclusions défendues sont
   ordinales, pas décimales. **Slide de secours B1.**
3. **Q-03** — « Votre problématique annonce du non supervisé, vos modèles focus sont
   supervisés. » Le §3.2 documente le basculement ; la phrase du §1.4 n'a pas été réalignée.

Ces trois-là se révisent **mot à mot**. Le reste se révise en compréhension.

### La règle de conduite

**Annoncer les limites avant que le jury ne les trouve.** Trois endroits dans l'exposé
(**slides 9, 10 et 14** de `S02`) et systématiquement dans les réponses. Une limite énoncée par
l'orateur est un point de rigueur ; trouvée par le jury, c'est un point faible.

### Ce que le jury ne peut pas anticiper

Quatre travaux existent mais **ne sont pas dans le manuscrit** — à garder en réserve, ils
donnent de la profondeur quand on est poussé : l'étude sub-INT8 jusqu'au binaire (**B6**),
la famille de détecteurs de dérive et leur portage (**B10**), les mesures de courant
(**B8**), l'article et la chaîne de publication (**B12**).

---

## Calendrier de révision

| Échéance | À faire |
|---|---|
| **J-7** | Lire `S01` en entier une fois. Relire le manuscrit avec `S01` à côté : chaque figure et chaque tableau doit évoquer sa question (checklist en fin de `S01`). |
| **J-5** | Construire le deck de **15 slides** depuis `S02`. Vérifier la lisibilité **projetée** des slides 4, 10 et 14, qui portent **deux figures chacune** — c'est le risque propre au format court. |
| **J-4** | Construire les 15 slides de secours `S03`. Les numéroter clairement pour pouvoir y sauter vite : à 15 min d'exposé, une part de l'argumentation est passée dans ce lot. |
| **J-3** | Première répétition chronométrée à voix haute. À 15 min il n'y a **aucune marge** : si ça déborde, comprimer les slides 2, 5 et 11, jamais 4, 6, 9 et 12. |
| **J-2** | Réviser les questions critiques Q-01 à Q-03 **mot à mot**. Réviser l'annexe « chiffres à connaître par cœur ». |
| **J-1** | Seconde répétition. Répéter spécifiquement les transitions **6 → 7, 9 → 10, 12 → 13** : ce sont les changements de gap, le jury doit les entendre. Vérifier que le **fil EWC** s'entend d'un bout à l'autre. |
| **J-0** | Relire uniquement : les trois questions critiques, l'annexe des chiffres, la table de routage de `S03`. Rien de neuf. |

---

## Sources

Manuscrit de référence : `docs/rapport_de_stage/manuscrit_overleaf/` (identique à
`Manuscrit_Final_Rivals/`, compilé plus récemment). 58 pages, 8 chapitres + annexes,
**15 figures**, **5 tableaux**.

**Figures projetées : `docs/figures/soutenance/`.** Les slides ne réutilisent plus les
figures du manuscrit — le jury l'a lu, revoir les mêmes images en projection coûte de
l'attention. Le catalogue `soutenance` régénère les figures depuis les **mêmes** mesures,
dans une forme adaptée à la salle (une figure = un message, aucun axe tronqué) :

```bash
python scripts/generate_figures.py --catalog soutenance --style slide
```

Nomenclature : `sNN_*.png`, `bNN_*.png` pour la slide de secours BNN de `S03`.

> ⚠️ **Les noms `sNN_` suivent l'ancienne numérotation à 26 slides et n'ont pas été
> renommés** — volontairement : renommer 26 fichiers casserait les renvois de `S03`, du
> manuscrit et des notebooks pour un bénéfice cosmétique. `S02` cite donc les figures par
> leur **chemin exact**, et sa table de traçabilité en fin de document donne la
> correspondance ancienne → nouvelle numérotation. Ne pas se fier au numéro dans le nom de
> fichier ; se fier au chemin écrit dans `S02`.

Toute valeur tracée est chargée depuis un JSON d'`experiments/` via
`src/figures/sources.py` — le même chargeur que `manuscrit_final`, pour que les chiffres
projetés et ceux du manuscrit ne puissent pas diverger. Une donnée absente s'affiche « N/A »,
jamais 0 (garde `tests/test_figures_library.py::test_no_hardcoded_results`).

**Toutes les figures projetées sont désormais dans `soutenance/`**, à une exception près et
assumée : les deux figures d'énergie de B8 (`docs/figures/energy_real/e5_cout_benefice.png`
et `e6_courant_moyen_mesure.png`) restent dans le catalogue `energy_real`, qui les produit
déjà avec valeurs chargées, N/A honnêtes et garde AST — les dupliquer créerait deux sources
pour une même figure.

Les scripts et notebooks hérités (`generate_presentation_plots.py`,
`generate_portage_plots.py`, `scenario_usecase_industriel.ipynb`,
`board_benchmark_all_datasets.ipynb`) ne sont pas supprimés : d'autres documents les citent.
Seuls les renvois de `docs/soutenance/` ont basculé.

**Trois captures de `Kirkpatrick2017EWC` sont déposées à la main** dans
`docs/figures/soutenance/` — elles ne sont **pas** produites par le catalogue et ne doivent
donc jamais être supprimées lors d'une régénération (`generate_figures.py` n'écrit que ses
propres noms, mais un `rm` du dossier les emporterait) :

| Fichier | Slide | Rôle |
|---|---|---|
| `KirkArticle_Frontpage.png` | 4 | page de titre — identifie le travail dont EWC est issu |
| `KirkArticle_Plot.png` | 3 | panneau A, contrepoint **publié** de notre mesure d'oubli |
| `KirkArticle_SchemaLoss.png` | 3 | bassins de perte — remplace la formule d'EWC |
| `MaintenancePredictive_Intro.png` | 2 | visuel d'introduction à la maintenance prédictive |

Ces quatre images sont **citées, pas produites par ce travail** : la mention de leur
source doit être visible sur la slide elle-même, pas seulement dite à l'oral.

Tous les chiffres cités dans ces documents sont ancrés soit dans le manuscrit — donc
opposables au jury — soit dans les campagnes du dépôt, et signalés comme tels lorsqu'ils ne
figurent pas dans le manuscrit.
