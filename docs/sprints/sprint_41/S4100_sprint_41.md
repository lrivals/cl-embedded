# Sprint 41 — Rédaction du manuscrit final M2

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 41 |
| **Démarrage** | 3 juillet 2026 |
| **Statut** | 🟡 En cours — S4101–S4106 ✅ + S4108 ✅ (infrastructure, cadrage, audits, ch. 1–4 + ch. 8 rédigés) ; S4107 (ch. 5–7), S4109, S4110 déclenchées à la demande |
| **Livrable final** | `Manuscrit Final RIVALS.pdf` — ~30 pages de texte FR (hors abstracts, TOC, biblio, annexes), dépôt Moodle « Dépôt Manuscrit Final » |
| **Dépendances** | Manuscrit intermédiaire + retours rapporteurs (Poss, Giroudeau) · Sprints 16–38 ✅ · Sprint 39 🟡 (INT8 v2, chiffres en évolution) · Sprint 40 🟡 |

## Consignes officielles

- ~30 pages de texte, en français, reprenant le rapport de présentation en détail,
  **en prenant en compte le retour des rapporteurs**.
- Exposé clair de : méthodologie, contribution, évaluation de la contribution, perspectives.
- Relu par les rapporteurs → communiqué au jury pour la notation.

## Retours rapporteurs à intégrer

1. **Limite de pages dépassée** (Poss : « un petit peu au-dessus » ; Giroudeau : consigne non
   respectée, « modèle à suivre : Non ») → discipline stricte sur les ~30 p., budget par chapitre.
2. **Poss — domaine flou** (« micro-électronique embarquée, IA, intersection des 2 ? ») →
   clarifier dès l'introduction : le sujet EST l'intersection (CL × TinyML × maintenance prédictive).

## Décisions de cadrage (validées par Léonard, 3 juillet 2026)

| Décision | Choix |
|---|---|
| Fil narratif contribution | **Par triple gap** (Gap 1 / Gap 2 / Gap 3) |
| Études avancées dans le corps | Socle (4 modèles, portage board, RAM, INT8 vs FP32) + **S36 (comparaison appariée PC↔board)** uniquement ; S34 Q15 / S35 features / S38 gate autonome → perspectives (± annexe) |
| Énergie (S33) | **Exclue** du corps (au plus une ligne en perspectives) |
| Chapitres 1–4 intermédiaires | **Condensés à ~8–10 p.** |
| Cadrage supervisé/non-supervisé | **Assumer l'évolution** vers un cadre mixte (supervisé online + Mahalanobis non supervisé), expliqué dès l'intro |
| Datasets dans le corps | **Focus CMAPSS + Pronostia + Monitoring** ; grilles complètes 4×5 et D1/D3/D6 en annexe |
| Chiffres RAM / INT8-FP32 (S39/S40 en cours) | **Placeholders `[à confirmer — exp_XXX]`** + tâche S4110 de résolution finale |

## Règles de production (workflow imposé)

1. Chaque tâche de rédaction produit un **fichier md** dans `docs/rapport_de_stage/FIchier_md/` —
   **jamais** de modification du projet Overleaf (`Manuscrit_Final_Rivals/`) sans instruction explicite.
2. Corrections/mise en forme dans l'Overleaf : uniquement sur demande, après passage par le md.
3. **Aucun chiffre inventé** : chaque valeur du manuscrit cite sa source
   (`experiments/exp_XXX/....json`, champ précis) ou porte un placeholder.
4. Rigueur glossaire + biblio (audits S4103/S4104).
5. Notebook regroupant tous les plots du manuscrit en fin de sprint (S4109).
6. `docs/rapport_de_stage/` est **gitignoré** (S4101) — les textes du manuscrit ne sont pas versionnés ;
   seuls les docs de sprint (fiches, audits) le sont.

## Plan cible du manuscrit (~30 p.)

| Ch. | Titre | Budget | Fiche de cadrage |
|---|---|---|---|
| 1 | Introduction (domaine clarifié, cadrage mixte assumé, objectifs, plan) | ~2.5 p. | `S4102_cadrage_ch1.md` |
| 2 | Contexte & état de l'art condensé | ~6 p. | `S4102_cadrage_ch2.md` |
| 3 | Problématique : triple lacune & positionnement | ~1.5 p. | `S4102_cadrage_ch3.md` |
| 4 | Méthodologie (datasets, modèles, pipeline PC→C, protocole de mesure board) | ~6 p. | `S4102_cadrage_ch4.md` |
| 5 | Gap 1 — Validation sur données industrielles | ~4 p. | `S4102_cadrage_ch5.md` |
| 6 | Gap 2 — RAM & latence mesurées + parité PC↔board (S36) | ~4.5 p. | `S4102_cadrage_ch6.md` |
| 7 | Gap 3 — Quantification pendant l'entraînement incrémental | ~3.5 p. | `S4102_cadrage_ch7.md` |
| 8 | Perspectives & conclusion | ~2 p. | `S4102_cadrage_ch8.md` |
| — | Abstracts FR/EN + annexes (grilles 4×5, D1/D3/D6, figures) | hors quota | fiche ch8 |

## Tâches

| Tâche | Description | Statut |
|---|---|---|
| S4101 | Infrastructure : gitignore `docs/rapport_de_stage/`, arborescence `FIchier_md/`, doc sprint | ✅ |
| S4102 | Fiches de cadrage des 8 chapitres (messages clés, chiffres+sources vérifiées, figures, refs, glossaire, budget) | ✅ |
| S4103 | Audit biblio `references.bib` : entrées manquantes + doublons + BibTeX prêts | ✅ |
| S4104 | Audit glossaire/acronymes : entrées obsolètes + nouvelles entrées à créer | ✅ |
| S4105 | Rédaction ch. 1–3 (md) | ✅ (`01_introduction.md`, `02_contexte_etat_art.md`, `03_problematique.md`) |
| S4106 | Rédaction ch. 4 (md) | ✅ (`04_methodologie.md`) |
| S4107 | Rédaction ch. 5–7 (md, placeholders RAM/INT8) | ⏳ à la demande |
| S4108 | Rédaction ch. 8 + abstracts FR/EN + annexes (md) | ✅ (`08_perspectives_conclusion.md`, `09_abstracts_annexes.md` ; perspectives = 3 axes drift/CL, features, énergie ; Q15 exclu par décision) |
| S4109 | Notebook figures `notebooks/manuscrit_final/figures.ipynb` (0 valeur en dur) → `docs/figures/manuscrit_final/` | ⏳ |
| S4110 | Consolidation : résolution placeholders depuis S39/S40 finalisés, vérif croisée chiffres↔JSON, comptage pages, checklist consignes | ⏳ |

## Points de vigilance (informations manquantes ou à trancher)

1. **Chiffres RAM / INT8 v2 en évolution** (Sprint 39 🟡, Sprint 40 board validation v2) —
   le ch. 7 doit intégrer le résultat final de `exp_S39_quant_sweep`/`exp_S39_matched` et,
   si disponible, la validation board v2 (S4002).
2. **Narratif RAM cohérent** : plusieurs `.bss` coexistent (1 000 B Sprint 20 → 105 036 B défaut →
   183 936 B S35 `all`) — le ch. 6 doit dire QUELLE config porte le claim Gap 2.
3. **Nuance Gap 3** : « comblé » côté PC/QAT (Δ≤0.006, S28) mais PTQ board historique dégradée
   (F1 0.07–0.15, S36) → corrigée par le kernel v2 per-channel (S39) ; formulation exacte à
   valider avec Léonard en S4107 selon l'état des mesures.
4. **Écart annoncé/réalisé** : D1 Pump central dans l'intermédiaire, marginal dans les résultats ;
   2 datasets annoncés → 6 utilisés — à assumer au ch. 3.
5. **Abstracts FR/EN à réécrire** (l'existant annonce du travail futur).
6. **Overleaf en retard d'une version** sur le rendu intermédiaire — les fiches se basent sur le
   PDF rendu (`Manuscrit_Presentation_Rivals_2026-1.md`), pas sur les `.tex`.
