# Fiche de cadrage — Ch. 2 Contexte & état de l'art condensé (~6 p., cible md `02_contexte_etat_art.md`)

## Messages clés

1. PdM industrielle + non-stationnarité (types de drift) — condensé de l'intermédiaire ch. 2.
2. Limites du batch en contexte embarqué (mémoire/connectivité/réactivité) → apprentissage incrémental.
3. Oubli catastrophique + dilemme stabilité–plasticité ; taxonomie régularisation/rejeu/architecture ;
   scénario pertinent = domain-incremental (et class-incremental pour Pronostia).
4. CL sur MCU : TinyOL, QLR-CL, HDC, LifeLearner — les 4 piliers de l'intermédiaire, resserrés.
5. **Nouveau vs intermédiaire — sous-section quantification** (fondement du Gap 3) : PTQ vs QAT,
   INT8/Q15, pourquoi la quantification *pendant* l'entraînement incrémental est un problème ouvert.
6. Détection d'anomalies (MTSAD) : garder mais **réduire fortement** (l'intermédiaire y consacrait
   ~2 p. ; le cadre final est mixte) — Mahalanobis comme baseline légère à introduire ici.

## Contenu source

- Ch. 2 + 3 du manuscrit intermédiaire (pages 7–13) — condenser de ~7 p. à ~6 p. en absorbant la
  section plateforme matérielle dans le ch. 4 (méthodologie).
- Nouvelle matière quantification : `docs/sprints/sprint_39/S3901_audit_int8_actuel.md` (PTQ vs QAT,
  causes de dégradation) pour le cadrage conceptuel — sans résultats (ils vont au ch. 7).

## Chiffres

- Uniquement des chiffres issus de la littérature (déjà dans l'intermédiaire : TinyOL +10 % latence,
  LifeLearner 212 Ko SRAM, QLR-CL ×4 buffer, HDC 85 % EMG). Aucun résultat du projet ici.

## Figures

- 0 à 1 max (taxonomie CL éventuelle). Recommandation : aucune, budget serré.

## Refs bib

Existantes : `Kirkpatrick2017`, `DeLange2021`, `Wang2023`, `Ren2021`, `Ravaglia2021`, `Kwon2023`,
`Benatti2019`, `Lin2024`, `Capogrosso2023`, `Hurtado2023`, `BesnardRagot2024`, `BerghoutBenbouzid2022`,
`Zenke2017`, `Li2018LwF`, `Aljundi2018MAS`, `Rebuffi2017`, `LopezPaz2017`.
**À ajouter (S4103)** : `Belay2023` (survey MTSAD, cité dans l'intermédiaire mais absent du .bib),
`Park2018`, `Su2019`, `Zong2018` (si la section MTSAD les garde), `Jacob2018` + `Krishnamoorthi2018`
(quantification PTQ/QAT), `Mahalanobis1936`.

## Glossaire touché

EWC, Online-EWC, FIM, TinyOL, QLR-CL, HDC (entrée à créer), MTSAD, PTQ/QAT (à créer), Q15 (à créer),
SGD (à créer), DIL/CIL.

## Points ouverts

- Arbitrage : combien garder de la section MTSAD (LSTM-VAE/OmniAnomaly/CURL/CaSSLe) ? Recommandation :
  1 paragraphe de synthèse + refs, en supprimant les descriptions détaillées (gain ~1 p.).
