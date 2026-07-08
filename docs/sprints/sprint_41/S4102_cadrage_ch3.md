# Fiche de cadrage — Ch. 3 Problématique & positionnement (~1.5 p., cible md `03_problematique.md`)

## Messages clés

1. **Triple lacune** reprise de l'intermédiaire mais reformulée au réalisé :
   - Gap 1 : validation sur données industrielles temporelles réelles → **traité au ch. 5**.
   - Gap 2 : démonstration ≤ 256 Ko RAM avec mesures précises → **traité au ch. 6**.
   - Gap 3 : quantification pendant l'entraînement incrémental → **traité au ch. 7**.
2. **Écarts assumés vs l'intermédiaire** (transparence pour les rapporteurs qui ont lu la v1) :
   - cadre mixte supervisé + non supervisé (justifié ch. 1) ;
   - périmètre datasets élargi de 2 → 6 (D1 Pump finalement marginal ; CMAPSS/Pronostia/Monitoring
     au cœur, CWRU/Paderborn en annexe) ;
   - la quantification, « non prioritaire » dans l'intermédiaire, est devenue un axe central (Gap 3).
3. Positionnement : aucun travail du corpus ne satisfait les 3 gaps simultanément ; la contribution
   du stage est la démonstration conjointe sur board réelle.

## Contenu source

- Ch. 4 du manuscrit intermédiaire (pages 14–15) — actualiser les formulations « à établir » → « établi/mesuré ».
- `docs/triple_gap.md` (état à jour des 3 gaps, avec nuances).

## Chiffres

- Aucun (les chiffres vont aux ch. 5–7). Au plus, renvois « (cf. chapitre N) ».

## Figures

- Aucune.

## Refs bib

Reprise des refs des lacunes : `Ren2021`, `Kwon2023`, `Ravaglia2021`, `Su2019`*, `Park2018`*,
`Zong2018`*, `Wu2025`, `Hurtado2023` (* = à ajouter S4103).

## Glossaire touché

Rien de nouveau.

## Points ouverts

- Formulation exacte du statut du Gap 3 (« comblé côté QAT PC, corrigé côté board par kernel v2 » vs
  « partiellement comblé ») — dépend de l'issue du Sprint 39/40 ; trancher en S4107/S4110.
