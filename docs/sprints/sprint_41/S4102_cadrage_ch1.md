# Fiche de cadrage — Ch. 1 Introduction (~2.5 p., cible md `01_introduction.md`)

## Messages clés

1. **Domaine clarifié dès le §1 (réponse directe au retour Poss)** : le stage se situe à
   l'**intersection** de trois champs — apprentissage incrémental (IA), systèmes embarqués à
   ressources limitées (TinyML/MCU) et maintenance prédictive industrielle. Ce n'est ni de la
   micro-électronique ni de l'IA « pure » : la contribution est précisément la conjonction.
2. **Contexte industriel** (repris condensé de l'intermédiaire) : drift de distribution → modèles
   figés se dégradent → réentraînement centralisé incompatible MCU.
3. **Évolution assumée du cadrage** : le rapport intermédiaire annonçait un cadre purement
   « détection d'anomalies non supervisée » ; le travail a évolué vers un **cadre mixte** —
   modèles supervisés online (EWC, TinyOL, HDC, labels disponibles dans les datasets PdM
   utilisés) + détecteur non supervisé (Mahalanobis) comme brique de référence et de gate.
   Justification : disponibilité des labels dans les jeux publics PdM, et comparabilité avec
   l'état de l'art embarqué (TinyOL/HDC supervisés).
4. **Objectifs réalisés** (plus « à venir ») : implémenter/comparer 3 méthodes CL + baseline
   Mahalanobis sur PC, les porter en C sur NUCLEO-F439ZI, mesurer précisément RAM/latence,
   étudier la quantification pendant l'entraînement — structurés par la triple lacune.
5. Annonce du plan (fil par triple gap).

## Contenu source

- Ch. 1 du manuscrit intermédiaire (`Manuscrit_Presentation_Rivals_2026-1.md`, pages 5–6) — condenser.
- Cadre du stage : ISAE-SUPAERO (DISC) × ENAC (LII) × Edge Spectrum, 16 mars – 6 août 2026 (CLAUDE.md).

## Chiffres (aucun résultat ici — seulement les specs matérielles)

- NUCLEO-F439ZI : STM32F439ZI, Cortex-M4 @ 180 MHz, FPU simple précision, 256 Ko SRAM
  (192 Ko SRAM + 64 Ko CCM), 2 Mo Flash, sans NPU — source `docs/context/` + CLAUDE.md.

## Figures

- Aucune (éventuellement 1 schéma « intersection des 3 domaines » si la place le permet — à créer S4109).

## Refs bib (clés `references.bib` existantes)

`Hurtado2023`, `Capogrosso2023`, `DeLange2021` (citation d'ancrage domaine). Pas plus en intro.

## Glossaire touché

CL/IL, MCU, RAM, TinyML, PdM (à créer, cf. S4104), NUCLEO-F439ZI (entrée à créer, remplace STM32N6 obsolète).

## Points ouverts

- Décider si la mention de la cible originale STM32N6 (indisponible → repli F439ZI) mérite
  1 phrase (honnêteté du déroulé) ou disparaît. Recommandation : 1 phrase au ch. 4.
