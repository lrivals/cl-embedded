# Fiche de cadrage — Ch. 4 Méthodologie (~6 p., cible md `04_methodologie.md`)

## Messages clés

1. **Datasets** : tableau synthétique des 6 (type, label, scénario CL, rôle), puis description
   détaillée des 3 datasets focus : **CMAPSS** (D5, RUL, domain-incremental FD001–FD004),
   **Pronostia** (D4, class-incremental, données réelles de roulements),
   **Monitoring** (D2, domain-incremental par type d'équipement). D1/D3/D6 → annexe.
2. **Modèles** : M1 TinyOL (backbone gelé + tête OtO), M2 EWC Online MLP (+ variante INT8),
   M3 HDC, M4 Mahalanobis (baseline non supervisée). Tailles paramétriques et empreintes.
3. **Pipeline PC → C** : implémentation Python (PyTorch), export des poids par
   `scripts/export_weights_c.py` → headers générés (jamais à la main), firmware C
   (`firmware/stm32f4_blink/`), protocole UART v3 (trame capteur → prédiction), streaming hôte
   `scripts/sensor_stream.py`.
4. **Protocole de mesure board** (cœur méthodologique — c'est ce qui rend les gaps « mesurés ») :
   - latence : compteur de cycles **DWT** (P50/P99) ;
   - RAM statique : `.bss + .data` par symboles linker (`_ebss - _sbss`), **contiguïté prouvée**
     (une seule section de sortie) + vérification croisée 4 sources (`arm-none-eabi-size`,
     `objdump -h`, symboles, runtime) ;
   - pile : watermark (peinture de stack, `scripts/measure_stack_watermark.py`) ;
   - intégrité : CRC sur chaque trame UART.
5. **Méthodologie de parité PC↔board** : mêmes poids exportés, mêmes échantillons rejoués,
   comparaison prédiction-à-prédiction (exacte en gelé, approchée en online float32 vs float64) —
   présentée ici comme méthode, résultats au ch. 6.
6. **Métriques** : acc_final, F1 (justifier vs accuracy trompeuse), AF/BWT, AUROC, RMSE_RUL,
   ram_peak, latence — renvoi `src/evaluation/`.
7. **Organisation du travail (méthodologie de conduite de projet)** — à intégrer au chapitre :
   - **Méthode agile par sprints** : découpage du stage en ~40 sprints courts, chaque sprint =
     objectif ciblé + tâches numérotées `SxxYY`, fiche de sprint versionnée dans `docs/sprints/`
     (`SxxYY_*.md`) et roadmap (`docs/roadmap_phase2.md`). Traçabilité incrémentale des décisions.
   - **Documentation automatique des expériences (reproductibilité)** : chaque run produit un
     dossier `experiments/exp_XXX/` avec `config_snapshot.yaml` + JSON de résultats horodatés,
     seed fixé (`utils/reproducibility.py::set_seed(42)`), aucun chiffre hardcodé (tout sort d'un
     script), poids C toujours générés (`scripts/export_weights_c.py`, jamais édités à la main).
   - **Points mensuels avec les encadrants** : présentations de synthèse (slides) à échéance
     mensuelle (Arnaud Dion, ISAE-SUPAERO ; ENAC ; Edge Spectrum), pilotant les décisions de
     cadrage et les priorités des sprints suivants.

## Contenu source (vérifié présent dans le dépôt)

- `docs/context/ram_measurement.md` — protocole RAM complet (contiguïté .bss, 4 sources concordantes).
- `docs/models/{tinyol_spec,ewc_mlp_spec,hdc_spec}.md`, `src/models/unsupervised/`.
- `firmware/stm32f4_blink/` (pipeline.c, profiling.c), `scripts/export_weights_c.py`,
  `scripts/sensor_stream.py`.
- Loaders : `src/data/{cmapss_loader,pronostia_dataset,monitoring...}.py` + configs YAML.

## Chiffres (specs, pas de résultats)

- Dimensions des modèles board de référence (ex. EWC 5→32→16→2, `scripts/train_board_reference.py`).
- `.bss` build par défaut : **105 036 B** + `.data` 460 B — source `docs/context/ram_measurement.md`
  (tableau des 4 sources concordantes). `[à confirmer — build courant S39/S40]`.
- Budget : 256 Ko SRAM (192 + 64 CCM).

## Figures prévues (S4109 ; existantes à réutiliser)

- `docs/figures/ram_measurement/fig_memory_map.png` (carte mémoire) et/ou `fig_shared_frame.png` —
  choisir 1–2 max.
- Schéma pipeline PC→board (existe dans `graphify-out/` / présentation ; sinon à créer).

## Refs bib

`Saxena2008` (CMAPSS), `Nectoux2012` (Pronostia), `Kirkpatrick2017`, `Ren2021`, `Benatti2019`.
**À ajouter (S4103)** : réf CWRU et `Lessmeier2016` (Paderborn) pour le tableau des 6 datasets,
réf dataset Monitoring/Pump (Kaggle — citation URL en note de bas de page acceptable).

## Glossaire touché

DWT, `.bss`, watermark, CRC, UART, FPU, DSP, SRAM, CCM, parité (entrées à créer, cf. S4104).

## Points ouverts

- Mention repli STM32N6 → NUCLEO-F439ZI : 1 phrase ici (recommandé ch. 1, décidé en S4106).
- Choix final des 1–2 figures RAM parmi les 13 de `docs/figures/ram_measurement/`.
