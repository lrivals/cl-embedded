# Sprint 52 — Correctifs de mesure carte (action 2.6 de l'audit S4110)

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 52 |
| **Date** | 30 juillet 2026 |
| **Statut** | ✅ implémenté |
| **Origine** | `docs/context/actions_restantes_resultats.txt` — ordre d'attaque, action **2.6** (priorité 1 : « correction d'un chiffre FAUX du manuscrit ») |
| **Périmètre** | Sélection de modèle UART TinyOL + portage des poids TinyOL à dimension quelconque + re-mesure des cellules carte concernées (S35, S32) |

## Pourquoi

Le manuscrit publie une ligne « TinyOL carte » (tableau 5.1, chapitre 5) qui **ne mesure pas
TinyOL**. `scripts/sensor_stream.py` n'attribuait aucun flag à `--model tinyol` : la trame
partait avec `model_flags = 0`, c'est-à-dire le chemin d'inférence **par défaut** du firmware
(`pipeline.c`, branche `else`) — le détecteur de **Mahalanobis**. Les JSON
`exp_S35_board_{5feat,all}_tinyol_*` sont, de ce fait, numériquement identiques à leurs
homologues `mahalanobis` sur les 10 paires concernées.

Le défaut ne se limitait pas au flag manquant : la route TinyOL n'aurait de toute façon
tourné qu'à **poids nuls** dans la majorité des cellules (garde figée à 5 features, aucun
export de poids TinyOL dans les drivers). Les deux moitiés sont corrigées ici.

## Tâches

| ID | Tâche | Statut |
|----|-------|--------|
| S5201 | Flag UART TinyOL + poids à dim k + re-mesure carte | ✅ |

## Livrables

- Correctifs : `scripts/sensor_stream.py`, `firmware/stm32f4_blink/{src/pipeline.c,src/tinyol.c}`,
  `scripts/export_weights_tinyol.py`, `scripts/train_board_reference.py`
- Drivers : `scripts/run_feature_condition_board.py`, `scripts/run_board_threshold_sweep.py`
- Tests : `tests/test_tinyol_board_flag.py`, `firmware/stm32f4_blink/tests/test_tinyol.c`
- Mesures : `experiments/exp_S35_board_*_tinyol_*`, `experiments/exp_S32_board_tinyol_*`
- Aval : `src/figures/catalogs/manuscrit_final.py`, manuscrit ch. 5

## Suite

Action suivante de l'ordre d'attaque : **2.2 — sub-INT8 en régime online**. L'action 1.1
(campagne énergie LPM01A) reste **bloquée** : la sonde X-NUCLEO-LPM01A n'est pas disponible.
