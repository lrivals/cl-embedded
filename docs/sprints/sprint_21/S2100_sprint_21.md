# Sprint 21 — Tests multi-datasets sur board (Monitoring complet + Pronostia)

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 21 |
| **Semaine** | 27 mai – 6 juin 2026 |
| **Statut** | ⬜ En cours |
| **Priorité globale** | 🔴 Critique — couverture dataset complète sur board + Gap 1 board |
| **Durée estimée totale** | ~22h |
| **Dépendances** | Sprint 20 ✅ (3 modèles C validés, protocole v3, recorder opérationnel) |

---

## Objectif

Après Sprint 20, les 3 modèles sont validés sur :
- **CWRU** → Mahalanobis ✅ + TinyOL ✅
- **Equipment Monitoring** → EWC seulement ✅

Sprint 21 complète la couverture :
1. **Mahalanobis + TinyOL sur Monitoring** — infrastructure déjà là, pas de firmware change
2. **Pronostia sur board** — nécessite réduction 13→5 features (UART protocol limite) + nouveau config + streamer

```
Pronostia features 13→5 (mutual info)   sensor_stream.py --dataset pronostia
             ↓                                          ↓
    configs/pronostia_feature_subset.yaml      configs/board_pronostia.yaml
                              ↓
        E21-01 : Mahalanobis / Monitoring  →  results.json
        E21-02 : TinyOL     / Monitoring  →  results.json
        E21-03 : Mahalanobis/ Pronostia   →  results.json
        E21-04 : EWC        / Pronostia   →  results.json
                              ↓
              comparison_sprint21.json (CWRU + Monitoring + Pronostia)
```

**Critère de succès** :
1. `python scripts/sensor_stream.py --dataset pronostia --dry-run` : passe sans erreur
2. 4 expériences board dans `experiments/exp_S21_0{1..4}/results.json` avec `gap2_latency_compliant: true`
3. `pytest tests/ -v -k pronostia` : vert

---

## Tâches

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Dépendances |
|----|-------|:--------:|:------:|--------------------|-------------|
| S2101 | Feature selection Pronostia 13→5 (mutual info / variance ranking) | 🔴 | ⬜ | `scripts/pronostia_feature_selection.py`, `configs/pronostia_feature_subset.yaml` | — |
| S2102 | Ajout `--dataset pronostia` dans `sensor_stream.py` + `sensor_sim.py` | 🔴 | ✅ | `scripts/sensor_stream.py`, `scripts/sensor_sim.py` | S2101 |
| S2103 | Créer `board_pronostia.yaml` (Mahalanobis + EWC, n_features=5) | 🔴 | ✅ | `configs/board_pronostia.yaml` | S2101 |
| S2104 | E21-01 : Mahalanobis sur Monitoring board (dry-run + live si dispo) | 🔴 | ✅ | `experiments/exp_S21_01/` | infra Sprint 20 |
| S2105 | E21-02 : TinyOL sur Monitoring board (reconstruction anomaly, dry-run) | 🟡 | ✅ | `experiments/exp_S21_02/` | S2104 |
| S2106 | E21-03 : Mahalanobis sur Pronostia board (3 conditions CL, dry-run) | 🔴 | ✅ | `experiments/exp_S21_03/` | S2102, S2103 |
| S2107 | E21-04 : EWC sur Pronostia board (CL cond1→cond2→cond3, λ=400) | 🔴 | ✅ | `experiments/exp_S21_04/` | S2102, S2103 |
| S2108 | RAM profiling `board_pronostia.yaml` (parse_map_file.py, ≤ 64 Ko) | 🟡 | ⬜ | `scripts/parse_map_file.py` | S2103 |
| S2109 | `comparison_sprint21.json` : CWRU vs Monitoring vs Pronostia cross-dataset | 🟡 | ✅ | `experiments/comparison_sprint21.json` | S2104–S2107 |
| S2110 | Tests `test_sensor_stream.py` + `test_board_recorder.py` pour Pronostia | 🟡 | ⬜ | `tests/test_sensor_stream.py`, `tests/test_board_recorder.py` | S2102 |
| S2111 | *(optionnel)* HDC sur Monitoring board — premier test live skeleton S2008 | 🟢 | ⬜ | `experiments/exp_S21_05/` | — |
| S2112 | Docs sprint 21 + `roadmap_phase2.md` update | 🟡 | ⬜ | `docs/sprints/sprint_21/`, `docs/roadmap_phase2.md` | tout |
| S2113 | Protocole expérimental board : cold/warm run, répétitions × 3, vérification état carte | 🟡 | ✅ | `scripts/board_experiment_recorder.py` | S1907 |
| S2114 | Mise à jour `presentation_board_sprint16_20.md` + `.ipynb` avec résultats sprints 16–21 | 🟡 | ✅ | `docs/presentation_board_sprint16_20.md`, `notebooks/presentation_board_sprint16_20.ipynb` | S2113 |

> Détail : [S2101](S2101_pronostia_feature_selection.md) · [S2102–S2103](S2102_sensor_stream_pronostia.md) · [S2104–S2105](S2103_exp_monitoring_allmodels.md) · [S2106–S2107](S2104_exp_pronostia_board.md) · [S2113](S2113_protocole_experimental_board.md) · [S2114](S2114_update_presentation.md)

---

## Contexte technique

### Pourquoi Pronostia nécessite une réduction de features

| Dataset | Features raw | Features board | Raison |
|---------|:-----------:|:--------------:|--------|
| CWRU | 5 | 5 | Compatible direct |
| Monitoring | 4 | 4 | Compatible direct |
| **Pronostia** | **13** | **5** | UART frame limité + modèles firmware compilés pour `N_FEATURES=5` |

Les 13 features Pronostia (6 stats × 2 canaux + position temporelle) doivent être réduites à 5.  
Approche retenue : **sélection par mutual information** vs label `faulty` — interprétable et reproductible.  
Fallback si données absentes : top-5 par expert domain (RMS×2, kurtosis×2, temporal_position).

### Scénario CL Pronostia board

```
Task 0 (cond1) : Bearing1_1 + Bearing1_2  → 1 800 rpm, 4 000 N
Task 1 (cond2) : Bearing2_1 + Bearing2_2  → 1 650 rpm, 4 200 N
Task 2 (cond3) : Bearing3_1 + Bearing3_2  → 1 500 rpm, 5 000 N
```

Commande CL sequence :
```bash
python scripts/sensor_stream.py --dataset pronostia \
    --cl-sequence cond1:200,cond2:200,cond3:200 \
    --update --consolidate-on-task-change --protocol-version 3
```

### Budget RAM — impact Pronostia (identique Monitoring, même n_features=5)

| Modèle | RAM (= Monitoring) | Note |
|--------|:-----------------:|------|
| Mahalanobis | 200 B | MAHA_DIM=5, inchangé |
| EWC | 9 728 B | EWC_IN=5, inchangé |
| TinyOL | 5 800 B | TINYOL_IN=5, inchangé |

Aucun changement firmware requis — seuls les poids pré-entraînés et le streamer Python changent.

---

## Livrable

- 4 fichiers `experiments/exp_S21_0X/results.json` (métriques CL unifiées Phase 1)
- `experiments/comparison_sprint21.json` — tableau comparatif 3 datasets × 3 modèles
- `configs/pronostia_feature_subset.yaml` — indices et noms des 5 features retenues
- `configs/board_pronostia.yaml` — config YAML board Pronostia

---

## Questions ouvertes

- `TODO(arnaud)` : Choix features Pronostia — préférer variance-ranking ou mutual_info_classif vs label ?
- `TODO(arnaud)` : Inclure E21-04 EWC Pronostia dans le tableau comparatif du chapitre 4 (aux côtés de E19-02 EWC Monitoring) ?
- `TODO(fred)` : Les 3 conditions opératoires Pronostia correspondent-elles à un scénario pertinent pour Edge Spectrum ? Quelle condition prioritaire pour la démo P2-06 ?
- `FIXME(gap1)` : Sprint 21 complétera la validation Gap 1 sur Pronostia *sur board* (actuellement PC-only).
