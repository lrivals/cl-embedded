# Sprint 32 — Étude d'impact du seuil de labélisation RUL → faulty

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 32 |
| **Semaine** | 14–20 juillet 2026 |
| **Statut** | ✅ S3201–S3207 implémentés — PC (balayage 60 runs) + **board réelle** (parité EWC+Maha exacte, HDC/TinyOL HW-only) + notebook + tests 16/16 |
| **Priorité globale** | 🔴 Critique — quantifier la sensibilité des 4 modèles au seuil RUL→faulty (réponse au `TODO(arnaud)` de `cmapss_config.yaml:50`) sur PC **et** board réelle |
| **Durée estimée totale** | ~26h |
| **Dépendances** | Loaders RUL existants (CMAPSS/Battery/Pronostia) · infra board Sprints 26-28 ✅ (`sensor_stream.py`, `board_experiment_recorder.py`, parité board↔PC) · `scripts/export_weights_c.py` ✅ |

---

## Contexte et motivation

Le label binaire `faulty` des datasets RUL est aujourd'hui dérivé d'un **seuil fixe codé en dur** :

- **CMAPSS** : `RUL ≤ 30` (`CMAPSS_FAULTY_THRESHOLD = 30`, `src/data/cmapss_loader.py:39`, appliqué L108-109). Le YAML `cmapss_config.yaml:50` contient `faulty_threshold: 30` **mais le code l'ignore** et lit la constante — d'où un `TODO(arnaud): valider seuil RUL ≤ 30`.
- **Battery** : `RUL < 200` (`RUL_FAILURE_THRESHOLD = 200`, `src/data/battery_dataset.py:71`).
- **Pronostia** : label dérivé d'un ratio temporel (`failure_ratio=0.10`), mais le dataset expose un RUL réel via `pronostia_rul_config.yaml` (RUL_CAP=300 s) → peut être seuillé sur RUL.

Le choix du seuil est un **paramètre de conception non validé** : trop restrictif (RUL élevé) → beaucoup de positifs, alerte précoce mais bruitée ; trop permissif (RUL faible) → peu de positifs, détection tardive. Ce sprint **répond au `TODO(arnaud)`** en quantifiant l'effet de 5 seuils sur les performances des 4 modèles (Mahalanobis, HDC, EWC, TinyOL) et sur les contraintes matérielles board.

Décisions validées (utilisateur) :
- **5 seuils** : `30` (référence), restrictifs `50` & `40`, permissifs `20` & `10` (échelle CMAPSS native).
- **3 datasets** : CMAPSS + Battery + Pronostia (les datasets dont `faulty` dérive d'un RUL).
- **Board réelle complète** — pas de dry-run.
- **HW re-profilé par seuil** — pour prouver empiriquement l'invariance RAM/latence au seuil (le seuil n'affecte que les labels, pas l'architecture).

### Échelle des seuils par dataset

Les 5 seuils `{10,20,30,40,50}` sont natifs CMAPSS (cycles, cap=125 → fractions du cap `{8,16,24,32,40}%`). Pour garder une **restrictivité comparable**, on applique les mêmes fractions du `RUL_CAP` de chaque dataset :

| Dataset | RUL_CAP | Seuils dérivés (fractions 8/16/24/32/40 %) | Référence (24 %) |
|---------|--------:|---------------------------------------------|:----------------:|
| CMAPSS | 125 cycles | 10, 20, 30, 40, 50 (natifs) | 30 |
| Pronostia | 300 s | 24, 48, 72, 96, 120 | 72 |
| Battery | ~1134 cycles (ancré sur 200) | 67, 133, 200, 267, 333 | 200 (≈ existant) |

Le mapping exact est stocké dans chaque `config_snapshot.yaml`. La valeur « référence » correspond au seuil 24 % (CMAPSS 30, Pronostia 72, Battery ≈ 200, soit le comportement actuel).

```
TODO(arnaud) cmapss_config.yaml:50            Sprint 32
seuil RUL ≤ 30 non validé          ──▶  S3201 loaders paramétrés (seuil ← config)
                                        S3202 configs/sweep/ (5 seuils × 3 datasets)
                                        S3203 run_threshold_sweep.py (4 modèles)
                                                  ↓
                                        S3204 éval perf + HW PC, par seuil
                                        S3205 éval board réelle, par seuil (parité)
                                                  ↓
                                        S3206 analyse comparative (notebook)
                                        S3207 tests + docs
```

---

## Critères de succès

1. Les 3 loaders lisent le seuil **depuis la config** ; seuil par défaut → **labels identiques à l'existant** (non-régression prouvée par tests).
2. 15 configs de balayage générées sans hyperparamètre modifié en dur (seul le champ seuil varie par dataset).
3. Les 4 modèles entraînés sur `{modèle} × {dataset} × {5 seuils}` ; expériences `exp_S32_*` reproductibles (snapshots).
4. Perf (F1/AUROC/précision/rappel + acc_final/AF/BWT) **et** RAM/latence mesurées **par seuil**, PC et board réelle, avec **parité board↔PC**.
5. Analyse comparative : sensibilité par modèle, seuil optimal par dataset, trade-off restrictif/permissif, **confirmation empirique** que RAM/latence sont invariantes au seuil.
6. Aucun chiffre inventé — champs « à mesurer » tant que non exécuté ; `make test` 0 nouvelle régression.

---

## Tâches

### O1 — Paramétrer le seuil dans les loaders (prérequis bloquant)

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S3201 | Threader le seuil depuis la config dans les 3 loaders. **CMAPSS** : `_load_raw()` accepte `faulty_threshold` (défaut = `CMAPSS_FAULTY_THRESHOLD`), propagé depuis `get_cl_dataloaders`/single-task en lisant `config["data"]["faulty_threshold"]`. **Battery** : idem `rul_failure_threshold` (L118-120). **Pronostia** : nouveau `label_mode: rul_threshold` → `faulty = RUL ≤ seuil` (réutilise le RUL de `pronostia_rul_config.yaml`), mode `failure_ratio` conservé par défaut. Constantes restent valeurs par défaut. | 🔴 | ⬜ | `src/data/cmapss_loader.py`, `src/data/battery_dataset.py`, `src/data/pronostia_dataset.py` | 4h |

### O2 — Générer les configs de balayage

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S3202 | `scripts/generate_threshold_sweep_configs.py` : à partir des configs de base (`cmapss_config.yaml`, `battery_config.yaml`, `pronostia_rul_config.yaml`), émet 15 configs `configs/sweep/{dataset}_thr{XX}.yaml` avec le champ seuil injecté (`faulty_threshold` / `rul_failure_threshold` / `label_mode`+seuil). **Aucun autre hyperparamètre modifié.** | 🔴 | ⬜ | `scripts/generate_threshold_sweep_configs.py`, `configs/sweep/*.yaml` | 2h |

### O3 — Entraîner les 4 modèles sur le balayage

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S3203 | `scripts/run_threshold_sweep.py` : orchestre `{modèle} × {dataset} × {seuil}` en appelant `train_{mahalanobis,hdc,ewc,tinyol}.py` (`--config configs/sweep/...`, `--profile_memory`). Crée les configs modèle×Battery manquantes par analogie aux configs CMAPSS (réutiliser archi, adapter loader/dim d'entrée). Sorties `experiments/exp_S32_{model}_{dataset}_thr{XX}/`. | 🔴 | ⬜ | `scripts/run_threshold_sweep.py`, `configs/*battery*.yaml`, `experiments/exp_S32_*/` | 4h |

### O4 — Évaluation perf + profiling HW PC, par seuil

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S3204 | Pour chaque run : métriques via `anomaly_metrics.py` (AUROC, F1, précision, rappel — critiques car le seuil change le ratio de positifs) + `metrics.py` (acc_final, AF, BWT). **RAM/latence re-profilées par seuil** via `profile_memory.py` + `memory_profiler.py` ; MACs via `compute_cost.py`. Consolider `experiments/exp_S32_*/results/`. | 🔴 | ⬜ | `experiments/exp_S32_*/results/*.json` | 3h |

### O5 — Évaluation board réelle, par seuil

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S3205 | Pour chaque (modèle, dataset, seuil) : export poids (`export_weights_c.py`/`export_weights_tinyol.py`) → `sensor_stream.py --port /dev/ttyACM0` → `board_experiment_recorder.py` (latence DWT µs, RAM `.bss`). **Parité board↔PC** des prédictions vérifiée (cf. Sprints 26-28) ; latence < 100 ms (Gap 2). RAM profiling obligatoire. | 🔴 | ⬜ | `experiments/exp_S32_board_*/` | 5h |

### O6 — Analyse comparative

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S3206 | `notebooks/cl_eval/threshold_impact/comparison.ipynb` : courbes F1/AUROC/précision/rappel vs seuil (par modèle/dataset), heatmaps, ratio de positifs vs seuil, tables PC↔board. Analyse écrite : sensibilité par modèle, seuil optimal par dataset, trade-off restrictif/permissif, confirmation invariance HW au seuil. | 🟡 | ⬜ | `notebooks/cl_eval/threshold_impact/comparison.ipynb` | 4h |

### O7 — Tests + docs

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S3207 | `tests/test_threshold_sweep.py` : paramétrage seuil des 3 loaders (non-régression seuil par défaut), génération configs, **monotonie du ratio de positifs** avec le seuil. Tests Unity firmware restent verts (`make test`). MAJ `docs/roadmap_phase2.md` + statut `CLAUDE.md`. Invoquer `graphify_sprint_update`. | 🟢 | ⬜ | `tests/test_threshold_sweep.py`, `docs/roadmap_phase2.md`, `CLAUDE.md` | 4h |

---

## Ordre d'exécution recommandé

```
S3201 (loaders ← seuil config)
        ↓
S3202 (generate_threshold_sweep_configs.py → configs/sweep/)
        ↓
S3203 (run_threshold_sweep.py — 4 modèles × 3 datasets × 5 seuils)
        ↓
S3204 (éval perf + HW PC, par seuil)
        ↓
S3205 (export poids → board réelle, parité board↔PC, latence/RAM)
        ↓
S3206 (notebook analyse comparative)
        ↓
S3207 (tests + docs + graphify)
```

---

## Nomenclature des expériences

| Exp ID | Modèle | Dataset | Seuil | Mesure |
|--------|--------|---------|:-----:|--------|
| exp_S32_ewc_cmapss_thr30 | EWC | CMAPSS | 30 | F1/AUROC/préc/rappel + RAM/lat PC |
| exp_S32_hdc_pronostia_thr72 | HDC | Pronostia | 72 | idem |
| exp_S32_maha_battery_thr200 | Mahalanobis | Battery | 200 | idem |
| … (4 modèles × 3 datasets × 5 seuils = 60) | | | | |
| exp_S32_board_{model}_{dataset}_thr{XX} | board | (3 datasets) | (5 seuils) | latence DWT µs, .bss B, parité |

---

## Budget mémoire firmware estimé

| Composant | RAM .bss | Notes |
|-----------|:--------:|-------|
| Modèles board (EWC/HDC/Maha/TinyOL) | inchangé vs Sprints 26-28 | **le seuil n'affecte pas l'architecture** |
| Données de seuil | 0 B | seuil = paramètre de labélisation côté PC, jamais embarqué |
| **Attendu** | **invariant au seuil** | S3205 le **prouve empiriquement** (re-profiling par seuil) |

---

## Notes d'implémentation

**S3201 non-régression** : la constante (`CMAPSS_FAULTY_THRESHOLD`, `RUL_FAILURE_THRESHOLD`) reste la valeur par défaut du paramètre. Tests : config sans champ seuil OU seuil = constante → labels bit-à-bit identiques à l'existant. Attention à l'opérateur : CMAPSS/Pronostia `RUL ≤ seuil` (inclusif), Battery `RUL < seuil` (exclusif) — conserver l'opérateur natif de chaque dataset.

**S3202 / règle CLAUDE.md** : ne jamais modifier d'hyperparamètres dans le code source — tout passe par les configs YAML. Le script de génération ne touche **que** le champ seuil entre configs d'un même dataset.

**S3203 Battery** : si une combinaison modèle×Battery n'a pas de config existante, la créer par analogie aux configs CMAPSS (même archi, adapter `input_dim`/loader). Ne pas inventer d'hyperparamètres exotiques.

**S3205 parité** : suivre le protocole des Sprints 26-28 (parité numérique board↔PC exacte des prédictions). Le seuil ne change que les labels de référence côté PC ; la sortie du modèle board doit rester identique à la prédiction PC sur la même entrée.

**HW invariant au seuil** : le résultat attendu est que RAM/latence ne dépendent pas du seuil. Le re-profiling par seuil (choix utilisateur) vise à le **démontrer**, pas à découvrir une variation. Si une variation apparaît, c'est un bug à investiguer.

---

## Questions ouvertes

- `TODO(arnaud)` **résolu par ce sprint** : le seuil RUL ≤ 30 de CMAPSS est-il optimal ? → réponse quantifiée par S3204/S3206 (seuil optimal par dataset).
- `TODO(arnaud)` : la mise à l'échelle par fraction du `RUL_CAP` (8/16/24/32/40 %) est-elle la bonne façon de rendre les seuils comparables entre datasets, ou faut-il un ancrage par taux de positifs cible ?
- `TODO(fred)` : du point de vue maintenance industrielle, quel trade-off privilégier — alerte précoce (seuil permissif, plus de faux positifs) vs détection tardive fiable (seuil restrictif) ?
- `FIXME(gap2)` : confirmer que latence board < 100 ms reste vraie pour tous les seuils (attendu : invariant).

---

## Livrables

1. Loaders paramétrés `src/data/{cmapss_loader,battery_dataset,pronostia_dataset}.py` (seuil ← config)
2. `scripts/generate_threshold_sweep_configs.py` + `configs/sweep/*.yaml` (15 configs)
3. `scripts/run_threshold_sweep.py` + `experiments/exp_S32_*/` (60 runs PC)
4. `experiments/exp_S32_board_*/` (board réelle, parité, latence/RAM par seuil)
5. `notebooks/cl_eval/threshold_impact/comparison.ipynb` + analyse écrite
6. `tests/test_threshold_sweep.py` + MAJ `docs/roadmap_phase2.md` + `CLAUDE.md`

---

## Bilan (à compléter)

| Tâche | Statut | Temps réel | Notes |
|-------|:------:|:----------:|-------|
| S3201 loaders ← seuil config | ✅ | — | CMAPSS déjà conforme ; Battery + Pronostia (`rul_threshold`) threadés ; 62/62 tests loaders PASS |
| S3202 generate_threshold_sweep_configs.py | ✅ | — | 15 configs ; base Pronostia = `pronostia_config.yaml` (décision utilisateur) |
| S3203 run_threshold_sweep.py | ✅ | — | **60/60 runs OK** ; Battery câblé dans les 4 scripts + normalizer + 3 configs ; fix TinyOL×Battery oto_head dim |
| S3204 éval perf + HW PC | ✅ | — | `positive_ratio` + perf/HW consolidés (`exp_S32_sweep_summary.json`) ; gradients monotones vérifiés |
| S3205 éval board réelle | ✅ | — | Modèles réf. board 5-feat (`train_board_reference.py`) ; firmware `model_weights_ewc.h` + chargement `g_ewc_head` (fallback Xavier, 0 régression) ; `export_weights_c.py --ewc-head` ; `sensor_stream.py --dump-samples` + battery ; driver `run_board_threshold_sweep.py` (1 flash/cellule). **Parité EWC+Maha exacte** ; HDC/TinyOL HW-only (parité N/A par construction). `.bss=104 596 B` invariant, latences ≪ 100 ms. CMAPSS 10/10 parité ; matrice complète → `exp_S32_board_sweep_summary.json` |
| S3206 notebook analyse | ✅ | — | `notebooks/cl_eval/threshold_impact/comparison.ipynb` (perf/HW vs seuil, heatmaps, invariance HW board, tables/parité PC↔board) |
| S3207 tests + docs | ✅ | — | `tests/test_threshold_sweep.py` 16/16 PASS ; Unity 94/96 (2 TinyOL préexistants) ; roadmap + CLAUDE.md + graphify |
