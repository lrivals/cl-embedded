# Sprint 19 — Explications détaillées

> Document de référence pour comprendre les objectifs, tâches et concepts clés du Sprint 19.

---

## Vue d'ensemble

| Champ | Valeur |
|-------|--------|
| **Période** | 1–8 juin 2026 |
| **Objectif** | Valider les 3 modèles CL Phase 1 en C sur NUCLEO-F439ZI |
| **Statut** | 12/13 tâches ✅ — S1903 (poids TinyOL → Flash) en cours |
| **Dépendances** | Sprint 16 (Mahalanobis C, EWC head esquissé) + Sprint 18 (pipeline données UART) |

### Objectif central

Porter les modèles Python (Phase 1) sur le microcontrôleur, et produire des résultats mesurés sur la carte dans le **même format JSON** que les expériences Python. Cela permet de comparer directement les performances PC ↔ MCU.

### Chaîne complète Sprint 19

```
Modèle PyTorch entraîné (Phase 1)
         │
         ▼
scripts/export_weights_c.py
         │
         ▼ (tableau de flottants)
firmware/stm32f4_blink/inc/model_weights.h  (stocké en Flash)
         │
         ▼
Firmware C sur NUCLEO-F439ZI
  ├── mahalanobis.c  — détecteur statistique
  ├── ewc_head.c     — MLP + Fisher EMA + consolidation
  └── tinyol.c       — encodeur autoencoder (inférence seule)
         │  (UART 115200 baud, protocol v3)
         ▼
scripts/sensor_stream.py  — envoie les frames de données
         │
         ▼
scripts/board_experiment_recorder.py
         │
         ▼
experiments/exp_S19_XX/results.json
  (acc_final, avg_forgetting, backward_transfer,
   ram_peak_bytes, inference_latency_ms, n_params)
```

---

## Les 13 tâches du sprint

### Groupe 1 — Modèles C (cœur, priorité critique)

#### S1901 — Validation Mahalanobis C ✅
- Connecte le détecteur Mahalanobis C (Sprint 16) au pipeline de streaming (Sprint 18)
- Vérifie que les prédictions C et Python divergent de moins de 1% sur les mêmes données
- Résultat board : 198 échantillons CWRU, accuracy=68.7%, latence=0.004 ms

#### S1902 — Compléter EWC head C ✅
- Implémente `ewc_consolidate()` : sauvegarde des poids θ* après chaque tâche, mise à jour Fisher par EMA
- Structure `EWCHead` (~9.5 Ko SRAM) sans aucun malloc — tout en place sur la pile ou .bss
- Résultat board : latence=0.004 ms ✅, accuracy=8% ⚠️ (bug init poids → corrigé Sprint 20)

#### S1903 — TinyOL encoder C skeleton 🔄
- Forward pass encodeur seul (pas de décodeur en inference)
- Poids exportés depuis PyTorch → `model_weights.h` (Flash) via `export_weights_tinyol.py`
- Budget : ~5.6 Ko Flash, ~212 B stack pour `tinyol_predict()`

### Groupe 2 — Infrastructure firmware

#### S1904 — Mock data framework C ✅
- Header `tests/mock_data.h` : 10 échantillons par classe × 3 tâches × 5 features, codés en dur
- Permet de faire tourner les tests Unity **sur le PC de développement** sans carte branchée
- Principe : le firmware compilé avec `TEST_MODE=1` utilise ces données synthétiques au lieu de l'UART

#### S1905 — Firmware metrics ✅
Trois structures en C pour mesurer les métriques CL directement sur la carte :

| Structure | RAM | Rôle |
|-----------|:---:|------|
| `OnlineAccuracy` | 8 B | Accuracy glissante par tâche |
| `OnlineAUROC` | 258 B | Wilcoxon-Mann-Whitney, fenêtre W=50 |
| `ForgettingTracker` | 36 B | Pic d'accuracy + valeur courante par tâche → calcule AF et BWT |
| **Total overhead** | **~302 B** | Très faible devant les 256 Ko SRAM |

#### S1906 — Protocol v3 ✅
Évolution du format de réponse UART envoyé par la carte au PC :

| Version | Taille | Contenu |
|---------|:------:|---------|
| v1 (Sprint 16) | 9 B | `pred` · `confidence` · `latency_us` |
| v2 (Sprint 18) | 14 B | + `ram_bytes` · `throughput_ips` · `status` |
| v3 (Sprint 19) | 21 B | remplace ram/throughput par `acc` · `auroc` · `forgetting` |

Le passage à v3 permet à la carte de **rapporter ses propres métriques CL** sans que le PC ait à les recalculer.

### Groupe 3 — Scripts Python

#### S1907 — Experiment recorder ✅
Script central `scripts/board_experiment_recorder.py` — orchestre toute une expérience board :
1. Charge le dataset Python et le config YAML
2. Lance le streaming (`sensor_stream.py`) ou la simulation (dry-run)
3. Collecte les métriques CL (accuracy par tâche, AF, BWT, latence)
4. Sauvegarde `experiments/exp_S19_XX/results.json` au format Phase 1

#### S1908 — Configs YAML modèles embarqués ✅
Trois fichiers de configuration spécifiques board dans `configs/` :
- `board_mahalanobis.yaml` : `threshold_init=3.0`, `ema_alpha=0.05`
- `board_ewc.yaml` : `lr=0.01`, `lambda_ewc=100.0`, `fisher_decay=0.9`
- `board_tinyol.yaml` : `threshold=0.05` (erreur de reconstruction MSE)

Chaque config inclut aussi `ram_model_bytes` et `n_params` pour la validation Gap 2.

#### S1909 — Tests Unity ✅ (28/28 PASS)
Tests sur host (PC) avec `make test` → `TEST_MODE=1` :
- Tests EWC : forward, predict, sgd_step, consolidate
- Tests TinyOL : encoding, reconstruction error
- Vérification absence de malloc via inspection `nm` du binaire

#### S1910 — Tests Python recorder ✅ (13 passed, 11 skipped)
Tests pytest pour le recorder :
- Valide le format JSON (clés obligatoires, types, plages de valeurs)
- Les 11 tests "skipped" requièrent une carte physique branchée

### Groupe 4 — Expériences et profiling

#### S1911 — Expérience E19-01 (Mahalanobis / CWRU) ✅
- 198 échantillons CWRU, 3 tâches
- Accuracy = 68.7% (sans `--update`, donc pas de mise à jour incrémentale activée)
- Latence = 0.004 ms ✅, RAM = ~200 B ✅

#### S1912 — Expérience E19-02 (EWC / Monitoring) ✅
- 300 échantillons Monitoring, 3 tâches
- Accuracy = 8% ⚠️ — bug d'initialisation des poids identifié, corrigé Sprint 20
- Valide que `ewc_consolidate()` tourne sur la vraie carte sans crash

#### S1913 — RAM profiling statique ✅
- Flag `-Wl,-Map` ajouté au Makefile → génère `build/firmware.map`
- Script `scripts/parse_map_file.py` : parse le fichier `.map`, extrait la taille de `.bss` et `.data`
- Valide que la RAM statique totale reste < 64 Ko (critère Gap 2)

---

## Dry-run vs Board — explication détaillée

C'est la distinction la plus importante pour utiliser `board_experiment_recorder.py` correctement.

### Mode Board (mode réel)

```bash
python scripts/board_experiment_recorder.py \
    --model ewc --dataset monitoring \
    --port /dev/ttyACM0 \
    --n-samples 300 --n-tasks 3 --update \
    --output experiments/exp_S19_02
```

**Ce qui se passe :**
1. Le script ouvre une connexion série UART sur `/dev/ttyACM0` (115200 baud)
2. Il charge le dataset Python (ici Monitoring) et découpe en 3 tâches
3. Pour chaque échantillon, il envoie une **frame protocol v3** (21 B) à la carte :
   - Magic `0xABCD` + version + task_id + timestamp
   - Les features en FP32 + le label + des flags bitwise
4. La carte répond avec une frame 21 B : prédiction, confiance, latence, acc, auroc, forgetting
5. Toutes les réponses sont agrégées et sauvegardées en JSON

**Durée** : ~20–30 secondes pour 300 échantillons.  
**Prérequis** : carte branchée, firmware flashé, bon port série.

### Mode Dry-run (simulation)

```bash
python scripts/board_experiment_recorder.py \
    --model ewc --dataset monitoring \
    --dry-run \
    --n-samples 300 --n-tasks 3 \
    --output experiments/exp_S19_02
```

Ce qui se passe — en détail :

#### Étape 1 — Aucune connexion série

La branche `if dry_run:` dans `_run_experiment()` court-circuite complètement `sensor_stream.py`. Aucun port série n'est ouvert, aucun dataset n'est chargé depuis le disque. Tout ce qui suit est purement calculé en mémoire.

#### Étape 2 — Lookup dans `_GENERIC_DRY_RUN_PARAMS`

Le script cherche la clé `"{model}/{dataset}"` dans un dictionnaire codé en dur, par exemple `"ewc/monitoring"` ou `"mahalanobis/cwru"`. Chaque entrée contient 5 paramètres qui calibrent la simulation :

```python
"mahalanobis/cwru": {
    "base_acc":  0.75,   # accuracy attendue sur la tâche courante (diagonale)
    "step_drop": 0.10,   # chute d'accuracy par tâche supplémentaire entraînée
    "lat_lo":    40,     # borne basse de la latence simulée (µs)
    "lat_hi":    80,     # borne haute de la latence simulée (µs)
    "ram_bytes": 1200,   # RAM statique du modèle (constante)
}
```

Ces valeurs ont été calibrées manuellement à partir des mesures réelles Sprint 18–23 pour rester plausibles. Si la combinaison n'existe pas dans le dictionnaire, le script lève une `ValueError` explicite.

#### Étape 3 — Construction de la matrice d'accuracy

C'est le cœur de la simulation. La fonction `_run_generic_dry_run_cl()` construit une matrice triangulaire inférieure `acc_matrix[T×T]` selon la formule :

```text
acc[i, j] = base_acc - step_drop × (i - j) + bruit_uniforme(−0.015, +0.015)
```

- `i` = index de la **tâche en cours d'entraînement** (ligne)
- `j` = index de la **tâche évaluée** (colonne, j ≤ i)
- Quand `i == j` : c'est la diagonale — accuracy sur la tâche courante ≈ `base_acc`
- Quand `j < i` : accuracy sur une tâche **passée** après que de nouvelles tâches ont été apprises → elle chute de `step_drop` à chaque tâche supplémentaire

Exemple concret avec `base_acc=0.75`, `step_drop=0.10`, 3 tâches :

```text
          tâche 0   tâche 1   tâche 2
après T0 [ 0.75      —         —     ]   ← T0 vient d'être apprise
après T1 [ 0.65      0.75      —     ]   ← T0 a chuté de 0.10
après T2 [ 0.55      0.65      0.75  ]   ← T0 a encore chuté, T1 aussi
```

La valeur finale `acc_final` = moyenne de la dernière ligne = (0.55 + 0.65 + 0.75) / 3 ≈ 0.65.  
L'Average Forgetting = chute moyenne des tâches passées ≈ 0.10.

Le bruit aléatoire (seed=42, reproductible) évite que la matrice soit parfaitement régulière.

#### Étape 4 — Génération des résultats bruts par échantillon

Une fois la matrice construite, le script génère une liste de `n_samples` entrées, réparties uniformément entre les `n_tasks` tâches. Pour chaque entrée :

- `pred` est tiré aléatoirement : correct avec probabilité `final_acc[task_j]`, faux sinon
- `latency_us` est tiré uniformément dans `[lat_lo, lat_hi]`
- `ram_bytes` = constante `ram_bytes` du dictionnaire

Cette liste simule ce que `sensor_stream.py` retournerait avec une vraie carte.

#### Étape 5 — Calcul des métriques et sauvegarde JSON

`_build_results_json()` prend la matrice `acc_matrix` et appelle `compute_cl_metrics()` de `src/evaluation/metrics.py` (le même module qu'en Phase 1 Python) pour calculer AF et BWT. Le JSON produit est **structurellement identique** à celui d'une vraie expérience board.

**Durée** : < 1 seconde.  
**Prérequis** : aucun matériel.

### Tableau comparatif

| Aspect | Dry-run | Board |
|--------|:-------:|:-----:|
| Carte NUCLEO requise | ❌ | ✅ |
| Port série (`--port`) | ignoré | obligatoire |
| Données réelles | ❌ synthétiques | ✅ vraies |
| Latence mesurée | simulée (distribution uniforme) | réelle (DWT Cortex-M4) |
| RAM mesurée | constante statique | constante statique (v3 ne rapporte plus la RAM) |
| Durée | < 1 s | 20–30 s |
| Format results.json | identique | identique |
| Cas d'usage | CI, développement, test format JSON | validation finale, mesures publiables |

### Quand utiliser lequel ?

**Dry-run** : toujours en premier, pour vérifier que le script tourne sans erreur, que le JSON est bien formé, et que les métriques ont des valeurs plausibles. C'est aussi ce que fait la CI GitHub Actions.

**Board** : pour les expériences publiables (Sprint papers), les validations Gap 2 (latence < 100 ms, RAM < 64 Ko), et les benchmarks PC vs MCU.

### Cas particulier : EWC et lambda

Pour EWC, le dry-run est plus sophistiqué. Il y a une fonction dédiée `_run_ewc_dry_run_cl()` qui simule deux régimes selon `lambda_ewc` :
- `lambda >= 100` → faible forgetting (EWC actif), AF ≈ 0.05
- `lambda == 0` → oubli catastrophique, AF ≈ 0.30

Cela permet de valider la logique de comparaison EWC vs baseline sans carte.

---

## Budget RAM — NUCLEO-F439ZI (192 Ko SRAM)

| Modèle | Poids (Flash) | SRAM modèle | Stack activation | Metrics overhead | **Total SRAM** | Marge / 64 Ko |
|--------|:------------:|:-----------:|:---------------:|:---------------:|:--------------:|:-------------:|
| Mahalanobis | — | ~128 B | 40 B | ~302 B | **~470 B** | ✅ 63.5 Ko free |
| EWC head | 3 Ko | ~6 Ko Fisher | 200 B | ~302 B | **~9.7 Ko** | ✅ 54 Ko free |
| TinyOL encoder | ~6 Ko | — | 512 B | ~302 B | **~7 Ko** | ✅ 57 Ko free |

> Les poids sont en Flash (lecture seule). La SRAM ne stocke que les activations, Fisher, et les structures de métriques.

---

## Format results.json — unifié Phase 1

Le même format JSON est produit que l'on soit en dry-run ou sur carte, et que l'expérience soit Python (Phase 1) ou firmware (Phase 2). Cela permet de comparer directement.

```json
{
  "exp_id": "S19_01",
  "model": "mahalanobis",
  "dataset": "cwru",
  "platform": "nucleo_f439zi",
  "date": "2026-06-02",

  // Les 6 métriques obligatoires (compatibles evaluate_all.py Phase 1)
  "acc_final": 0.687,
  "avg_forgetting": 0.023,
  "backward_transfer": -0.023,
  "ram_peak_bytes": 470,
  "inference_latency_ms": 0.004,
  "n_params": 30,

  // Champs supplémentaires board
  "n_tasks": 3,
  "n_samples_total": 198,
  "latency_p99_ms": 0.006,
  "throughput_mean_ips": 250000,
  "per_task_acc": {"0": 0.71, "1": 0.65, "2": 0.70},
  "collection_time_s": 24.5,
  "config_snapshot": "experiments/exp_S19_01/config_snapshot.yaml",

  // Validation automatique Gap 2
  "gap2_ram_compliant": true,
  "gap2_latency_compliant": true
}
```

### Les 6 métriques obligatoires

| Métrique | Signification |
|----------|--------------|
| `acc_final` | Accuracy moyenne sur toutes les tâches vues, en fin d'entraînement |
| `avg_forgetting` (AF) | Chute moyenne entre le pic d'accuracy d'une tâche et sa valeur finale |
| `backward_transfer` (BWT) | Impact de l'apprentissage des tâches futures sur les tâches passées (négatif = oubli) |
| `ram_peak_bytes` | RAM maximale utilisée (statique sur board ; tracemalloc sur PC) |
| `inference_latency_ms` | Durée moyenne d'un forward pass + mise à jour (100 runs) |
| `n_params` | Nombre de paramètres entraînables du modèle |

---

## Questions ouvertes à fin de sprint

- `TODO(arnaud)` : Priorité TinyOL skeleton (M1) vs exploration INT8 backprop (Gap 3) dans ce sprint ?
- `TODO(dorra)` : Format `model_weights.h` — array C statique FP32 ou struct nommée ?
- `FIXME(gap2)` : Validation RAM < 64 Ko requise sur Cortex-M55 réel — NUCLEO-F439ZI (192 Ko) est indicatif seulement
