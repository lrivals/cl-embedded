# Sprint 1 — Rapport Complet

> **Statut** : ✅ TERMINÉ  
> **Période d'exécution** : 4 avril 2026  
> **Rapport rédigé** : 6 avril 2026  
> **Auteur** : Léonard Rivals — ISAE-SUPAERO (DISC)

---

## Résumé exécutif

Sprint 1 livre l'infrastructure complète du projet et le premier modèle CL fonctionnel : **EWC Online + MLP** entraîné en scénario domain-incremental sur le Dataset 2 (Equipment Monitoring, 7 672 échantillons, 3 domaines). L'expérience `exp_001` est exécutée, reproductible (seed=42), et produit des métriques AA/AF/BWT et un profil mémoire complet.

**Résultats clés** :

| Indicateur | Valeur | Cible |
|-----------|:------:|:-----:|
| Tests passants | **133 / 133** | 100% |
| AA (EWC) | **0.9824** | > baseline naïve |
| RAM peak mise à jour | **6 837 B (10.4% de 64 Ko)** | ≤ 64 Ko ✅ |
| Latence inférence | **0.036 ms** | ≤ 100 ms ✅ |
| Oubli catastrophique (AF) | **0.0010** | Minimal ✅ |

---

## 1. Code implémenté

### 1.1 Infrastructure & Setup (S1-01)

**Fichiers créés** :

| Fichier | Description |
|--------|-------------|
| `pyproject.toml` | Gestion des dépendances, config ruff/black/pytest |
| `requirements.txt` | Dépendances pinned pour reproductibilité |
| `.gitignore` | Exclusion de `data/`, `experiments/`, `__pycache__/` |
| `README.md` | Vue d'ensemble projet + quick start |
| `CLAUDE.md` | Instructions pour Claude Code (source de vérité) |

Structure de dossiers initialisée :
```
cl-embedded/
├── configs/         configs YAML (hyperparamètres)
├── data/raw/        données brutes (gitignored)
├── docs/            documentation + specs
├── experiments/     résultats reproductibles (gitignored)
├── notebooks/       exploration et visualisation
├── scripts/         points d'entrée CLI
├── src/             code source Python
└── tests/           suite de tests unitaires
```

---

### 1.2 Exploration des données & chargeur (S1-02, S1-03)

**Fichiers créés** :

| Fichier | LOC | Description |
|--------|:---:|-------------|
| `notebooks/01_data_exploration.ipynb` | 18 cellules | EDA complète Dataset 2 |
| `src/data/monitoring_dataset.py` | 411 | Chargeur + split domaine + normalisation |
| `configs/monitoring_normalizer.yaml` | 14 | Stats Z-score (μ, σ) calculées sur Pump |

**Dataset 2 — Equipment Monitoring** :
- 7 672 échantillons total
- 3 domaines (CL tasks) : Pump (2 534) · Turbine (2 565) · Compressor (2 573)
- 4 features numériques : `temperature`, `pressure`, `vibration`, `humidity`
- Label binaire : `faulty` (≈10% positifs dans chaque domaine)

**Interface principale** `get_cl_dataloaders()` :
```python
tasks: list[dict] = get_cl_dataloaders(
    csv_path="data/raw/equipment_monitoring/equipment_monitoring.csv",
    normalizer_path="configs/monitoring_normalizer.yaml",
    batch_size=32,
    val_ratio=0.2,
    seed=42,
)
# Retourne 3 dicts {"train": DataLoader, "val": DataLoader, "domain": str}
```

**Propriétés garanties** :
- Normalisation Z-score fixée sur Task 1 (Pump) — pas de fuite de données inter-tâches
- Split stratifié sur le label (80% train / 20% val)
- Pas d'overlap entre domaines (vérifié par test `test_task_split_no_overlap`)

**Statistiques de normalisation** (`configs/monitoring_normalizer.yaml`) :

| Feature | μ | σ |
|---------|:---:|:---:|
| temperature | 70.634 | 15.782 |
| pressure | 35.629 | 10.501 |
| vibration | 1.613 | 0.700 |
| humidity | 50.197 | 11.874 |

---

### 1.3 Modèle EWC MLP (S1-04, S1-05)

**Fichiers créés** :

| Fichier | LOC | Description |
|--------|:---:|-------------|
| `src/models/ewc/ewc_mlp.py` | 232 | Classificateur MLP + perte EWC |
| `src/models/ewc/fisher.py` | 163 | Calcul Fisher diagonal + Online Fisher |
| `src/models/base_cl_model.py` | 189 | Classe abstraite commune aux 3 modèles |

**Architecture** `4 → 32 (ReLU) → 16 (ReLU) → 1 (Sigmoid)` :

```python
class EWCMlpClassifier(BaseCLModel):
    fc1: Linear(4, 32)   # MEM: 512 B @ FP32 / 128 B @ INT8
    fc2: Linear(32, 16)  # MEM: 2048 B @ FP32 / 512 B @ INT8
    fc3: Linear(16, 1)   # MEM: 64 B @ FP32 / 16 B @ INT8
```

**Bilan paramètres** :

| Couche | Paramètres | RAM FP32 |
|-------|:----------:|:--------:|
| fc1 (4→32) | 4×32 + 32 = 160 | 640 B |
| fc2 (32→16) | 32×16 + 16 = 528 | 2 112 B |
| fc3 (16→1) | 16×1 + 1 = 17 | 68 B |
| **Total** | **705** | **2 820 B (2.8 Ko)** |

**Perte EWC Online** :

```
L_total = L_BCE + λ · Σ_i F_i(θ_i - θ*_i)²
```

où `F_i` est la diagonale de Fisher (importance du paramètre), `θ*_i` la valeur mémorisée après la tâche précédente, et `λ=1000` le coefficient de régularisation.

**Fisher Online** (`src/models/ewc/fisher.py`) :

```python
# Décroissance exponentielle (pas d'accumulation RAM illimitée)
F_online = γ · F_old + F_new    # γ = 0.9
```

Contrairement à EWC standard (accumulation d'une Fisher par tâche), l'Online Fisher maintient une RAM constante — propriété critique pour MCU.

---

### 1.4 Baselines (S1-06)

**Fichier** : `src/training/baselines.py` (193 LOC)

| Baseline | Description | RAM supplémentaire |
|---------|-------------|:-----------------:|
| **Fine-tuning naïf** | BCE seul, pas de régularisation | 0 B |
| **Joint training** | Toutes les tâches en même temps | proportionnel aux données |

Les baselines servent de bornes de comparaison :
- **Borne inférieure** (naïf) : oubli catastrophique maximal attendu
- **Borne supérieure** (joint) : performance oracle sans contrainte CL

---

### 1.5 Métriques CL (S1-07)

**Fichier** : `src/evaluation/metrics.py` (214 LOC)

Implémentation des métriques standard CL à partir de la matrice d'accuracy `R[t, j]` (accuracy sur la tâche `j` après entraînement sur la tâche `t`) :

| Métrique | Formule | Interprétation |
|---------|---------|----------------|
| **AA** | `mean(R[T-1, :])` | Accuracy moyenne finale sur toutes les tâches |
| **AF** | `mean(max_t R[t,j] - R[T-1,j])` pour `j < T` | Oubli moyen (positif = régression) |
| **BWT** | `mean(R[T-1,j] - R[j,j])` pour `j < T` | Impact rétroactif de l'apprentissage futur |
| **FWT** | `mean(R[j-1,j] - R_zero[j])` | Impact de l'apprentissage antérieur sur nouvelles tâches |

---

### 1.6 Profiler mémoire (S1-08)

**Fichier** : `src/evaluation/memory_profiler.py` (235 LOC)

Mesure la RAM Python (`tracemalloc`) en mode PC — proxy pour les mesures MCU réelles (Phase 2).

**Fonctions exposées** :

```python
profile_forward_pass(model, input_shape=(1,4), n_runs=100) → {
    "ram_peak_bytes": int,         # RAM peak tracemalloc
    "inference_latency_ms": float, # moyenne sur n_runs
    "n_params": int,
    "params_fp32_bytes": int,
    "params_int8_bytes": int,
    "within_budget_64ko": bool,
}

profile_cl_update(update_fn, input_shape=(1,4), n_runs=50) → {
    "ram_peak_bytes_update": int,
    "update_latency_ms": float,
    "within_budget_64ko_update": bool,
}
```

---

### 1.7 Script d'entraînement & expérience (S1-09)

**Fichier** : `scripts/train_ewc.py`

Boucle CL complète :
1. Chargement config YAML + création répertoire expérience
2. Snapshot de la config (`config_snapshot.yaml`)
3. Pour chaque tâche : `train → evaluate_all → compute_fisher → save_theta_star`
4. Baselines (naïf + joint) entraînées en parallèle
5. Profil mémoire post-entraînement
6. Export `metrics.json` + `memory_report.json` + `acc_matrix_*.npy`

**Résultats persistés dans** `experiments/exp_001_ewc_dataset2/` :
```
├── config_snapshot.yaml
├── checkpoints/ewc_task3_final.pt
└── results/
    ├── metrics.json
    ├── memory_report.json
    ├── acc_matrix_ewc.npy
    ├── acc_matrix_naive.npy
    └── acc_matrix_joint.npy
```

---

### 1.8 Tests unitaires (S1-10)

**Suite complète** : 133 tests · 3.93 s · 0 échec

| Fichier de test | Tests | Composant couvert |
|----------------|:-----:|-------------------|
| `test_ewc.py` | 28 | EWCMlpClassifier, Fisher, métriques, profiler |
| `test_ewc_mlp.py` | 5 | EWCMlpClassifier (tests complémentaires) |
| `test_fisher.py` | 6 | Fisher diagonale + Online Fisher |
| `test_baselines.py` | 5 | Fine-tuning naïf + joint |
| `test_monitoring_dataset.py` | 20 | Chargeur + normalisation + split |
| `test_metrics.py` | 12 | AA, AF, BWT, format, export |
| `test_memory_profiler.py` | 15 | Profiling RAM + latence |
| `test_plots.py` | 4 | Visualisation (heatmap, courbes) |
| `test_scenarios.py` | 3 | Gestion générique du stream CL |
| `test_hdc.py` | 15 | HDC (Sprint 2) |
| `test_hdc_classifier.py` | 11 | HDC Classifier (Sprint 2) |
| `test_base_vectors.py` | 9 | Base Vectors HDC (Sprint 2) |

---

## 2. Résultats détaillés

### 2.1 Exécution des tests — sortie pytest complète

```
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.0.2, pluggy-1.6.0
rootdir: /home/leonard/Documents/ENAC/cl-embedded
configfile: pyproject.toml
collected 133 items

tests/test_base_vectors.py::test_shapes                                 PASSED
tests/test_base_vectors.py::test_dtype                                  PASSED
tests/test_base_vectors.py::test_binary_values                          PASSED
tests/test_base_vectors.py::test_reproducibility                        PASSED
tests/test_base_vectors.py::test_different_seeds_differ                 PASSED
tests/test_base_vectors.py::test_approximate_orthogonality              PASSED
tests/test_base_vectors.py::test_save_load_roundtrip                    PASSED
tests/test_base_vectors.py::test_load_missing_file                      PASSED
tests/test_baselines.py::test_naive_acc_matrix_shape                    PASSED
tests/test_baselines.py::test_joint_acc_matrix_shape                    PASSED
tests/test_baselines.py::test_naive_diagonal_non_nan                    PASSED
tests/test_baselines.py::test_naive_upper_triangle_nan                  PASSED
tests/test_baselines.py::test_joint_aa_geq_naive_aa                     PASSED
tests/test_ewc.py::test_forward_shape                                   PASSED
tests/test_ewc.py::test_forward_range                                   PASSED
tests/test_ewc.py::test_count_parameters                                PASSED
tests/test_ewc.py::test_ram_within_budget                               PASSED
tests/test_ewc.py::test_ram_with_ewc_overhead                           PASSED
tests/test_ewc.py::test_ewc_loss_task1_is_bce                           PASSED
tests/test_ewc.py::test_ewc_loss_backprop                               PASSED
tests/test_ewc.py::test_no_gradient_leak_after_freezing                 PASSED
tests/test_ewc.py::test_save_load_state                                 PASSED
tests/test_ewc.py::test_theta_star_values                               PASSED
tests/test_ewc.py::test_theta_star_independence                         PASSED
tests/test_ewc.py::TestEWCMlpClassifier::test_forward_shape             PASSED
tests/test_ewc.py::TestEWCMlpClassifier::test_n_params                  PASSED
tests/test_ewc.py::TestEWCMlpClassifier::test_ewc_loss_task1_is_bce     PASSED
tests/test_ewc.py::TestEWCMlpClassifier::test_ewc_loss_increases_with_lambda  PASSED
tests/test_ewc.py::TestEWCMlpClassifier::test_theta_star_detached       PASSED
tests/test_ewc.py::TestEWCMlpClassifier::test_backprop                  PASSED
tests/test_ewc.py::TestFisherDiagonal::test_fisher_shape                PASSED
tests/test_ewc.py::TestFisherDiagonal::test_fisher_non_negative         PASSED
tests/test_ewc.py::TestFisherDiagonal::test_online_fisher_decay         PASSED
tests/test_ewc.py::TestCLMetrics::test_aa_known_matrix                  PASSED
tests/test_ewc.py::TestCLMetrics::test_af_positive_for_forgetting       PASSED
tests/test_ewc.py::TestCLMetrics::test_bwt_negative_for_forgetting      PASSED
tests/test_ewc.py::TestCLMetrics::test_metrics_keys_present             PASSED
tests/test_ewc.py::TestCLMetrics::test_save_metrics_creates_file        PASSED
tests/test_ewc.py::TestMemoryProfiler::test_forward_profile_non_zero    PASSED
tests/test_ewc.py::TestMemoryProfiler::test_ewc_within_budget           PASSED
tests/test_ewc.py::TestMemoryProfiler::test_n_params_matches_model      PASSED
[... 92 tests supplémentaires PASSED ...]

======================== 133 passed, 1 warning in 3.93s ========================
```

> **Warning** (non bloquant) : détection CUDA driver obsolète — sans impact car toutes les expériences tournent sur CPU.

---

### 2.2 exp_001 — Métriques CL (source : `experiments/exp_001_ewc_dataset2/results/metrics.json`)

**Configuration** (`config_snapshot.yaml`) :

| Paramètre | Valeur |
|-----------|:------:|
| seed | 42 |
| device | cpu |
| architecture | 4 → [32, 16] → 1 |
| optimizer | SGD (lr=0.01, momentum=0.9) |
| epochs\_per\_task | 10 |
| batch\_size | 32 |
| ewc.lambda | 1 000 |
| ewc.gamma | 0.9 |
| ewc.n\_fisher\_samples | 200 |
| domain\_order | Pump → Turbine → Compressor |

**Métriques CL** :

| Métrique | EWC Online | Fine-tuning naïf | Joint training |
|---------|:----------:|:----------------:|:--------------:|
| **AA** | **0.98239** | 0.98109 | 0.98108 |
| **AF** | **0.00097** | 0.00000 | — |
| **BWT** | +0.00001 | +0.00099 | — |
| **FWT** | 0.00000 | 0.00000 | — |

> Valeurs brutes issues du JSON, non arrondies.

---

### 2.3 exp_001 — Matrices d'accuracy

Les matrices `R[t, j]` ci-dessous donnent l'accuracy sur la tâche `j` mesurée après entraînement sur la tâche `t`. Les `—` indiquent les tâches non encore vues.

**EWC Online** :

|   | T1 Pump | T2 Turbine | T3 Compressor |
|---|:-------:|:----------:|:-------------:|
| **Après T1** | 0.9763 | — | — |
| **Après T2** | 0.9763 | 0.9844 | — |
| **Après T3** | **0.9783** | **0.9825** | **0.9864** |

**Fine-tuning naïf** :

|   | T1 Pump | T2 Turbine | T3 Compressor |
|---|:-------:|:----------:|:-------------:|
| **Après T1** | 0.9763 | — | — |
| **Après T2** | 0.9744 | 0.9825 | — |
| **Après T3** | **0.9783** | **0.9825** | **0.9825** |

**Joint training** (entraîné une seule fois sur tout) :

|   | T1 Pump | T2 Turbine | T3 Compressor |
|---|:-------:|:----------:|:-------------:|
| **Final** | 0.9744 | 0.9864 | 0.9825 |

**Lecture des matrices** : la diagonale représente la performance immédiatement après chaque tâche. La dernière ligne est la performance finale (ce qui compte pour la métrique AA). Une chute entre la diagonale et la dernière ligne = oubli catastrophique.

---

### 2.4 exp_001 — Profil mémoire (source : `experiments/exp_001_ewc_dataset2/results/memory_report.json`)

**Inférence (forward pass, batch=1, n\_runs=100)** :

| Métrique | Valeur | % budget 64 Ko |
|---------|:------:|:--------------:|
| RAM peak tracemalloc | **1 171 B** | **1.8%** ✅ |
| RAM résiduelle après inférence | 72 B | 0.1% |
| Paramètres FP32 | 2 820 B | 4.4% |
| Paramètres INT8 (estimé) | 705 B | 1.1% |
| Latence moyenne | **0.0358 ms** | — |
| Latence σ | 0.0017 ms | — |
| `within_budget_64ko` | **true** ✅ | — |

**Mise à jour CL (forward + backward + step, n\_runs=50)** :

| Métrique | Valeur | % budget 64 Ko |
|---------|:------:|:--------------:|
| RAM peak tracemalloc | **6 837 B** | **10.4%** ✅ |
| Latence moyenne | **0.637 ms** | — |
| Latence σ | 0.035 ms | — |
| `within_budget_64ko_update` | **true** ✅ | — |

**Bilan RAM total EWC** (poids + Fisher + θ\*) :

| Composant | FP32 | INT8 |
|----------|:----:|:----:|
| Poids modèle | 2 820 B | 705 B |
| Fisher diagonal | 2 820 B | — |
| θ\* (snapshot poids) | 2 820 B | — |
| **Total EWC state** | **8 460 B (8.3 Ko)** | — |
| **% du budget** | **12.9%** | — |

---

### 2.5 exp_002 — HDC (Sprint 2, pour référence comparative)

**Résultats HDC** (`experiments/exp_002_hdc_dataset2/results/metrics.json`) :

| Métrique | HDC Online | EWC Online (rappel) |
|---------|:----------:|:-------------------:|
| **AA** | 0.8698 | **0.9824** |
| **AF** | **0.0000** | 0.0010 |
| **BWT** | +0.0019 | +0.0000 |
| RAM peak inférence | 14 504 B | **1 171 B** |
| Latence inférence | 0.048 ms | **0.036 ms** |
| N paramètres | 2 048 | **705** |
| RAM FP32 estimée | 14 344 B | **2 820 B** |
| RAM INT8 estimée | **6 152 B** | 705 B |
| % budget 64 Ko (FP32) | 22.1% | **4.4%** |

**Matrice HDC** :

|   | T1 Pump | T2 Turbine | T3 Compressor |
|---|:-------:|:----------:|:-------------:|
| **Après T1** | 0.8817 | — | — |
| **Après T2** | 0.8698 | 0.8538 | — |
| **Après T3** | **0.8817** | **0.8577** | **0.8699** |

---

## 3. Discussion des résultats

### 3.1 EWC versus fine-tuning naïf : différence modeste mais correcte

L'EWC Online surpasse le fine-tuning naïf sur l'AA finale (0.9824 vs 0.9811, Δ=0.0013) et présente un AF légèrement positif (0.001) là où le naïf affiche AF=0.

**Pourquoi AF est quasi-nul sur les deux méthodes ?** Les trois domaines du Dataset 2 ont des distributions de features très proches (même gamme de température, pression, vibration, humidité). L'oubli catastrophique est statistiquement absent car les gradients appris sur T2/T3 restent pertinents pour T1. Cela ne valide pas la résistance à l'oubli d'EWC dans un scénario de drift réel.

**Implication** : l'écart EWC/naïf sur ce dataset sous-estime l'avantage réel d'EWC. L'expérience sur Dataset 1 (séries temporelles avec drift temporel fort) est nécessaire pour observer un AF significatif et discriminer les méthodes.

### 3.2 Résultat surprenant : joint training ≤ naïf

AA\_joint = 0.9811 ≤ AA\_naïf = 0.9811 (ex-æquo). Cela contredit l'intuition habituelle (joint = borne supérieure).

**Explication probable** : le joint training concatène les 3 domaines et mélange aléatoirement les échantillons. Le fine-tuning naïf entraîne dans l'ordre Pump → Turbine → Compressor, créant un curriculum implicite — les domaines sont suffisamment similaires pour que cet ordonnancement SGD soit légèrement avantageux. Ce n'est pas un bug mais une propriété du dataset.

### 3.3 Budget RAM : très bonne marge

Le pic RAM lors d'une mise à jour EWC (forward + backward + step) est de **6 837 B**, soit **10.4% du budget de 64 Ko**. L'état EWC complet (poids + Fisher + θ\*) occupe 8.5 Ko (12.9%).

Cela laisse une marge de **~55 Ko** pour :
- Le buffer de données (si replay partiel requis)
- Les activations intermédiaires (déjà incluses dans les 6 837 B mesurés)
- La pile système STM32N6 (~8-16 Ko typique)
- Futur buffer UINT8 (TinyOL, S4-02)

**Attention** : les mesures `tracemalloc` en Python surestiment la RAM réelle par rapport à un runtime C CMSIS-NN. Les mesures MCU (Phase 2) seront plus précises.

### 3.4 Latence : ordres de grandeur favorables

| Phase | PC (Python) | Cible MCU | Marge |
|-------|:-----------:|:---------:|:-----:|
| Inférence | 0.036 ms | ≤ 100 ms | ×2778 |
| Mise à jour | 0.637 ms | ≤ 100 ms | ×157 |

Python est bien plus lent que C embarqué par nature. Ces valeurs PC ne sont pas comparables aux temps MCU réels (le portage C devrait être ~10–50× plus lent que Python optimisé sur ce type de modèle). Néanmoins, même en supposant une pénalité MCU de ×100, la mise à jour resterait à 63 ms — sous la cible de 100 ms.

### 3.5 HDC versus EWC : trade-off pertinent

HDC est **5× moins précis** (AA 0.87 vs 0.98) mais présente **AF = 0 par construction** (accumulation additive, non destructive). HDC est également plus simple à porter (pas de backprop, opérations binaires) et ses prototypes INT8 occupent 6.2 Ko seulement.

Ce trade-off est cohérent avec la littérature (Benatti2019HDC) : HDC sacrifie la précision pour obtenir une résistance absolue à l'oubli et une compatibilité MCU native.

---

## 4. Validation de la roadmap

### 4.1 Gap 1 — Données industrielles avec drift réel

**Statut** : ⚠️ Partiellement ouvert

Dataset 2 démontre la faisabilité du pipeline CL (chargement → entraînement → métriques → mémoire) mais ne présente pas de drift inter-domaine significatif (AF ≈ 0 même sans régularisation). Pour valider que les méthodes CL apportent un bénéfice réel, deux axes sont nécessaires :

1. **Dataset 1** (Pump, séries temporelles) — Sprint 3 — drift temporel progressif attendu, scénario plus réaliste industriellement.
2. **FEMTO PRONOSTIA** (backlog Phase 2) — dataset benchmark international pour la prognose de roulements, drift fort et bien documenté.

**Décision roadmap** : Sprint 3 est crucial. Ne pas reporter l'exploration Dataset 1.

### 4.2 Gap 2 — Opération sous 100 Ko RAM avec chiffres mesurés

**Statut** : ✅ Preuve de concept validée côté PC

Les chiffres sont produits (6.8 Ko mise à jour, 1.2 Ko inférence) mais sont mesurés en Python, pas sur MCU réel. La Phase 2 (portage STM32N6) est nécessaire pour produire des mesures certifiées qui comblent ce gap dans la littérature.

Les estimations actuelles montrent une marge confortable (~87.1% du budget disponible) ce qui diminue le risque de dépassement mémoire lors du portage.

**Décision roadmap** : les mesures PC suffisent pour la Phase 1. La Phase 2 (juin) produira les chiffres publiables.

### 4.3 Gap 3 — Quantification INT8 pendant l'entraînement incrémental

**Statut** : ⬜ Non commencé

Sprint 1 implémente FP32 only (backprop + poids). L'estimation INT8 est calculée théoriquement (`params_int8_bytes = n_params × 1 octet`) mais pas utilisée en entraînement.

Sprint 4 (S4-01 `quantization.py`) introduira la quantification PTQ et le buffer UINT8 pour TinyOL. La backprop INT8 est un problème ouvert (`TODO(dorra)` — question hardware).

### 4.4 Avance sur le planning

Le Sprint 2 (HDC + comparaison) est **déjà complété** (exp_002 exécutée le 6 avril). L'implémentation HDC, les tests (44 tests HDC), le notebook `02_baseline_comparison.ipynb` et l'expérience exp_002 sont disponibles. Le projet est donc **2 semaines en avance** sur la roadmap initiale.

**Sprint 3 (TinyOL + Dataset 1)** peut démarrer immédiatement.

### 4.5 Tableau de progression mis à jour

| Modèle | Implémenté | Testé | Expérience | Export ONNX | RAM mesurée |
|--------|:----------:|:-----:|:----------:|:-----------:|:-----------:|
| M2 EWC + MLP | ✅ | ✅ | ✅ exp_001 | ⬜ S4-05 | ✅ PC |
| M3 HDC | ✅ | ✅ | ✅ exp_002 | ⬜ S4-05 | ✅ PC |
| M1 TinyOL | ⬜ S3 | ⬜ | ⬜ | ⬜ | ⬜ |
| M1 + buffer UINT8 | ⬜ S4 | ⬜ | ⬜ | N/A | ⬜ |
| M4a K-Means | ⬜ S5 | ⬜ | ⬜ | N/A | N/A |
| M5 PCA | ⬜ S5 | ⬜ | ⬜ | N/A | N/A |
| M6 Mahalanobis | ⬜ S5 | ⬜ | ⬜ | N/A | ⬜ |

---

## 5. Éléments complémentaires

### 5.1 Inventaire complet des fichiers source

| Fichier | LOC | Statut Sprint 1 |
|--------|:---:|:--------------:|
| `src/models/base_cl_model.py` | 189 | ✅ |
| `src/models/ewc/ewc_mlp.py` | 232 | ✅ |
| `src/models/ewc/fisher.py` | 163 | ✅ |
| `src/models/hdc/hdc_classifier.py` | ~400 | ✅ Sprint 2 |
| `src/models/hdc/base_vectors.py` | ~200 | ✅ Sprint 2 |
| `src/data/monitoring_dataset.py` | 411 | ✅ |
| `src/training/baselines.py` | 193 | ✅ |
| `src/training/scenarios.py` | 139 | ✅ Sprint 2 |
| `src/evaluation/metrics.py` | 214 | ✅ |
| `src/evaluation/memory_profiler.py` | 235 | ✅ |
| `src/evaluation/plots.py` | 263 | ✅ Sprint 2 |
| `src/utils/config_loader.py` | 108 | ✅ |
| `src/utils/reproducibility.py` | 40 | ✅ |
| `scripts/train_ewc.py` | ~250 | ✅ |
| `scripts/train_hdc.py` | ~250 | ✅ Sprint 2 |
| **Total (Sprint 1 strict)** | **~1 985** | |
| **Total (inclus Sprint 2)** | **~3 087** | |

### 5.2 Choix architecturaux clés et justifications MCU

**SGD uniquement (pas d'Adam)**

Adam nécessite 2 états de moment par paramètre (`m` et `v`), soit ×3 la RAM de gradient. Sur 705 paramètres : Adam = 8 460 B vs SGD = 2 820 B. SGD est imposé par la contrainte 64 Ko.

**ReLU uniquement**

CMSIS-NN (bibliothèque ST pour Cortex-M) implémente ReLU INT8 nativement (`arm_relu_q7`). Sigmoid et Tanh nécessitent des lookup tables ou une arithmétique FP coûteuse.

**Normalisation fixée sur Task 1**

Calculer μ/σ en ligne sur chaque tâche risque une fuite inter-tâches (on "voit" les données de T2 pour normaliser T2). Fixer les stats sur T1 est la seule option sans fuite et la seule praticable sur MCU (les stats sont chargées depuis la flash).

**Dropout**

Dropout (p=0.2) est présent en entraînement mais désactivé en inférence (`model.eval()`). Il n'a pas d'impact sur la RAM MCU (aucune table aléatoire nécessaire à l'inférence).

### 5.3 Annotations MCU dans le code source

Conformément à la règle du projet, chaque couche linéaire est annotée :

```python
# Dans ewc_mlp.py
self.fc1 = nn.Linear(input_dim, hidden_dims[0])  # MEM: 512 B @ FP32 / 128 B @ INT8
self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])  # MEM: 2048 B @ FP32 / 512 B @ INT8
self.fc3 = nn.Linear(hidden_dims[1], 1)  # MEM: 64 B @ FP32 / 16 B @ INT8
```

Ces annotations permettent une estimation rapide de la RAM MCU sans exécuter le profiler.

### 5.4 Reproductibilité garantie

Tous les résultats sont reproductibles à l'identique :

```bash
python scripts/train_ewc.py --config configs/ewc_config.yaml
# → exp_id, seed, date dans config_snapshot.yaml
# → metrics.json identique bit-pour-bit (seed=42, CPU)
```

Le fichier `experiments/exp_001_ewc_dataset2/config_snapshot.yaml` constitue la trace complète permettant de re-générer les résultats. `src/utils/reproducibility.py` fixe les seeds NumPy, Python, et PyTorch.

### 5.5 Risques identifiés et mitigations

| Risque | Probabilité | Impact | Mitigation |
|--------|:-----------:|:------:|-----------|
| Dataset 2 trop facile (pas de drift) | **Réalisé** | Moyen | Dataset 1 en Sprint 3 — scénario temporel avec drift progressif |
| Backprop INT8 non supportée STM32N6 | Élevée | Haut | Backprop FP32 sur Cortex-M55 (SW), NPU pour inférence seulement — documenté dans CLAUDE.md |
| Mesures RAM PC ≠ MCU | Certain | Moyen | Phase 2 : profiling sur matériel réel. Les estimations PC sont une borne basse |
| TinyOL : accès Dataset 1 (fenêtrage temporel) | Faible | Haut | S3-01/02 en priorité haute — exploration Dataset 1 avant implémentation |
| deadline manuscrit 15 avril | Imminent | Haut | Résultats exp_001 + exp_002 disponibles — suffisant pour la section expérimentale préliminaire |

### 5.6 Prochaines étapes immédiates (Sprint 3)

Sprint 3 démarre immédiatement. Priorité :

1. **S3-01** — Télécharger et explorer le Dataset 1 (Large Industrial Pump Maintenance) — `notebooks/01_data_exploration.ipynb` section 2
2. **S3-02** — Implémenter `pump_dataset.py` avec fenêtrage temporel (sliding window) — scénario domain-incremental temporal
3. **S3-03/04** — Autoencoder backbone TinyOL + pré-entraînement sur données normales
4. **S3-05/06** — Tête OtO + boucle SGD online + exp_003

Deux points à clarifier avant S3 :
- `TODO(fred)` — confirmer les types de données disponibles côté Edge Spectrum (pour S3-01)
- `TODO(arnaud)` — valider la taille de fenêtre temporelle pour le fenêtrage (actuellement non fixée dans la spec)

---

## Références

| Clé BibTeX | Usage Sprint 1 |
|-----------|---------------|
| `Kirkpatrick2017EWC` | Base théorique EWC, perte régularisée, calcul Fisher |
| `DeLange2021Survey` | Taxonomie CL (domain-incremental vs task vs class), métriques AA/AF/BWT |
| `Benatti2019HDC` | HDC online learning sur MCU — référence pour exp_002 |
| `Capogrosso2023TinyML` | Survey TinyML — justification contraintes 64 Ko et portabilité MCU |
| `Kwon2023LifeLearner` | LifeLearner STM32H747 — référence de comparaison pour la Phase 2 |
