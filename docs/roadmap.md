# Roadmap — CL-Embedded

> Version : 1.0 | Mise à jour : 1er avril 2026  
> Horizon : Phase 1 (PC Python) = avril–mai 2026

---

## Vue macro — Phases du stage

```
Phase 0 : Revue de littérature    [16 mars → 15 avril 2026]
  └── Manuscrit préliminaire (deadline : 15 avril)

Phase 1 : Implémentation Python   [15 avril → 15 mai 2026]
  └── Ce dépôt — 3 modèles sur PC

Phase 2 : Portage MCU             [15 mai → 15 juin 2026]
  └── STM32N6 — profiling mémoire + latence

Phase 3 : Expériences + rédaction [15 juin → 6 août 2026]
  └── Rapport final + code GitHub public
```

---

## Sprints Phase 1 (détail)

### Sprint 1 — Semaine 1 (15–22 avril)
**Objectif** : Infrastructure du projet + M2 (EWC) fonctionnel sur Dataset 2

#### Tâches

| ID | Tâche | Priorité | Fichier cible | Durée est. |
|----|-------|:--------:|---------------|------------|
| S1-01 | Créer le dépôt GitHub + structure de dossiers | 🔴 | `README.md`, `.gitignore`, `pyproject.toml` | 1h |
| S1-02 | Télécharger Dataset 2 (Monitoring) + exploration | 🔴 | `notebooks/01_data_exploration.ipynb` | 2h |
| S1-03 | Implémenter `monitoring_dataset.py` (loader + split domaine) | 🔴 | `src/data/monitoring_dataset.py` | 3h |
| S1-04 | Implémenter `ewc_mlp.py` (MLP + perte EWC) | 🔴 | `src/models/ewc/ewc_mlp.py` | 4h |
| S1-05 | Implémenter `fisher.py` (calcul Fisher diagonale) | 🔴 | `src/models/ewc/fisher.py` | 2h |
| S1-06 | Implémenter `baselines.py` (fine-tuning naïf + joint) | 🔴 | `src/training/baselines.py` | 2h |
| S1-07 | Implémenter `metrics.py` (AA, AF, BWT) | 🔴 | `src/evaluation/metrics.py` | 2h |
| S1-08 | Implémenter `memory_profiler.py` (tracemalloc) | 🟡 | `src/evaluation/memory_profiler.py` | 2h |
| S1-09 | Première expérience EWC : exp_001 | 🔴 | `experiments/exp_001_ewc_dataset2/` | 2h |
| S1-10 | Tests unitaires modèle EWC + métriques | 🟡 | `tests/test_ewc.py` | 2h |

**Livrable sprint 1** : EWC Online + MLP entraîné sur 3 domaines, métriques AA/AF/BWT et RAM mesurés, résultats dans `experiments/exp_001/`.

---

### Sprint 2 — Semaine 2 (22–29 avril)
**Objectif** : M3 (HDC) fonctionnel + comparaison EWC vs HDC sur Dataset 2

| ID | Tâche | Priorité | Fichier cible | Durée est. |
|----|-------|:--------:|---------------|------------|
| S2-01 | Implémenter `base_vectors.py` (génération + save HDC) | 🔴 | `src/models/hdc/base_vectors.py` | 2h |
| S2-02 | Implémenter `hdc_classifier.py` (encodage + prototypes + inférence) | 🔴 | `src/models/hdc/hdc_classifier.py` | 4h |
| S2-03 | Expérience HDC : exp_002 | 🔴 | `experiments/exp_002_hdc_dataset2/` | 2h |
| S2-04 | Notebook comparaison EWC vs HDC vs Fine-tuning | 🔴 | `notebooks/02_baseline_comparison.ipynb` | 3h |
| S2-05 | `scenarios.py` (gestion générique des streams CL) | 🟡 | `src/training/scenarios.py` | 2h |
| S2-06 | Visualisation accuracy matrix (heatmap forgetting) | 🟡 | `src/evaluation/plots.py` | 2h |
| S2-07 | Config YAML pour HDC + refactoring configs | 🟡 | `configs/hdc_config.yaml` | 1h |
| S2-08 | Tests unitaires HDC | 🟡 | `tests/test_hdc.py` | 2h |
| S2-09 | Mise à jour README avec résultats préliminaires | 🟢 | `README.md` | 1h |

**Livrable sprint 2** : comparaison EWC vs HDC vs Fine-tuning naïf sur Dataset 2, tableau de résultats complet.

---

### Sprint 3 — Semaine 3 (29 avril – 6 mai)
**Objectif** : M1 (TinyOL) — pré-entraînement + boucle online sur Dataset 1

| ID | Tâche | Priorité | Fichier cible | Durée est. |
|----|-------|:--------:|---------------|------------|
| S3-01 | Télécharger Dataset 1 (Pump) + exploration | 🔴 | `notebooks/01_data_exploration.ipynb` (section 2) | 2h |
| S3-02 | Implémenter `pump_dataset.py` (fenêtrage + features) | 🔴 | `src/data/pump_dataset.py` | 4h |
| S3-03 | Implémenter `autoencoder.py` (backbone + décodeur) | 🔴 | `src/models/tinyol/autoencoder.py` | 3h |
| S3-04 | Pré-entraînement backbone (données normales uniquement) | 🔴 | `scripts/pretrain_tinyol.py` | 2h |
| S3-05 | Implémenter `oto_head.py` (tête OtO + boucle SGD online) | 🔴 | `src/models/tinyol/oto_head.py` | 3h |
| S3-06 | Expérience TinyOL : exp_003 | 🔴 | `experiments/exp_003_tinyol_dataset1/` | 3h |
| S3-07 | Tests unitaires TinyOL | 🟡 | `tests/test_tinyol.py` | 2h |
| S3-08 | Notebook CL évaluation Dataset 1 | 🟡 | `notebooks/03_cl_evaluation.ipynb` | 2h |

**Livrable sprint 3** : TinyOL entraîné et évalué sur Dataset 1, comparaison avec fine-tuning naïf.

---

### Sprint 4 — Semaine 4 (6–13 mai)
**Objectif** : Extension buffer UINT8 + comparaison finale 3 modèles

| ID | Tâche | Priorité | Fichier cible | Durée est. |
|----|-------|:--------:|---------------|------------|
| S4-01 | Implémenter `quantization.py` (UINT8 encoder/decoder) | 🔴 | `src/utils/quantization.py` | 3h |
| S4-02 | Extension buffer UINT8 sur TinyOL | 🔴 | `src/models/tinyol/oto_head.py` (extension) | 3h |
| S4-03 | Exp buffer UINT8 vs FP32 : delta précision | 🔴 | `experiments/exp_004_tinyol_uint8/` | 2h |
| S4-04 | Tableau comparatif final 3 modèles (+ baselines) | 🔴 | `notebooks/04_final_comparison.ipynb` | 3h |
| S4-05 | Export ONNX des 3 modèles (vérification portabilité) | 🟡 | `scripts/export_onnx.py` | 3h |
| S4-06 | Profiling mémoire systématique (3 modèles) | 🔴 | `scripts/profile_memory.py` | 2h |
| S4-07 | Refactoring final + documentation docstrings | 🟡 | Tout `src/` | 4h |
| S4-08 | `CONTRIBUTING.md` + `LICENSE` | 🟢 | Racine | 1h |

**Livrable sprint 4** : tableau comparatif complet 3 modèles, chiffres RAM mesurés, export ONNX validé. Prêt pour portage MCU (Phase 2).

---

## Backlog (Phase 2 — portage MCU)

| Tâche | Priorité | Notes |
|-------|:--------:|-------|
| Validation sur FEMTO PRONOSTIA | 🔴 | Gap 1 scientifique |
| Quantification PTQ + export TFLite Micro | 🔴 | Via STM32Cube.AI |
| Profiling RAM/latence sur STM32N6 réel | 🔴 | Gap 2 — mesures précises |
| Exploration backprop INT8 (MLP minimal) | 🟡 | Gap 3 |
| Benchmark sur équipement Edge Spectrum | 🟡 | Contexte industriel Frédéric |

---

## Indicateurs de progression

Mettre à jour ce tableau après chaque sprint :

| Modèle | Implémenté | Testé | Expérience | Export ONNX | RAM mesurée |
|--------|:----------:|:-----:|:----------:|:-----------:|:-----------:|
| M2 EWC + MLP | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| M3 HDC | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| M1 TinyOL | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| M1 + buffer UINT8 | ⬜ | ⬜ | ⬜ | N/A | ⬜ |
