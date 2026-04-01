# CLAUDE.md — Contexte du projet pour Claude Code

> Ce fichier est la source de vérité pour Claude Code dans ce dépôt.
> Lire entièrement avant toute intervention dans le code.

---

## Identité du projet

**Titre** : Apprentissage Incrémental pour Systèmes Embarqués à Ressources Limitées  
**Acronyme** : CL-Embedded  
**Type** : Recherche M2 + prototype industriel  
**Institution** : ISAE-SUPAERO (DISC), en collaboration avec ENAC (LII) et Edge Spectrum  
**Auteur** : Léonard Rivals  
**Période** : 16 mars – 6 août 2026  
**Deadline manuscrit préliminaire** : 15 avril 2026  

---

## Objectif principal (Objectif 1 du stage)

Implémenter et comparer trois méthodes d'apprentissage incrémental (continual learning, CL) sur PC en Python, conçues pour être portées sur un microcontrôleur **STM32N6** (Cortex-M55, ~64 Ko RAM, NPU inférence-only).

Application visée : **maintenance prédictive industrielle** (détection de panne, classification d'état).

---

## Les trois modèles du projet

| ID | Modèle | Dataset | Méthode CL | Fichier spec |
|----|--------|---------|------------|--------------|
| M1 | TinyOL + tête OtO | Dataset 1 — Pump (temporel) | Architecture-based | `docs/models/tinyol_spec.md` |
| M2 | EWC Online + MLP | Dataset 2 — Monitoring (tabulaire) | Regularization-based | `docs/models/ewc_mlp_spec.md` |
| M3 | HDC Hyperdimensional | Dataset 2 — Monitoring (tabulaire) | Architecture-based (non-neuronal) | `docs/models/hdc_spec.md` |

**Priorité d'implémentation** : M2 → M3 → M1 → extension buffer UINT8 sur M1.

---

## Contraintes non négociables (hardware STM32N6)

1. **RAM ≤ 64 Ko** — modèle + activations + buffer CL réunis
2. **NPU = inférence uniquement** — la backpropagation s'exécute sur Cortex-M55 (SW)
3. **FP32 pour la backprop** — pas de backprop INT8 native (à explorer sur MLP minimal)
4. **Latence ≤ 100 ms** par inférence + mise à jour
5. **Pas d'accès à un dataset complet en RAM** — le modèle voit chaque échantillon une seule fois (online learning) ou via un buffer borné

> **Règle de code** : tout paramètre de taille (couches, buffer, embeddings) doit avoir une constante nommée dans `configs/` avec une valeur par défaut respectant la contrainte 64 Ko.

---

## Triple gap — Positionnement scientifique

Le projet cherche à être le **premier travail** à satisfaire simultanément :

| Gap | Description | Statut littérature |
|-----|-------------|-------------------|
| **Gap 1** | Validation sur données industrielles de séries temporelles réelles | ❌ Non comblé |
| **Gap 2** | Opération sous 100 Ko RAM avec chiffres précis mesurés | ❌ Non comblé |
| **Gap 3** | Quantification INT8 pendant l'entraînement incrémental | ❌ Non comblé |

**Chaque décision d'architecture doit être justifiée par rapport à ces trois gaps.**

---

## Datasets

### Dataset 1 — Large Industrial Pump Maintenance Dataset (Kaggle)
- **Type** : Séries temporelles multivariées (température, vibration, pression, RPM)
- **Label** : `maintenance_required` (binaire)
- **Scénario CL** : Domain-incremental avec drift temporel
- **Chemin** : `data/raw/pump_maintenance/`
- **Spec** : `docs/context/datasets.md`

### Dataset 2 — Industrial Equipment Monitoring Dataset (Kaggle)
- **Type** : Tabulaire statique (température, pression, vibration, humidité, type équipement)
- **Label** : `faulty` (0/1 binaire)
- **Scénario CL** : Domain-incremental par type d'équipement (pump → turbine → compressor)
- **Chemin** : `data/raw/equipment_monitoring/`

---

## Structure du dépôt

```
cl-embedded/
├── CLAUDE.md                   ← CE FICHIER
├── README.md
├── .gitignore
├── requirements.txt
├── pyproject.toml
├── configs/
│   ├── tinyol_config.yaml
│   ├── ewc_config.yaml
│   └── hdc_config.yaml
├── data/
│   ├── raw/                    ← données brutes (gitignore)
│   └── processed/              ← features extraites (gitignore)
├── docs/
│   ├── models/                 ← specs détaillées des 3 modèles
│   ├── context/                ← contexte projet, hardware, datasets
│   ├── roadmap.md
│   └── triple_gap.md
├── skills/                     ← prompts Claude spécialisés
├── src/
│   ├── data/                   ← loaders, feature engineering
│   ├── models/                 ← implémentations des 3 modèles
│   ├── training/               ← boucles CL, scénarios
│   ├── evaluation/             ← métriques CL, profiler mémoire
│   └── utils/                  ← quantization helpers, misc
├── experiments/                ← résultats reproductibles (configs + outputs)
├── notebooks/                  ← exploration, visualisation
├── tests/
└── scripts/                    ← points d'entrée CLI
```

---

## Conventions de code

### Style
- **Python ≥ 3.10**
- Type hints obligatoires sur toutes les fonctions publiques
- Docstrings format NumPy
- `black` pour le formatage, `ruff` pour le linting
- Pas de dépendances lourdes non justifiées (pas de HuggingFace transformers, etc.)

### Nommage
- Classes : `CamelCase` — ex. `EWCMlpClassifier`, `TinyOLAutoencoder`
- Fonctions/variables : `snake_case`
- Constantes de config : `UPPER_SNAKE_CASE` dans les configs YAML
- Fichiers de résultats : `{exp_id}_{model}_{dataset}_{date}.json`

### Reproductibilité
- Seed fixé via `utils/reproducibility.py` : `set_seed(42)` par défaut
- Chaque expérience génère un `config_snapshot.yaml` dans `experiments/exp_XXX/`
- Pas de résultats hardcodés — tout sort d'une exécution de script

### Contrainte embarquée dans le code
```python
# Toujours annoter les tenseurs avec leur empreinte mémoire estimée
# Format : # MEM: {taille en octets} @ FP32 / {taille en octets} @ INT8
hidden = torch.relu(self.fc1(x))  # MEM: 256 B @ FP32 / 64 B @ INT8
```

---

## Métriques d'évaluation obligatoires

Pour chaque expérience CL, reporter systématiquement :

| Métrique | Formule | Module |
|---------|---------|--------|
| `acc_final` | Accuracy sur toutes les tâches vues après entraînement complet | `evaluation/metrics.py` |
| `avg_forgetting` (AF) | Chute moyenne d'accuracy entre pic et fin par tâche | `evaluation/metrics.py` |
| `backward_transfer` (BWT) | Impact de l'apprentissage futur sur les tâches passées | `evaluation/metrics.py` |
| `ram_peak_bytes` | RAM maximale mesurée à l'exécution (tracemalloc) | `evaluation/memory_profiler.py` |
| `inference_latency_ms` | Latence forward pass (moyenne sur 100 runs) | `evaluation/memory_profiler.py` |
| `n_params` | Nombre total de paramètres entraînables | automatique via `model.parameters()` |

---

## Références bibliographiques clés

Utiliser ces clés BibTeX exactes (issues de `references.bib` du projet manuscrit) :

- `Ren2021TinyOL` — TinyOL (architecture-based, MCU-validated)
- `Kirkpatrick2017EWC` — EWC original
- `Benatti2019HDC` — HDC online learning sur MCU
- `Ravaglia2021QLRCL` — QLR-CL, rejeu latent UINT8
- `Kwon2023LifeLearner` — LifeLearner Tiny (STM32H747)
- `Capogrosso2023TinyML` — Survey TinyML de référence
- `DeLange2021Survey` — Taxonomie CL de référence
- `Hurtado2023CLPdM` — CL × maintenance prédictive

---

## Ce que Claude Code NE DOIT PAS faire

- ❌ Inventer des résultats de benchmark ou des chiffres RAM non mesurés
- ❌ Introduire des dépendances qui ne seraient pas portables sur MCU (ex. bibliothèques de visualisation dans le code de modèle)
- ❌ Modifier les hyperparamètres dans les fichiers source — toujours passer par les configs YAML
- ❌ Supprimer les annotations `# MEM:` dans les couches de modèle
- ❌ Créer des notebooks sans les déplacer dans `notebooks/`
- ❌ Committer des données brutes (les répertoires `data/` sont dans `.gitignore`)

---

## Interlocuteurs et rôles (pour les commentaires / TODO)

- `TODO(arnaud)` — question pour Arnaud Dion (superviseur ISAE-SUPAERO)
- `TODO(dorra)` — question technique pour Dorra Ben Khalifa (quantification, hardware)
- `TODO(fred)` — point pour Frédéric Zbierski (Edge Spectrum, contexte industriel)
- `FIXME(gap1/2/3)` — point bloquant lié à l'un des trois gaps

---

## Commandes rapides

```bash
# Setup
pip install -e ".[dev]"

# Lancer un entraînement
python scripts/train_ewc.py --config configs/ewc_config.yaml
python scripts/train_hdc.py --config configs/hdc_config.yaml
python scripts/train_tinyol.py --config configs/tinyol_config.yaml

# Évaluation complète
python scripts/evaluate_all.py --exp_dir experiments/

# Tests
pytest tests/ -v

# Profiling mémoire
python scripts/profile_memory.py --model ewc --dataset monitoring
```
