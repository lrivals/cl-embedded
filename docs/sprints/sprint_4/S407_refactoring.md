# S4-07 — Refactoring final + documentation docstrings

| Champ | Valeur |
|-------|--------|
| **ID** | S4-07 |
| **Sprint** | Sprint 4 — Semaine 4 (6–13 mai 2026) |
| **Priorité** | 🟡 Important |
| **Durée estimée** | 4h |
| **Dépendances** | — (peut être fait en parallèle des autres tâches) |
| **Fichiers cibles** | Tout `src/` |
| **Statut** | ⬜ Non démarré |

---

## Objectif

Nettoyer et documenter l'ensemble du code source `src/` avant le portage Phase 2, pour garantir :

1. **Lisibilité** : docstrings NumPy complètes sur toutes les fonctions publiques
2. **Conformité style** : `black` + `ruff` sans avertissement
3. **Pas de valeur hardcodée** : tous les hyperparamètres passent par les configs YAML
4. **Annotations `# MEM:` à jour** : cohérentes avec les architectures finales Sprint 4

**Critère de succès** : `ruff check src/ && black --check src/ && pytest tests/ -v` sans erreur ni warning.

---

## Périmètre des modifications

### Règles de style obligatoires

```bash
# Vérifier avant de committer
black src/                  # formatage automatique
ruff check src/ --fix       # lint + autofix
mypy src/ --ignore-missing-imports  # type checking (best effort)
```

### Checklist par module

#### `src/models/ewc/`
- [ ] `ewc_mlp.py` : docstring NumPy sur `forward()`, `ewc_loss()`, `get_theta_star()` — vérifier format
- [ ] `fisher.py` : docstring sur `compute_fisher_diagonal()`, `update_online_fisher()`
- [ ] Vérifier que `lambda`, `gamma` ne sont jamais hardcodés dans le code source

#### `src/models/hdc/`
- [ ] `hdc_classifier.py` : docstring sur `encode()`, `train_task()`, `predict()`
- [ ] `base_vectors.py` : docstring sur `generate()`, `save()`, `load()`
- [ ] Annotations `# MEM:` sur les matrices de prototypes (dimension × 4 B @ FP32)

#### `src/models/tinyol/`
- [ ] `autoencoder.py` : docstring sur `encode()`, `decode()`, `reconstruction_loss()`
- [ ] `oto_head.py` : docstring sur `OtOHead.forward()`, `TinyOLOnlineTrainer.update()`
- [ ] Annotations `# MEM:` sur le buffer d'embeddings (avant/après S4-02)

#### `src/data/`
- [ ] `monitoring_dataset.py` : docstring sur `get_cl_dataloaders()` — paramètres et retours
- [ ] `pump_dataset.py` : docstring sur `get_pump_dataloaders()`, fenêtrage temporel

#### `src/evaluation/`
- [ ] `metrics.py` : docstring sur `compute_aa()`, `compute_af()`, `compute_bwt()`
- [ ] `memory_profiler.py` : docstring sur `full_memory_report()`

#### `src/utils/`
- [ ] `quantization.py` (S4-01) : déjà documenté
- [ ] `config_loader.py` : docstring sur `load_config()`
- [ ] `reproducibility.py` : docstring sur `set_seed()`

---

## Format docstring NumPy attendu

```python
def compute_af(acc_matrix: np.ndarray) -> float:
    """
    Average Forgetting (AF) — chute moyenne d'accuracy entre pic et fin.

    Parameters
    ----------
    acc_matrix : np.ndarray, shape [T, T]
        Matrice R[t, j] = accuracy sur la tâche j après entraînement sur t.
        Les entrées R[t, j] avec t < j sont NaN (tâche pas encore vue).

    Returns
    -------
    float
        AF moyen sur T-1 tâches. Positif = oubli, négatif = plasticité rétroactive.

    References
    ----------
    DeLange2021Survey, eq. 4
    """
```

---

## Règles sur les hyperparamètres

Toute valeur de configuration doit venir exclusivement du YAML. Exemples de patterns **interdits** :

```python
# ❌ Interdit — valeur hardcodée
ewc_lambda = 1000
lr = 0.01
hidden_dims = [32, 16]

# ✅ Correct — lu depuis la config
ewc_lambda = config["ewc"]["lambda"]
lr = config["training"]["lr"]
hidden_dims = config["model"]["hidden_dims"]
```

Si une constante est nécessaire dans le fichier source (ex. `INPUT_DIM`), elle doit être au niveau module et accompagnée d'un commentaire indiquant son origine YAML :

```python
# Valeur par défaut — doit correspondre à ewc_config.yaml → model.input_dim
INPUT_DIM: int = 4
```

---

## Vérification finale

```bash
# Lint + format
ruff check src/ --statistics
black --check src/ --diff

# Tests sans régression
pytest tests/ -v --tb=short

# Type checking (best effort)
mypy src/models/ src/evaluation/ src/utils/ --ignore-missing-imports
```

---

## Critères d'acceptation

- [ ] `ruff check src/` : 0 erreur (warnings documentés si non critiques)
- [ ] `black --check src/` : 0 modification requise
- [ ] Toutes les fonctions publiques ont une docstring NumPy (paramètres + returns)
- [ ] 0 hyperparamètre hardcodé dans `src/` (vérifiable via `grep -rn "lr = 0\." src/`)
- [ ] `pytest tests/ -v` : 0 régression
- [ ] Annotations `# MEM:` présentes sur chaque couche Linear + buffer intermédiaire

---

## Questions ouvertes

- `TODO(arnaud)` : Faut-il générer une documentation HTML via `pdoc` ou `mkdocs` pour la présentation finale Phase 1 ?
- `FIXME(gap2)` : S'assurer que les annotations `# MEM:` dans `ewc_mlp.py` sont cohérentes avec les mesures réelles de `memory_report.json` (exp_001).
