# M2 — Spécification d'implémentation : EWC Online + MLP

> **Statut** : Priorité 1 (premier à implémenter)  
> **Dataset** : Dataset 2 — Industrial Equipment Monitoring  
> **Référence** : Kirkpatrick et al. (2017), *Overcoming catastrophic forgetting in neural networks*  
> **Taxonomie CL** : Regularization-based

---

## 1. Principe de la méthode

EWC (Elastic Weight Consolidation) ajoute un terme de régularisation à la fonction de perte qui pénalise les changements de poids *importants* pour les tâches précédentes. L'importance de chaque poids est estimée par sa **matrice d'information de Fisher diagonale**.

**Intuition** : les poids importants pour la tâche A sont « élastiquement ancrés » à leurs valeurs après l'apprentissage de A. L'apprentissage de B peut modifier librement les poids non importants pour A, sans « détruire » les connaissances acquises.

### Fonction de perte EWC

```
L_EWC(θ) = L_B(θ) + λ/2 · Σᵢ Fᵢ (θᵢ - θ*ᵢ)²
```

Où :
- `L_B(θ)` : perte sur la tâche courante B (cross-entropy)
- `λ` : coefficient de régularisation (hyperparamètre)
- `Fᵢ` : diagonale de la matrice de Fisher pour le poids i (calculée sur la tâche A)
- `θ*ᵢ` : valeur du poids i après apprentissage de la tâche A (snapshot)

### EWC Online (variante implémentée)

La version Online (Schwarz et al., 2018) accumule la Fisher diagonale avec un facteur de décroissance γ au lieu de stocker une Fisher par tâche :

```
F_online ← γ · F_old + F_new_task
θ*_online ← mis à jour après chaque tâche
```

**Avantage MCU** : une seule copie de la Fisher diagonale en RAM, quelle que soit le nombre de tâches vues → overhead mémoire fixe (×3 vs modèle seul).

---

## 2. Architecture du MLP

```
Input: [batch, 6]    ← 6 features tabulaires
   │
   ├── Linear(6 → 32) + ReLU     # MEM: 224 B @ FP32 / 56 B @ INT8
   ├── Dropout(p=0.2)             # désactivé à l'inférence MCU
   ├── Linear(32 → 16) + ReLU    # MEM: 520 B @ FP32 / 130 B @ INT8
   └── Linear(16 → 1)  + Sigmoid  # MEM: 68 B @ FP32 / 17 B @ INT8

Sortie : probabilité de défaut ŷ ∈ [0, 1]
Perte : Binary Cross-Entropy (BCE)
```

### Compte des paramètres

| Couche | Params | FP32 |
|--------|--------|------|
| Linear(6→32) | 6×32 + 32 = 224 | 896 B |
| Linear(32→16) | 32×16 + 16 = 528 | 2 112 B |
| Linear(16→1) | 16×1 + 1 = 17 | 68 B |
| **Total modèle** | **769 params** | **~3 Ko** |
| Fisher diagonale | 769 scalaires | ~3 Ko |
| Snapshot θ* | 769 scalaires | ~3 Ko |
| **TOTAL EWC** | **2 307 scalaires** | **~9 Ko @ FP32** |

> ✅ Très largement dans la cible 64 Ko. Ce modèle est le plus frugal des trois.

---

## 3. Features d'entrée (Dataset 2)

Le Dataset 2 est déjà tabulaire — aucun feature engineering temporel requis.

| Feature | Type | Preprocessing |
|---------|------|---------------|
| `temperature` | Float | Z-score normalisé |
| `pressure` | Float | Z-score normalisé |
| `vibration` | Float | Z-score normalisé |
| `humidity` | Float | Z-score normalisé |
| `equipment_type` | Catégoriel | One-hot (3 catégories → 2 features, drop first) ou label encoding |
| `location` | Catégoriel | Label encoding (si faible cardinalité) |

> **Note implémentation** : le type d'équipement (`pump`, `turbine`, `compressor`) est la variable qui définit les **domaines CL**. Elle doit être encodée mais aussi utilisée pour splitter les tâches.

### Normalisation

```python
# Calculée sur le premier domaine (Task 1 = pumps) uniquement
# Appliquée à tous les domaines avec les mêmes statistiques
# → stockée dans configs/ewc_config.yaml pour reproductibilité
FEATURE_MEAN = [mu_temp, mu_pres, mu_vib, mu_hum, ...]
FEATURE_STD  = [std_temp, std_pres, std_vib, std_hum, ...]
```

---

## 4. Scénario de Continual Learning

### 4.1 Type de scénario
**Domain-Incremental** : le modèle apprend séquentiellement sur 3 types d'équipements. Le type d'équipement est connu à l'entraînement (task label disponible), mais pas à l'inférence (le modèle doit généraliser sans connaître le domaine).

### 4.2 Construction des tâches

```python
TASKS = {
    "task_1": {"equipment": "pump",       "n_samples": None},  # utiliser tout
    "task_2": {"equipment": "turbine",    "n_samples": None},
    "task_3": {"equipment": "compressor", "n_samples": None},
}
# Ordre : pump → turbine → compressor (ordre fixe pour reproductibilité)
# Chaque tâche est vue une seule fois (1 epoch par tâche)
```

### 4.3 Baselines à comparer

| Baseline | Description | Objectif |
|----------|-------------|---------|
| Fine-tuning naïf | SGD sans régularisation | Mesure l'oubli catastrophique brut |
| **EWC Online** | Notre méthode | Méthode principale |
| Joint training | Entraînement batch sur toutes les tâches réunies | Borne supérieure (irréaliste MCU) |

> Le Fine-tuning naïf et le Joint training doivent être implémentés comme baselines obligatoires dans `src/training/baselines.py`.

### 4.4 Protocole d'évaluation

```
Initialisation : Fisher = 0, θ* = None

Pour chaque tâche tᵢ :
  1. Entraîner sur données tᵢ avec L_EWC (ou L_BCE si t₁)
  2. Calculer Fisher diagonale sur tᵢ (accumulation Online)
  3. Mettre à jour θ*
  4. Évaluer sur T1, T2, ..., Ti
  5. Enregistrer acc_matrix[i, j] pour j ≤ i

Post-entraînement :
  → AA = moyenne des acc finales
  → AF = Σ (acc_max_j - acc_final_j) / (T-1)
  → BWT = Σ (acc_final_j - acc_after_j_training) / (T-1)
```

---

## 5. Calcul de la Fisher diagonale

```python
def compute_fisher_diagonal(model, dataloader, device):
    """
    Calcule la diagonale de la matrice de Fisher par méthode empirique.
    F_i ≈ E[(∂ log p(y|x,θ) / ∂θᵢ)²]
    
    Returns
    -------
    dict[str, Tensor] : F diagonale par nom de paramètre
    """
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    model.eval()
    
    for x, y in dataloader:
        model.zero_grad()
        output = model(x.to(device))
        loss = F.binary_cross_entropy(output, y.to(device))
        loss.backward()
        
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.detach() ** 2
    
    # Normaliser par le nombre de batches
    n_batches = len(dataloader)
    fisher = {n: f / n_batches for n, f in fisher.items()}
    return fisher
```

> ⚠️ **Contrainte MCU** : cette fonction s'exécute une seule fois en fin de tâche, sur PC ou hors-ligne. Sur MCU, la Fisher est pré-calculée et chargée depuis Flash.

---

## 6. Hyperparamètres

```yaml
# configs/ewc_config.yaml

model:
  architecture: "mlp"
  input_dim: 6
  hidden_dims: [32, 16]
  output_dim: 1
  activation: "relu"
  dropout: 0.2            # désactivé à l'inférence

training:
  optimizer: "sgd"        # Adam interdit (overhead MCU)
  learning_rate: 0.01
  momentum: 0.9           # acceptable (1 vecteur momentum en RAM)
  epochs_per_task: 10     # nombre de passes sur chaque tâche
  batch_size: 32

ewc:
  lambda: 1000            # coefficient de régularisation (à tuner)
  gamma: 0.9              # facteur de décroissance Fisher Online
  n_fisher_samples: 200   # échantillons pour estimer la Fisher

evaluation:
  seed: 42
  metrics: ["acc_final", "avg_forgetting", "bwt", "ram_peak_bytes"]

memory:
  target_ram_bytes: 65536   # 64 Ko cible STM32N6
  expected_ram_bytes: 9216  # ~9 Ko estimé
```

---

## 7. Sensibilité au coefficient λ

Le coefficient λ est le principal hyperparamètre à tuner. Sa valeur contrôle le compromis plasticité/stabilité :

| λ faible (< 100) | λ intermédiaire (~1000) | λ élevé (> 10 000) |
|---|---|---|
| Forte plasticité, oubli important | Équilibre stabilité/plasticité | Forte stabilité, peu d'adaptation |
| ≈ Fine-tuning naïf | **Valeur recommandée** | ≈ Modèle gelé |

**Protocole de tuning** : grid search sur λ ∈ {10, 100, 500, 1000, 5000} sur les deux premières tâches, puis validation sur la troisième.

---

## 8. Portabilité MCU — Checklist

| Critère | Statut | Note |
|---------|--------|------|
| SGD (pas Adam) | ✅ | Configurable via YAML |
| ReLU uniquement | ✅ | INT8-friendly (CMSIS-NN) |
| Pas de BatchNorm | ✅ | Stats fixes pour normalisation |
| Taille < 9 Ko (FP32) | ✅ estimé | À vérifier `memory_profiler.py` |
| Fisher stockable en Flash | ✅ | Dictionnaire de 769 scalaires |
| Backprop sur Cortex-M55 | ✅ | Opérations matricielles standards |
| Inférence exportable INT8 | À valider | Via ONNX → TFLite Micro |

---

## 9. Fichiers à produire

```
src/models/ewc/
├── __init__.py
├── ewc_mlp.py             ← modèle MLP + perte EWC
└── fisher.py              ← calcul Fisher diagonale

src/training/
├── cl_trainer.py          ← boucle d'entraînement CL générique
└── baselines.py           ← fine-tuning naïf + joint training

src/data/
└── monitoring_dataset.py  ← loader Dataset 2 + split par domaine

configs/
└── ewc_config.yaml

experiments/exp_001_ewc_dataset2/
├── config_snapshot.yaml
└── results/
    ├── metrics.json
    ├── accuracy_matrix.png
    └── forgetting_curve.png
```

---

## 10. Références

- Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. *PNAS*, 114(13), 3521–3526.
- Schwarz, J., et al. (2018). Progress & compress: A scalable framework for continual learning. *ICML*.
- De Lange, M., et al. (2021). A continual learning survey: Defying forgetting in classification tasks. *TPAMI*.
- Hurtado, J., et al. (2023). Continual learning for predictive maintenance: Overview and challenges. *IFAC-PapersOnLine*.
