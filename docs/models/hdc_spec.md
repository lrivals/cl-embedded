# M3 — Spécification d'implémentation : HDC (Hyperdimensional Computing)

> **Statut** : Priorité 2  
> **Dataset** : Dataset 2 — Industrial Equipment Monitoring  
> **Référence** : Benatti et al. (2019), *Online learning with HDC for EMG classification on MCU*  
> **Taxonomie CL** : Architecture-based (non-neuronal, vecteurs haute dimension)

---

## 1. Principe de la méthode

L'Hyperdimensional Computing (HDC) est un paradigme de calcul qui représente l'information dans des **hypervecteurs binaires** de très grande dimension (D = 1 000 à 10 000 dimensions). L'apprentissage HDC est fondamentalement différent du gradient descent :

1. **Encoder** chaque observation en un hypervecteur binaire via des projections aléatoires pseudo-orthogonales
2. **Accumuler** les hypervecteurs d'une même classe par addition vectorielle → **vecteur de prototype de classe**
3. **Classer** une nouvelle observation par similarité cosinus (ou Hamming) avec les prototypes

**Mécanisme CL natif** : l'accumulation de nouveaux exemples met à jour directement le prototype de classe → **pas de gradient, pas d'oubli catastrophique par construction**.

**Avantage MCU** : opérations binaires (XOR, POPCOUNT) → très efficace sur Cortex-M55 avec instructions DSP/SIMD. Pas de multiplication flottante dans la phase d'entraînement.

---

## 2. Architecture HDC

### 2.1 Vue d'ensemble

```
Input: x ∈ ℝ⁶  (6 features tabulaires, normalisées)
   │
   ├── Encodage niveau (Level Encoding)
   │   ├── Discrétiser chaque feature en Q niveaux quantifiés
   │   └── Associer chaque niveau à un hypervecteur de niveau H_level ∈ {-1, +1}^D
   │
   ├── Encodage position (Position Encoding)  
   │   ├── Associer chaque position i à un hypervecteur de position H_pos_i ∈ {-1, +1}^D
   │   └── H_feature_i = H_level_i ⊗ H_pos_i  (produit Hadamard / XOR)
   │
   ├── Encodage observation
   │   └── H_obs = ∑ᵢ H_feature_i  (sommation → hypervecteur d'observation)
   │       → binarisation : H_obs_bin = sign(H_obs) ∈ {-1, +1}^D
   │
   └── Classification
       ├── Prototype classe c : C_c = ∑ H_obs pour chaque obs de classe c
       └── ŷ = argmax_c cosine_similarity(H_obs_bin, C_c)
```

### 2.2 Dimension des hypervecteurs

```python
D = 1024    # dimension des hypervecteurs (puissance de 2 pour SIMD)
            # MEM: D × n_classes × 4 octets = 1024 × 2 × 4 = 8 Ko @ INT32
            # MEM: D × n_classes × 1 octet  = 1024 × 2 × 1 = 2 Ko @ INT8 (prototypes binarisés)
```

**Choix de D = 1 024** : compromis entre capacité discriminante et mémoire. Benatti et al. (2019) utilisent D = 4 096 pour EMG (données haute fréquence) ; pour données tabulaires simples, D = 1 024 est suffisant.

### 2.3 Budget mémoire

| Composant | Taille | Stockage |
|-----------|--------|---------|
| Hypervecteurs de base (H_level) : Q × D | 10 × 1 024 × 1 bit = 1,25 Ko | Flash |
| Hypervecteurs de position (H_pos) : F × D | 6 × 1 024 × 1 bit = 768 B | Flash |
| Prototypes de classes (INT32 accumulateurs) | 2 × 1 024 × 4 = 8 Ko | RAM |
| Prototypes binarisés (inférence) | 2 × 1 024 × 1 = 256 B | RAM |
| Buffer encodage (1 observation) | 1 024 × 4 = 4 Ko | RAM temporaire |
| **TOTAL RAM** | **< 12 Ko** | **RAM** |
| **TOTAL Flash (matrices de base)** | **~2 Ko** | **Flash** |

> ✅ Budget mémoire exceptionnel. HDC est la méthode la plus frugale de toutes.

---

## 3. Encodage détaillé

### 3.1 Génération des hypervecteurs de base (offline, une seule fois)

```python
def generate_base_hvectors(D: int, n_levels: int, n_features: int, seed: int = 42):
    """
    Génère les hypervecteurs de base pseudo-aléatoires.
    Ces vecteurs sont FIXES (non appris) et doivent être sauvegardés
    pour être rechargés identiquement sur MCU.
    
    Returns
    -------
    H_level : np.ndarray [n_levels, D], dtype=int8, valeurs ∈ {-1, +1}
    H_pos   : np.ndarray [n_features, D], dtype=int8, valeurs ∈ {-1, +1}
    """
    rng = np.random.default_rng(seed)
    H_level = rng.choice([-1, 1], size=(n_levels, D)).astype(np.int8)
    H_pos   = rng.choice([-1, 1], size=(n_features, D)).astype(np.int8)
    return H_level, H_pos
```

> ⚠️ Le seed DOIT être fixé et sauvegardé — les hypervecteurs de base sont le "modèle" de HDC.

### 3.2 Quantification des features en niveaux

```python
N_LEVELS = 10   # Q = 10 niveaux de quantification par feature

def quantize_feature(value: float, feature_min: float, feature_max: float, n_levels: int) -> int:
    """
    Mappe une feature continue dans [feature_min, feature_max] vers un indice ∈ [0, n_levels-1].
    Linéaire par défaut ; peut être remplacé par quantification non-uniforme.
    """
    normalized = (value - feature_min) / (feature_max - feature_min + 1e-8)
    level_idx = int(normalized * (n_levels - 1))
    return np.clip(level_idx, 0, n_levels - 1)
```

> Les bornes `feature_min` / `feature_max` sont calculées sur Task 1 et stockées dans les configs (comme pour EWC).

### 3.3 Encodage d'une observation

```python
def encode_observation(x: np.ndarray, H_level: np.ndarray, H_pos: np.ndarray,
                        feature_bounds: list, n_levels: int, D: int) -> np.ndarray:
    """
    Encode un vecteur de features en hypervecteur d'observation.
    
    Parameters
    ----------
    x : [n_features] feature vector
    
    Returns
    -------
    H_obs_bin : [D] binarized observation hypervector, dtype=int8
    """
    H_sum = np.zeros(D, dtype=np.int32)  # accumulateur
    
    for i, (feat_val, (f_min, f_max)) in enumerate(zip(x, feature_bounds)):
        level_idx = quantize_feature(feat_val, f_min, f_max, n_levels)
        # Produit Hadamard : H_level[level_idx] ⊗ H_pos[i]
        H_feature = H_level[level_idx] * H_pos[i]  # élément par élément
        H_sum += H_feature.astype(np.int32)
    
    # Binarisation
    H_obs_bin = np.sign(H_sum).astype(np.int8)
    H_obs_bin[H_obs_bin == 0] = 1  # cas dégénéré (parité)
    return H_obs_bin
```

---

## 4. Apprentissage et mise à jour incrémentale

### 4.1 Entraînement initial (Task 1)

```python
def train_hdc(H_level, H_pos, dataloader, n_classes, D, feature_bounds, n_levels):
    """
    Entraînement HDC : accumulation des hypervecteurs par classe.
    Complexité : O(N × F × D) en temps, O(C × D) en mémoire.
    """
    prototypes_acc = np.zeros((n_classes, D), dtype=np.int32)  # accumulateurs
    class_counts   = np.zeros(n_classes, dtype=np.int32)
    
    for x_batch, y_batch in dataloader:
        for x, y in zip(x_batch, y_batch):
            H_obs = encode_observation(x.numpy(), H_level, H_pos, 
                                        feature_bounds, n_levels, D)
            prototypes_acc[int(y)] += H_obs.astype(np.int32)
            class_counts[int(y)] += 1
    
    # Binariser les prototypes finaux
    prototypes_bin = np.sign(prototypes_acc).astype(np.int8)
    return prototypes_bin, prototypes_acc, class_counts
```

### 4.2 Mise à jour incrémentale (Tasks 2, 3)

```python
def update_hdc_incremental(prototypes_acc, class_counts, H_level, H_pos,
                            new_dataloader, feature_bounds, n_levels, D):
    """
    Met à jour les prototypes avec de nouveaux exemples.
    Pas de recalcul depuis zéro : accumulation pure → O(N_new × F × D).
    Pas d'oubli catastrophique par construction.
    """
    for x_batch, y_batch in new_dataloader:
        for x, y in zip(x_batch, y_batch):
            H_obs = encode_observation(x.numpy(), H_level, H_pos,
                                        feature_bounds, n_levels, D)
            prototypes_acc[int(y)] += H_obs.astype(np.int32)
            class_counts[int(y)] += 1
    
    # Re-binariser
    prototypes_bin = np.sign(prototypes_acc).astype(np.int8)
    return prototypes_bin, prototypes_acc, class_counts
```

> ✅ Cette mise à jour ne nécessite pas de gradient. Elle est O(1) en mémoire additionnelle.

### 4.3 Inférence

```python
def predict_hdc(H_obs_bin, prototypes_bin):
    """
    Classification par similarité cosinus (ou distance de Hamming).
    Sur MCU : calcul via POPCOUNT(XOR(H_obs, C_c)) pour chaque classe c.
    """
    similarities = np.dot(prototypes_bin.astype(np.float32),
                          H_obs_bin.astype(np.float32))
    return np.argmax(similarities)
```

---

## 5. Scénario de Continual Learning

### 5.1 Type de scénario
**Class-Incremental ou Domain-Incremental** : HDC gère les deux naturellement.

Pour Dataset 2 :
- **Domain-Incremental** (recommandé) : même 2 classes (faulty/normal), distribution change par type d'équipement → les prototypes s'enrichissent au fil des domaines
- **Class-Incremental** : si on considère `(faulty_pump, normal_pump, faulty_turbine, normal_turbine, ...)` comme 6 classes distinctes → prototype séparé par classe-domaine

### 5.2 Avantage fondamental de HDC pour le CL

HDC n'a **pas** de catastrophic forgetting au sens du gradient. Les prototypes sont une mémoire **additive** : chaque nouvel exemple enrichit le prototype sans effacer les exemples précédents. C'est la différence fondamentale avec les réseaux neuronaux.

**Limite** : si la distribution d'une classe change drastiquement entre domaines (ex. "faulty" pour une pompe vs une turbine sont très différents), le prototype accumulé devient un vecteur « moyen » peu discriminant → dégradation de précision inter-domaine.

---

## 6. Hyperparamètres

```yaml
# configs/hdc_config.yaml

hdc:
  D: 1024             # dimension des hypervecteurs
  n_levels: 10        # niveaux de quantification par feature
  seed: 42            # seed pour la génération des vecteurs de base

data:
  n_features: 6       # features d'entrée après preprocessing
  n_classes: 2        # faulty / normal

features:
  # Bornes pour quantification (calculées sur Task 1)
  temperature:  [min: null, max: null]   # à remplir après exploration données
  pressure:     [min: null, max: null]
  vibration:    [min: null, max: null]
  humidity:     [min: null, max: null]
  equip_enc_1:  [min: 0, max: 1]        # one-hot équipement
  equip_enc_2:  [min: 0, max: 1]

evaluation:
  seed: 42
  metrics: ["acc_final", "avg_forgetting", "bwt", "ram_peak_bytes"]
  
memory:
  target_ram_bytes: 65536     # 64 Ko cible STM32N6
  expected_ram_bytes: 12288   # ~12 Ko estimé
```

---

## 7. Comparaison avec les méthodes neuronales

| Critère | HDC | EWC + MLP | TinyOL |
|---------|-----|-----------|--------|
| Oubli catastrophique | ✅ Nul par construction | ⚠️ Réduit par λ | ⚠️ Réduit par backbone gelé |
| RAM totale | ~12 Ko | ~9 Ko | ~10 Ko |
| Entraînement MCU | ✅ Accumulation binaire | ✅ SGD FP32 | ✅ SGD FP32 |
| Besoin de gradient | ❌ Non | ✅ Oui | ✅ Oui (tête seulement) |
| Précision attendue | ⚠️ Inférieure | ✅ Bonne | ✅ Bonne |
| Interprétabilité | ✅ Haute | ⚠️ Faible | ⚠️ Faible |
| Applicable données temporelles | ⚠️ Avec feature eng. | ✅ Oui | ✅ Natif |

> **Rôle de HDC dans le projet** : borne inférieure de précision + borne supérieure de frugalité. Si HDC atteint > 90 % de précision sur Dataset 2, il constitue une solution industriellement viable pour les cas simples.

---

## 8. Extension possible — HDC avec N-gram temporal

Pour Dataset 1 (série temporelle), HDC peut être étendu avec un encodage **N-gram** :

```python
# Encoder T fenêtres temporelles consécutives avec rotation cyclique
# H_ngram(t) = ρ(H_obs(t-N+1)) ⊗ ρ²(H_obs(t-N+2)) ⊗ ... ⊗ H_obs(t)
# où ρ = rotation cyclique de 1 position (permutation fixe)
```

Cette extension est **hors scope pour l'implémentation initiale** — à considérer si HDC se montre compétitif sur Dataset 2.

---

## 9. Portabilité MCU — Checklist

| Critère | Statut | Note |
|---------|--------|------|
| Pas de FP dans la phase d'entraînement | ✅ | Accumulation INT32 + signe |
| Opérations SIMD/DSP compatibles Cortex-M55 | ✅ | XOR, POPCOUNT, ADD vectoriels |
| RAM < 12 Ko | ✅ estimé | À vérifier |
| Inférence par similarité (pas de softmax) | ✅ | Dot product INT8 |
| Pas de dépendance PyTorch à l'inférence | ✅ | NumPy pur → C portable |
| Vecteurs de base exportables en tableaux C | ✅ | Tableaux statiques |

---

## 10. Fichiers à produire

```
src/models/hdc/
├── __init__.py
├── hdc_classifier.py     ← encodage + prototypes + inférence
└── base_vectors.py       ← génération et sauvegarde des vecteurs de base

configs/
└── hdc_config.yaml

experiments/exp_002_hdc_dataset2/
├── config_snapshot.yaml
└── results/
    ├── metrics.json
    ├── prototype_evolution.png
    └── similarity_matrix.png
```

---

## 11. Références

- Benatti, S., et al. (2019). Online learning with brain-inspired hyperdimensional computing for EMG-based gesture recognition. *IEEE BioCAS*.
- De Lange, M., et al. (2021). A continual learning survey: Defying forgetting in classification tasks. *TPAMI*.
- Capogrosso, L., et al. (2023). A machine learning-oriented survey on tiny machine learning. *IEEE Access*.
