# M1 — Spécification d'implémentation : TinyOL + tête OtO

> **Statut** : Priorité 1 (après M2 et M3)  
> **Dataset** : Dataset 1 — Large Industrial Pump Maintenance  
> **Référence** : Ren et al. (2021), *TinyOL: TinyML with Online-Learning on Microcontrollers*  
> **Taxonomie CL** : Architecture-based (backbone gelé + tête entraînable)

---

## 1. Principe de la méthode

TinyOL sépare le réseau en deux parties au moment du déploiement :

1. **Backbone (Feature Extractor)** — pré-entraîné hors-device en batch, puis **gelé** (*frozen*). Stocké en Flash sur le MCU. Ne reçoit jamais de gradient lors du déploiement.
2. **Tête OtO (One-to-One layer)** — couche légère entraînable, mise à jour en ligne échantillon par échantillon via SGD. C'est la seule partie qui évolue après déploiement.

**Mécanisme anti-oubli** : le gel du backbone garantit que les représentations apprises ne dégradent pas. L'oubli résiduel au niveau de la tête OtO est limité par la petite taille de celle-ci.

**Analogie embarquée** : le backbone est un programme en Flash (lecture seule), la tête OtO est un état en RAM (lecture/écriture).

---

## 2. Architecture détaillée

### 2.1 Backbone — Autoencoder MLP

```
Input: [batch, 25]   ← 25 features statistiques par fenêtre
   │
   ├── Encodeur
   │   ├── Linear(25 → 32)  + ReLU   # MEM: 800 B @ FP32 / 200 B @ INT8
   │   ├── Linear(32 → 16)  + ReLU   # MEM: 512 B @ FP32 / 128 B @ INT8
   │   └── Linear(16 → 8)            # MEM: 128 B @ FP32 / 32 B @ INT8
   │       → embedding z ∈ ℝ⁸
   │
   └── Décodeur (utilisé uniquement lors du pré-entraînement)
       ├── Linear(8  → 16)  + ReLU
       ├── Linear(16 → 32)  + ReLU
       └── Linear(32 → 25)
           → reconstruction x̂ ∈ ℝ²⁵

Perte pré-entraînement : MSE(x, x̂)
```

**Paramètres backbone (encodeur seul, après gel)** :
- Linear(25→32) : 25×32 + 32 = 832 params
- Linear(32→16) : 32×16 + 16 = 528 params
- Linear(16→8)  : 16×8  + 8  = 136 params
- **Total encodeur : 1 496 params → ~5,8 Ko @ FP32 / ~1,5 Ko @ INT8**

### 2.2 Tête OtO (Online learning head)

```
Input: [batch, 9]    ← embedding z (8D) + MSE scalaire = 9D
   │
   └── Linear(9 → 1) + Sigmoid   # MEM: 36 B @ FP32 / 9 B @ INT8
       → probabilité de panne ŷ ∈ [0, 1]
```

**Paramètres tête OtO** :
- Linear(9→1) : 9×1 + 1 = 10 params → **40 octets @ FP32**

### 2.3 Budget mémoire total estimé

| Composant | FP32 | INT8 | Stockage |
|-----------|------|------|---------|
| Encodeur (backbone gelé) | 5,8 Ko | 1,5 Ko | Flash |
| Tête OtO (poids) | 40 B | 10 B | RAM |
| Activations (forward pass) | ~512 B | ~128 B | RAM (temporaire) |
| Gradient tête OtO (backward) | 40 B | N/A | RAM (temporaire) |
| MSE scalaire (feature) | 4 B | — | RAM |
| **TOTAL RAM dynamique** | **~600 B** | **~140 B** | **RAM** |
| **TOTAL Flash (backbone)** | **5,8 Ko** | **1,5 Ko** | **Flash** |

> ✅ Largement dans la cible de 64 Ko RAM. Marge disponible pour le buffer UINT8 (extension).

---

## 3. Pipeline de feature engineering

Les données brutes du Dataset 1 (séries temporelles) doivent être transformées en vecteurs de features fixes avant d'entrer dans le réseau.

### 3.1 Fenêtrage glissant

```python
WINDOW_SIZE = 32        # points temporels par fenêtre
STEP_SIZE = 16          # chevauchement 50%
SAMPLING_RATE = None    # à déterminer depuis le dataset
```

### 3.2 Features extraites par fenêtre (25 features)

Pour chaque canal (température, vibration, pression, RPM — 4 canaux si disponibles, sinon adapter) :

| Feature | Formule | Pertinence PdM |
|---------|---------|----------------|
| `mean` | μ = (1/N)Σxᵢ | Valeur centrale |
| `std` | σ = √((1/N)Σ(xᵢ-μ)²) | Dispersion |
| `rms` | √((1/N)Σxᵢ²) | Énergie signal |
| `kurtosis` | E[(x-μ)⁴]/σ⁴ | Détection chocs |
| `peak` | max(\|xᵢ\|) | Amplitude max |
| `crest_factor` | peak / rms | Forme signal |

→ 6 features × 4 canaux = 24 + 1 feature globale (label temporel normalisé) = **25 features**

### 3.3 Normalisation

```python
# Normalisation par z-score sur le set d'entraînement batch (offline)
# Les statistiques (mean, std) sont stockées dans configs/ pour le portage MCU
NORMALIZE_MEAN = [...]  # à calculer lors du pré-entraînement
NORMALIZE_STD  = [...]  # à calculer lors du pré-entraînement
```

> ⚠️ La normalisation doit utiliser des statistiques fixes calculées une fois. Ne pas recalculer en ligne (trop coûteux sur MCU et source de drift).

---

## 4. Scénario de Continual Learning

### 4.1 Type de scénario
**Domain-Incremental** avec drift temporel naturel : les données de pompe évoluent au cours du temps (usure progressive), les frontières entre domaines ne sont pas explicitement définies.

### 4.2 Construction du stream CL

```python
# Découper le dataset chronologiquement en T tâches (non mélangées)
N_TASKS = 3             # T1: pompe saine / T2: usure naissante / T3: pré-panne
TASK_LABELS_AVAILABLE = False   # task-free scenario (plus réaliste)
```

### 4.3 Protocole d'évaluation

```
Pour chaque tâche tᵢ (i = 1..T) :
  1. Entraîner le modèle sur le stream de tᵢ (online, 1 pass)
  2. Évaluer sur TOUS les domaines vus jusqu'à tᵢ
  3. Enregistrer : acc(tᵢ, tⱼ) pour j ≤ i

Métriques finales (après T) :
  → Average Accuracy (AA)
  → Average Forgetting (AF)
  → Backward Transfer (BWT)
  → RAM peak mesurée
```

---

## 5. Phase de pré-entraînement (offline, sur PC)

```python
# Objectif : entraîner le backbone autoencoder sur données normales uniquement
# (pas de pannes — apprentissage de la représentation "normale")

PRE_TRAIN_EPOCHS = 50
PRE_TRAIN_LR = 1e-3
PRE_TRAIN_BATCH_SIZE = 64
PRE_TRAIN_LOSS = "mse"
PRE_TRAIN_OPTIMIZER = "adam"

# Données utilisées : première portion du dataset (avant drift)
# Label utilisé : aucun (apprentissage non supervisé de l'autoencoder)
```

**Sortie du pré-entraînement** :
- `backbone_encoder.pt` — poids de l'encodeur (à geler)
- `normalize_stats.yaml` — mean/std de normalisation
- `pretrain_loss_curve.png` — courbe de convergence

---

## 6. Phase d'apprentissage incrémental (online)

```python
# Mise à jour de la tête OtO uniquement
OTO_LR = 1e-2               # taux d'apprentissage tête OtO
OTO_OPTIMIZER = "sgd"       # SGD pur (pas d'Adam — trop coûteux MCU)
OTO_LOSS = "bce"            # Binary Cross-Entropy
OTO_UPDATE_FREQ = 1         # mise à jour à chaque échantillon
OTO_MOMENTUM = 0.0          # pas de momentum (mémoire supplémentaire)
```

**Pseudo-code de la boucle online** :
```python
for x, y in stream:
    # 1. Forward backbone (gelé, pas de gradient)
    with torch.no_grad():
        z = encoder(normalize(x))
        x_hat = decoder(z)
        mse = F.mse_loss(x_hat, normalize(x))
    
    # 2. Construction feature OtO
    oto_input = torch.cat([z, mse.unsqueeze(0)])  # [9]
    
    # 3. Forward + backward tête OtO
    y_hat = oto_head(oto_input)
    loss = F.binary_cross_entropy(y_hat, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 4. Discard x, y (pas de stockage)
```

---

## 7. Extension — Buffer de rejeu latent UINT8

> Cette extension s'appuie sur Ravaglia et al. (2021) — QLR-CL.

```python
# Stocker les embeddings z en UINT8 au lieu de FP32 → ×4 de compression
REPLAY_BUFFER_SIZE = 50     # nombre d'exemples stockés
REPLAY_BUFFER_DTYPE = "uint8"
REPLAY_EMBED_DIM = 8        # dimension de z

# RAM buffer : 50 × 8 × 1 octet = 400 octets (vs 1 600 octets @ FP32)
# + labels : 50 × 1 = 50 octets
# TOTAL buffer : ~450 octets

# Quantification : z_uint8 = clamp((z - z_min) / (z_max - z_min) * 255, 0, 255)
# à calibrer sur les données de pré-entraînement
```

**Question ouverte pour Dorra** : la dégradation de précision due à la quantification UINT8 des embeddings est-elle similaire à celle observée par Ravaglia (−0,26 % sur CIFAR) sur des données industrielles ? À mesurer expérimentalement.

---

## 8. Portabilité MCU — Checklist de conception

| Critère | Statut | Note |
|---------|--------|------|
| Pas de boucle for Python dans le forward | ✅ | Opérations matricielles PyTorch uniquement |
| Pas d'allocation dynamique dans le forward | ✅ | Tailles fixes (WINDOW_SIZE, embed_dim constants) |
| SGD sans momentum | ✅ | Adam interdit (état supplémentaire en RAM) |
| Backbone exportable ONNX / TFLite | À vérifier | Après implémentation |
| Activations ReLU (INT8-friendly) | ✅ | Pas de GELU, SiLU, etc. |
| Normalisation offline (stats fixes) | ✅ | Pas de BatchNorm online |
| Taille modèle < 64 Ko (total) | ✅ estimé | À mesurer avec `memory_profiler.py` |

---

## 9. Fichiers à produire

```
src/models/tinyol/
├── __init__.py
├── autoencoder.py          ← backbone encodeur + décodeur
└── oto_head.py             ← tête OtO + boucle online

src/data/
├── pump_dataset.py         ← loader + fenêtrage + features

configs/
└── tinyol_config.yaml      ← tous les hyperparamètres

experiments/exp_003_tinyol_dataset1/
├── config_snapshot.yaml
└── results/
    ├── metrics.json
    ├── pretrain_loss.png
    └── cl_accuracy_matrix.png
```

---

## 10. Références

- Ren, H., Anicic, D., & Runkler, T. A. (2021). TinyOL: TinyML with online-learning on microcontrollers. *arXiv*.
- Ravaglia, M., et al. (2021). Continual learning on microcontrollers with quantized latent replays. *NeurIPS Workshop*.
- Capogrosso, L., et al. (2023). A machine learning-oriented survey on tiny machine learning. *IEEE Access*.
