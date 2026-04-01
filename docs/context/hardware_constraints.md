# Contraintes hardware — STM32N6 (cible) et implications pour l'implémentation Python

---

## Microcontrôleur cible : STM32N6

| Propriété | Valeur | Source |
|-----------|--------|--------|
| CPU | ARM Cortex-M55 | ST Microelectronics |
| Fréquence | À confirmer (typiquement 400–800 MHz pour N6) | TODO(dorra) |
| RAM SRAM | ~64 Ko (cible stage) | Contrainte projet |
| Flash | 512 Ko | ST datasheet |
| NPU | Intégré (NeuralART Turbo) | ST Microelectronics |
| Mode NPU | **Inférence uniquement** — ✅ Confirmé | Dorra Ben Khalifa |
| FPU | SP FP32 (Cortex-M55 = ARMv8.1-M + Helium) | ARM |
| Instructions DSP/SIMD | Helium (M-Profile Vector Extension) | ARM |
| Latence cible | < 100 ms par inférence + update | Spec stage |

> **Règle fondamentale** : le NPU accélère uniquement le forward pass (inférence). La backpropagation s'exécute sur le Cortex-M55 en logiciel, en FP32. Toute implémentation Python doit refléter cette séparation.

---

## Budget mémoire — Allocation cible (64 Ko total)

```
┌─────────────────────────────────────────────────┐
│              RAM SRAM (~64 Ko)                  │
├─────────────────┬───────────────────────────────┤
│  Système / HAL  │  ~8–15 Ko (estimation)         │
│  Pile (stack)   │  ~4 Ko                         │
├─────────────────┼───────────────────────────────┤
│  Modèle (poids) │  Variable (voir tableau)       │
│  Activations    │  Variable (forward pass)       │
│  Buffer CL      │  Variable (replay/Fisher)      │
│  Données I/O    │  ~1–2 Ko (fenêtre capteur)     │
└─────────────────┴───────────────────────────────┘
Marge cible pour le modèle + CL : ~40–48 Ko
```

### Empreinte mémoire estimée par modèle

| Modèle | Poids FP32 | Overhead CL | Activations | Total estimé | Marge/64 Ko |
|--------|-----------|-------------|-------------|-------------|-------------|
| M2 EWC + MLP | ~3 Ko | ~6 Ko (Fisher×2) | ~512 B | ~10 Ko | ✅ 54 Ko libres |
| M3 HDC | ~0 (vecteurs base en Flash) | 0 | ~4 Ko | ~12 Ko | ✅ 52 Ko libres |
| M1 TinyOL | ~6 Ko (backbone Flash) | ~40 B (tête OtO) | ~512 B | ~7 Ko RAM | ✅ 57 Ko libres |
| M1 + buffer UINT8 | ~6 Ko | ~40 B + ~450 B buffer | ~512 B | ~8 Ko | ✅ 56 Ko libres |

> ✅ Les trois modèles sont confortablement dans les 64 Ko. La mesure précise via `tracemalloc` sur PC donnera les chiffres réels à reporter dans le manuscrit.

---

## Implications pour le code Python (PC)

### Règle 1 — Séparation backbone / tête entraînable

```python
# Sur PC : utiliser torch.no_grad() pour simuler le comportement MCU
# où le backbone est stocké en Flash (lecture seule)
with torch.no_grad():
    features = backbone(x)        # ← simule l'inférence NPU
gradient_zone = trainable_head(features)   # ← simule la backprop Cortex-M55
```

### Règle 2 — Optimizer SGD uniquement (pas Adam)

Adam maintient deux états supplémentaires par paramètre (m et v), doublant l'empreinte mémoire des poids entraînables. Sur Cortex-M55 avec 64 Ko total, c'est incompatible.

```python
# ✅ Correct pour MCU
optimizer = torch.optim.SGD(trainable_head.parameters(), lr=0.01, momentum=0.9)

# ❌ Interdit pour MCU (sauf si explicitement justifié)
# optimizer = torch.optim.Adam(...)
```

### Règle 3 — Annotations mémoire obligatoires

```python
class TinyOLEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # MEM: Linear(25→32) = 832 params = 3 328 B @ FP32 / 832 B @ INT8
        self.fc1 = nn.Linear(25, 32)
        # MEM: Linear(32→16) = 528 params = 2 112 B @ FP32 / 528 B @ INT8
        self.fc2 = nn.Linear(32, 16)
        # MEM: Linear(16→8) = 136 params = 544 B @ FP32 / 136 B @ INT8
        self.fc3 = nn.Linear(16, 8)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))   # MEM activations: 32 × 4 = 128 B @ FP32
        x = F.relu(self.fc2(x))   # MEM activations: 16 × 4 = 64 B @ FP32
        return self.fc3(x)         # MEM activations: 8 × 4 = 32 B @ FP32
```

### Règle 4 — Pas de BatchNorm, pas de LayerNorm

Ces couches requièrent soit un mini-batch (BatchNorm), soit des statistiques en ligne qui varient (LayerNorm). Sur MCU avec contrainte de RAM, les statistiques de normalisation doivent être **calculées offline et fixées**.

```python
# ✅ Normalisation offline (stats fixes en Flash)
x_normalized = (x - torch.tensor(MEAN)) / torch.tensor(STD)

# ❌ Interdit
# self.bn = nn.BatchNorm1d(32)
```

### Règle 5 — Activations ReLU uniquement

GELU, SiLU, Mish et autres activations modernes nécessitent des calculs en virgule flottante complexes ou des tables de lookup larges. ReLU = `max(0, x)` → 1 instruction sur MCU.

---

## Pipeline de déploiement MCU (Phase 2)

Cette section décrit le chemin prévu de PC → MCU pour guider les choix d'implémentation Python.

```
PC (Python/PyTorch)
    │
    ├── 1. Entraînement batch du backbone (autoencoder / préentraînement EWC)
    │       → fichier .pt (poids FP32)
    │
    ├── 2. Export ONNX
    │       torch.onnx.export(backbone, ...)
    │
    ├── 3. Conversion TFLite Micro (via TFLite Converter ou AI Cube ST)
    │       → Quantification PTQ INT8 du backbone
    │       → fichier .tflite
    │
    ├── 4. Génération code C (ST AI Cube / tflite-micro)
    │       → tableaux de poids en C (Flash)
    │
    └── 5. Intégration MCU
            → backbone en Flash (inférence NPU INT8)
            → tête/Fisher/buffer en RAM (update Cortex-M55 FP32)
```

> **Implication pour Phase 1 (PC)** : le backbone doit être exportable ONNX sans opérations custom. Vérifier la compatibilité ONNX de chaque couche lors de l'implémentation.

---

## Outils de portage (Phase 2 — pour référence)

| Outil | Rôle | Compatibilité STM32N6 |
|-------|------|----------------------|
| STM32Cube.AI | Conversion réseau → code C optimisé | ✅ Officiel ST |
| TFLite Micro | Inférence INT8 sur Cortex-M | ✅ Avec CMSIS-NN |
| CMSIS-NN | Kernels INT8 optimisés ARM | ✅ Natif Cortex-M55 |
| X-CUBE-AI | Plugin STM32CubeIDE | ✅ Officiel ST |

---

## Carte de développement (Phase 1 — si nécessaire pour validation rapide)

Bien que la cible finale soit le STM32N6, des validations intermédiaires peuvent être effectuées sur :

| Carte | MCU | RAM | NPU | Latence backprop |
|-------|-----|-----|-----|-----------------|
| **STM32N6 (cible)** | Cortex-M55 | 64 Ko | ✅ Inf-only | À mesurer |
| Nucleo-F439ZI | Cortex-M4 @ 180 MHz | 256 Ko | ❌ | Référence Benatti (< 100 ms) |
| Arduino Nano 33 BLE | Cortex-M4 @ 64 MHz | 256 Ko | ❌ | Référence TinyOL (~10 % overhead) |
