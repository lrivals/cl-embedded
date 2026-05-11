# S1007 — Simulateur de capteur & évaluation en ligne

| Champ | Valeur |
|-------|--------|
| **ID** | S1007 |
| **Sprint** | Sprint 16 — Semaine 1b (20–27 mai 2026) |
| **Priorité** | Haute |
| **Durée estimée** | 6h |
| **Dépendances** | S1001 ✅ (toolchain OK), datasets Phase 1 disponibles (CWRU, monitoring) |
| **Fichiers cibles** | `scripts/sensor_sim.py`, `src/evaluation/online_metrics.py` |

---

## Objectif

Fournir le lien entre les datasets Python (Phase 1) et le firmware MCU (Sprint 16) :
un simulateur de capteur qui injecte des données via UART, et des métriques
d'évaluation adaptées au contexte online (streaming, pas de batch complet).

**Pipeline cible** :

```
Dataset CWRU/monitoring (Python)
  → sensor_sim.py (frames UART)
  → firmware STM32 (inférence + décision)
  → UART réponse (label prédit)
  → online_metrics.py (AUROC streaming, accuracy cumulée)
```

---

## Contexte

En Phase 1, l'évaluation se fait en batch : on charge tout le jeu de test en RAM.
En Phase 2 sur MCU, le modèle voit les données une par une (online, 64 Ko RAM max).
Il faut donc :
1. Un émetteur Python qui simule l'arrivée séquentielle de données capteur via UART
2. Des métriques qui calculent l'accuracy/AUROC de façon incrémentale (sans stocker
   toutes les prédictions en mémoire)

---

## Protocole de communication UART

Format d'une trame (binaire, little-endian) :

```
[MAGIC: 0xABCD (2 octets)] [N_FEATURES: uint8 (1 octet)] [features: float32 × N (4N octets)] [label: uint8 (1 octet)] [CRC8 (1 octet)]
```

Réponse du firmware :
```
[pred_label: uint8 (1 octet)] [confidence: float32 (4 octets)] [latency_us: uint32 (4 octets)]
```

Taille trame (monitoring, 5 features) : 2 + 1 + 20 + 1 + 1 = **25 octets**  
Débit à 115200 baud : ~460 trames/s (largement suffisant)

---

## Sous-tâches

### 1. Créer `scripts/sensor_sim.py`

Voir implémentation dans `scripts/sensor_sim.py`.

Usage :
```bash
# Mode simulation UART (board connectée)
python scripts/sensor_sim.py \
    --dataset cwru \
    --port /dev/ttyUSB0 \
    --baud 115200 \
    --n-samples 500

# Mode dry-run (pas de board — valide le protocole en loopback)
python scripts/sensor_sim.py \
    --dataset monitoring \
    --dry-run \
    --n-samples 100
```

### 2. Créer `src/evaluation/online_metrics.py`

Voir implémentation dans `src/evaluation/online_metrics.py`.

Métriques calculées de façon incrémentale :
- `OnlineAccuracy` — accuracy cumulée, mise à jour sample par sample
- `OnlineAUROC` — approximation par fenêtre glissante (buffer ≤ 1000 samples)
- `OnlineForgetting` — chute d'accuracy entre tâches successives

Usage :
```python
from src.evaluation.online_metrics import OnlineAccuracy, OnlineAUROC

acc = OnlineAccuracy()
auroc = OnlineAUROC(window_size=500)

for y_true, y_pred, y_score in stream:
    acc.update(y_true, y_pred)
    auroc.update(y_true, y_score)

print(f"Accuracy: {acc.compute():.3f}")
print(f"AUROC: {auroc.compute():.3f}")
```

### 3. Validation end-to-end (dry-run)

```bash
# Test du protocole sans board (loopback)
python scripts/sensor_sim.py \
    --dataset cwru \
    --dry-run \
    --n-samples 100 \
    --verbose

# Sortie attendue :
# [0/100] features=[0.12, -0.34, ...] label=0 → OK (loopback)
# ...
# Accuracy (loopback): 1.000 (protocol validation)
```

### 4. Validation end-to-end (board connectée)

```bash
# Avec NUCLEO-F439ZI flashée avec pipeline.c
python scripts/sensor_sim.py \
    --dataset monitoring \
    --port /dev/ttyUSB0 \
    --n-samples 200 \
    --output results/sensor_sim_monitoring_200.json
```

**Sorties** :
- `results/sensor_sim_monitoring_200.json` : accuracy, AUROC, latency stats
- Log console : une ligne par trame + résumé final

---

## Critères d'acceptation

- [x] `python scripts/sensor_sim.py --dry-run --n-samples 100` se termine sans erreur
- [x] Protocole UART : 0 erreurs CRC sur 100 trames en loopback
- [x] `OnlineAccuracy.compute()` == `sklearn.accuracy_score` sur même séquence (tol 1e-9)
- [x] `OnlineAUROC.compute()` ≈ `sklearn.roc_auc_score` (tol 0.01 sur 500 samples)
- [ ] `scripts/sensor_sim.py --port /dev/ttyUSB0` envoie 10 trames sans timeout — en attente accès board
  (test board optionnel — marqué `SKIP` si board non connectée en CI)

---

## Questions ouvertes

- Baud rate optimal : 115200 vs 921600 pour des mesures de latency précises ?
- Format confidence : float32 ou Q8.8 fixe (économiser 2 octets par réponse) ?
