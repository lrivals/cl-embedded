# S2711 — Extension protocole UART : réponse duale 25 octets dans `sensor_stream.py`

| Champ | Valeur |
|-------|--------|
| **Sprint** | 27 |
| **Priorité** | 🔴 Critique |
| **Statut** | ✅ Implémenté — `sensor_stream.py` étendu (flag 0x70, format 25 B, parser, `--model dual`), validé board |
| **Durée estimée** | 1h |
| **Dépendances** | S2701–S2703 ✅ (DUAL_MODE firmware), `scripts/sensor_stream.py` (protocole v3 existant) |
| **Fichiers cibles** | `scripts/sensor_stream.py` |
| **Référence** | Constantes `FRAME_FLAGS_RUL_MODE = 0x50`, `RESPONSE_V3_FMT`, `parse_response()` dans `sensor_stream.py` |

---

## Contexte

`sensor_stream.py` est le module Python central du protocole UART : il contient les constantes de flags, le builder de trames, et le parser de réponses. Il est utilisé par tous les scripts d'expérience board (`simulate_rul_board.py`, `simulate_multiclass_board.py`, `board_experiment_recorder.py`).

Sprint 27 ajoute le support du mode DUAL : constante flag `0x70`, format de réponse 25 octets, et une option `--model dual` dans la CLI.

**Règle** (`CLAUDE.md`) : toute modification du protocole UART dans `pipeline.c` doit être reflétée simultanément dans `sensor_stream.py`.

---

## Format de réponse DUAL_MODE (25 octets)

```
Offset  Taille  Type  Champ         Description
  0       1 B   u8    pred_fault    Classe faute prédite ∈ [0, N_CLASSES-1]
  1       4 B   f32   conf_fault    Confiance softmax du modèle EWC_MC
  5       4 B   f32   rul_pred      RUL prédit par EWC_REG (normalisé [0,1] sur board)
  9       4 B   u32   lat_us        Latence combinée DWT (µs)
 13       4 B   f32   f1_macro      OnlineF1Macro accumulé (EWC_MC)
 17       4 B   f32   rmse_rul      OnlineRMSE accumulé (EWC_REG)
 21       4 B   f32   forgetting    ForgettingTracker AF moyen
         25 B   total
```

Python struct format : `"<BffIfff"` (little-endian, u8 + f32 + f32 + u32 + f32 + f32 + f32).

---

## Modifications `sensor_stream.py`

### 1. Constantes à ajouter

```python
# Sprint 27 — DUAL_MODE : EWC_REG (RUL) + EWC_MC (faute) simultané
FRAME_FLAGS_DUAL_MODE   = (FRAME_FLAGS_EWC_MODE | FRAME_FLAGS_HDC_MODE | FRAME_FLAGS_INT8_MODE)  # 0x70

RESPONSE_DUAL_FMT  = "<BffIfff"                        # 25 B
RESPONSE_DUAL_SIZE = struct.calcsize(RESPONSE_DUAL_FMT)  # doit valoir 25
```

### 2. Mise à jour de `parse_response()`

Ajouter une branche pour la taille 25 **avant** la branche 21 (v3) pour éviter toute confusion :

```python
def parse_response(data: bytes) -> dict:
    if len(data) == RESPONSE_DUAL_SIZE:
        # Réponse 25 B — DUAL_MODE
        pred_fault, conf_fault, rul_pred, lat_us, f1_macro, rmse_rul, forgetting = \
            struct.unpack(RESPONSE_DUAL_FMT, data)
        return {
            "mode":        "dual",
            "pred":        int(pred_fault),
            "confidence":  float(conf_fault),
            "rul_pred":    float(rul_pred),          # normalisé [0,1] — multiplier ×RUL_CAP pour cycles
            "latency_us":  int(lat_us),
            "f1_macro":    float(f1_macro),
            "rmse_rul":    float(rmse_rul),
            "forgetting":  float(forgetting),
            "ram_bytes":   0,
            "throughput_ips": 0,
            "status":      STATUS_OK,
        }
    elif len(data) == RESPONSE_V3_SIZE:
        # Réponse 21 B — v3 (inchangé)
        ...
```

### 3. Mise à jour de la CLI (`main()`)

```python
parser.add_argument(
    "--model",
    choices=["mahalanobis", "ewc", "ewc-int8", "tinyol", "hdc", "rul", "multiclass", "dual"],
    default="mahalanobis",
    help="Modèle board à utiliser"
)
# ...
elif args.model == "dual":
    model_flags = FRAME_FLAGS_DUAL_MODE
    expected_response_size = RESPONSE_DUAL_SIZE
```

### 4. Mise à jour de `build_frame_v2()` (si nécessaire)

La fonction existante accepte déjà `task_id` comme paramètre. En DUAL_MODE, l'appelant passe `task_id=fault_label` — aucune modification de signature requise.

---

## Test de non-régression

```python
# Vérifier que RESPONSE_DUAL_SIZE == 25
import struct
assert struct.calcsize("<BffIfff") == 25, "Format dual incorrect"

# Vérifier que FRAME_FLAGS_DUAL_MODE == 0x70
assert FRAME_FLAGS_DUAL_MODE == 0x70

# Vérifier que les modes existants sont inchangés
assert FRAME_FLAGS_RUL_MODE == 0x50
assert FRAME_FLAGS_MULTICLASS_MODE == 0x30
```

---

## Encodage des labels en DUAL_MODE (rappel)

| Champ trame | Valeur en DUAL_MODE | Valeur hors DUAL_MODE |
|-------------|--------------------|-----------------------|
| `TASK_ID` (byte 3) | `fault_label` ∈ [0, N-1] | ID de tâche CL |
| `label` (avant CRC) | `rul_u8 = round(RUL / RUL_CAP × 255)` | label de classe ou 0/1 |

```python
# Côté Python (board_dual_pipeline.py)
RUL_CAP = 300  # CMAPSS FD001 — RUL max en cycles

def encode_dual_frame(features_9, rul_true, fault_label, flags, task_id=None):
    rul_u8    = int(round(min(rul_true, RUL_CAP) / RUL_CAP * 255))
    task_id_  = int(fault_label)   # TASK_ID réutilisé comme fault_label
    return build_frame_v2(
        features=features_9,
        label=rul_u8,
        task_id=task_id_,
        ts_ms=int(time.time() * 1000) & 0xFFFFFFFF,
        flags=flags,
    )
```

---

## Questions ouvertes

- `TODO(dorra)` : Si le PC lit `RESPONSE_DUAL_SIZE = 25` octets mais que la board envoie 21 (trame v3 reçue en non-dual mode), le parser se retrouve en décalage. Ajouter un timeout de lecture et un flush de buffer avant chaque expérience.
- `TODO(arnaud)` : `rul_pred` est retourné normalisé [0,1] par la board. Le script doit le multiplier par `RUL_CAP = 300` pour obtenir des cycles interprétables. Vérifier la cohérence avec `simulate_rul_board.py` qui fait la même conversion.
