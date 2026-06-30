# S2903 — Intégration `pipeline.c` : FLAGS HDC_INT8 + TINYOL_INT8

| Champ | Valeur |
|-------|--------|
| **Sprint** | 29 |
| **Priorité** | 🔴 Critique — bloquant pour S2904 et S2905 |
| **Statut** | ✅ Implémenté (12 juin 2026) — FLAGS `0x60`/`0xC0` retenus (les `0x22`/`0x81` du doc S2900 entrent en collision avec PROFILING `0x02` / UPDATE `0x01`). S2902 (`tinyol_int8.c/.h`) implémenté en parallèle pour câbler la branche TinyOL. Branches insérées AVANT le check INT8_MODE (`0x60`/`0xC0` ont le bit `0x40`). `make test` : 79 tests, 2 échecs TinyOL FP32 pré-existants hors périmètre, 0 régression. Board flashée (Verified OK). |
| **Durée estimée** | 2h |
| **Dépendances** | S2901 ✅ (`hdc_int8.c`) · S2902 ✅ (`tinyol_int8.c`) · `firmware/stm32f4_blink/src/pipeline.c` ✅ (v3) · `scripts/sensor_stream.py` ✅ |
| **Fichiers cibles** | `firmware/stm32f4_blink/src/pipeline.c`, `scripts/sensor_stream.py` |

---

## Contexte

Sprint 26 a noté que le byte FLAGS est **saturé** (8 bits tous utilisés). Tout nouveau mode doit utiliser une combinaison de bits existants sans collision. Le tableau des FLAGS actuels :

| Flag | Valeur | Mode |
|------|:------:|------|
| `PROTO_FLAG_EWC_MODE` | `0x10` | EWC FP32 |
| `PROTO_FLAG_HDC_MODE` | `0x20` | HDC FP32 |
| `PROTO_FLAG_INT8_MODE` | `0x40` | INT8 (EWC seul, S23) |
| `PROTO_FLAG_TINYOL_MODE` | `0x80` | TinyOL FP32 |
| `PROTO_FLAG_RUL_MODE` | `0x50` | EWC Régression (Sprint 26) |
| `PROTO_FLAG_MULTICLASS_MODE` | `0x30` | EWC Multi-class (Sprint 26) |
| `PROTO_FLAG_MAHA_MODE` | `0x08` | Mahalanobis FP32 |

**Nouvelles combinaisons proposées** :
- `PROTO_FLAG_HDC_INT8 = 0x60` = HDC_MODE `0x20` | INT8_MODE `0x40` — vérifier absence de collision
- `PROTO_FLAG_TINYOL_INT8 = 0xC0` = TINYOL_MODE `0x80` | INT8_MODE `0x40` — vérifier absence de collision

> `TODO(dorra)` : Si 0x60 ou 0xC0 entrent en collision avec un routing existant, envisager Protocol V4 avec FLAGS sur 2 octets.

---

## Modifications `pipeline.c`

```c
/* Ajouter dans pipeline.h */
#define PROTO_FLAG_HDC_INT8    0x60U   /* HDC_MODE | INT8_MODE */
#define PROTO_FLAG_TINYOL_INT8 0xC0U   /* TINYOL_MODE | INT8_MODE */

/* Dans pipeline_run() — routing if/else (vérifier avant flags simples) */
} else if (flags == PROTO_FLAG_HDC_INT8) {
    hdc_int8_encode(&g_hdc_int8, features, hv_int8);
    pred = hdc_int8_predict(&g_hdc_int8, hv_int8);
    hdc_int8_update(&g_hdc_int8, hv_int8, (int)label);

} else if (flags == PROTO_FLAG_TINYOL_INT8) {
    tinyol_int8_encode(&g_tinyol_int8, features, encoded_int8);
    pred = oto_int8_predict(&g_oto_int8, encoded_int8);
    oto_int8_update(&g_oto_int8, encoded_int8, (int)label);
```

---

## Modifications `sensor_stream.py`

```python
# Ajouter dans sensor_stream.py
MODE_FLAGS = {
    "ewc": 0x10,
    "hdc": 0x20,
    "tinyol": 0x80,
    "maha": 0x08,
    "ewc-int8": 0x40,       # existant Sprint 23
    "hdc-int8": 0x60,        # nouveau Sprint 29
    "tinyol-int8": 0xC0,     # nouveau Sprint 29
    "ewc-rul": 0x50,
    "ewc-mc": 0x30,
}
```

---

## Vérification

```bash
# Compilation sans warning
cd firmware/stm32f4_blink && make all

# Test flags sans collision (script Python)
python -c "
flags = [0x10, 0x20, 0x40, 0x80, 0x50, 0x30, 0x08, 0x60, 0xC0]
assert len(flags) == len(set(flags)), 'Collision détectée !'
print('Pas de collision ✅')
"

# Test pipeline routing (host)
make test  # vérifier 0 régression
```
