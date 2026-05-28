# S2002 — Protocole UART v3 : réponse 21B avec métriques CL

| Champ | Valeur |
|-------|--------|
| **Sprint** | 20 |
| **Priorité** | 🔴 Critique |
| **Statut** | ✅ Terminé |
| **Durée estimée** | 4h |
| **Dépendances** | S2001 (EWC consolidate), S1905 (métriques firmware ✅), S1906 (spec v3 ✅) |
| **Fichiers cibles** | `firmware/stm32f4_blink/src/pipeline.c`, `scripts/sensor_stream.py` |
| **Référence** | `docs/sprints/sprint_19/S1906_protocol_v3.md` (spec existante) |

---

## Contexte

Le protocole v2 (Sprint 18, validé board) envoie une réponse de **14 octets** : `[pred:u8][conf:f32][lat_us:u32][ram_b:u32][thr_ips:f32][status:u8]`.

La spec v3 (S1906) a défini une extension à **21 octets** ajoutant `[acc:f32][auroc:f32][forgetting:f32]` — mais l'implémentation firmware et le parser Python n'ont pas été faits.

---

## Ce qu'il faut modifier

### Firmware — `pipeline.c`

Étendre `uart_send_response()` pour encoder 21B :

```c
/* Protocol v3 response — 21 octets little-endian
 * [pred:u8][conf:f32][lat_us:u32][ram_b:u32][thr_ips:f32][status:u8]
 * [acc:f32][auroc:f32][forgetting:f32]
 * MEM: 21 B stack frame (réponse statique, pas de malloc)
 */
void uart_send_response_v3(uint8_t pred, float conf,
                            uint32_t lat_us, uint32_t ram_b, float thr_ips,
                            uint8_t status, MetricsSnapshot *snap);
```

- Encoder les floats en little-endian (memcpy de `float` vers `uint8_t[4]`)
- Garder compatibilité v2 : si `snap == NULL`, envoyer 14B (VERSION=0x02)
- VERSION=0x03 dans le premier octet du header frame

### Python — `sensor_stream.py`

Étendre `parse_response()` pour détecter VERSION et parser 14B ou 21B :

```python
def parse_response(data: bytes) -> dict:
    if len(data) == 14:  # v2
        ...
    elif len(data) == 21:  # v3
        *v2_fields, acc, auroc, forgetting = struct.unpack('<BfIIfBfff', data)
        return {**v2_dict, 'acc': acc, 'auroc': auroc, 'forgetting': forgetting}
```

---

## Tests

| Test | Type | Assertion |
|------|------|-----------|
| `test_protocol_v3_length` | Unity C | `uart_send_response_v3()` émet exactement 21 B |
| `test_protocol_v3_fields` | Unity C | acc/auroc/forgetting décodés identiques à l'entrée (±1e-6) |
| `test_sensor_stream_parse_v3` | pytest | `parse_response(21B)` retourne dict avec clés acc/auroc/forgetting |
| `test_sensor_stream_backward_compat` | pytest | `parse_response(14B)` ne crash pas, retourne dict sans clés métriques |

---

## Vérification

- [ ] Renode simulation : asserter les 21 octets de réponse sur frame test
- [ ] `sensor_stream.py --dry-run` : log affiche `acc=X auroc=Y forgetting=Z`
- [ ] CI `.github/workflows/firmware.yml` : ajouter step assert réponse v3 sur Renode
