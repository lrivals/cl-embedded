# S3403 — Expérience board débit/buffer

| Champ | Valeur |
|-------|--------|
| **Sprint** | 34 |
| **Priorité** | 🔴 Critique |
| **Statut** | ✅ Implémenté (board réelle) |
| **Durée estimée** | 4h |
| **Dépendances** | S3401 (`streaming_model.py`) · S3402 (`ring_buffer.c/.h`) |
| **Fichiers cibles** | `scripts/sensor_stream.py`, `experiments/exp_S34_streaming/` |
| **Références** | `sensor_stream.py:552` (`--rate-hz`), `scripts/board_experiment_recorder.py` (pattern enregistrement) |

---

## Contexte

Valide empiriquement le modèle de streaming (S3401) et l'abstraction buffer (S3402) sur la
NUCLEO-F439ZI réelle : trouve le **point de saturation** (drop de trame / timeout / erreur
CRC) en balayant `rate_hz` et stride, et mesure le `.bss` réel par configuration de fenêtre
`W`. Aucun chiffre n'est inventé tant que la board n'a pas tourné.

---

## Spec

- Étendre `sensor_stream.py` (déjà paramétré par `--rate-hz`, ligne 552) avec un mode
  balayage : boucle sur une liste de `(rate_hz, stride)` issue de
  `configs/streaming_profile.yaml`, envoie les trames, capture latence DWT (réponse UART
  existante, champ `lat_us`) et détecte les anomalies (timeout, CRC8 invalide, trame
  manquante par comptage de séquence).
- Pour chaque config, mesurer `.bss` via `arm-none-eabi-size` (la taille du ring buffer
  dépend de `W`, donc varie par config de build).
- Identifier le **point de saturation** : première config où `debit_streaming > debit_max`
  (prédit par S3401) ET confirmée par des drops/timeouts réels.

```json
// experiments/exp_S34_streaming/rate{XX}_stride{YY}.json
{
  "rate_hz": 100,
  "stride": 1,
  "window": 5,
  "latence_dwt_us": "à mesurer",
  "drops": "à mesurer",
  "timeouts": "à mesurer",
  "bss_bytes": "à mesurer",
  "saturation_atteinte": "à mesurer"
}
```

**Règles** :
- **RAM profiling par config de buffer** obligatoire (règle CLAUDE.md : nouveau chemin
  d'exécution mesuré).
- Champs `"à mesurer"` tant que l'exécution board réelle n'a pas eu lieu — aucun chiffre
  inventé.

---

## Vérification

```bash
python scripts/sensor_stream.py --port /dev/ttyACM0 --sweep configs/streaming_profile.yaml \
    --output experiments/exp_S34_streaming/
arm-none-eabi-size firmware/stm32f4_blink/build/stm32f4_blink.elf
```
