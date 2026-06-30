# S3401 — `src/evaluation/streaming_model.py` + `configs/streaming_profile.yaml`

| Champ | Valeur |
|-------|--------|
| **Sprint** | 34 |
| **Priorité** | 🔴 Critique — bloquant pour S3403 (exp board débit/buffer) |
| **Statut** | ✅ Implémenté |
| **Durée estimée** | 3h |
| **Dépendances** | Latences DWT déjà mesurées Sprints 18-29 (`profiling.c`) |
| **Fichiers cibles** | `src/evaluation/streaming_model.py`, `configs/streaming_profile.yaml` |
| **Références** | `scripts/sensor_stream.py:552` (`--rate-hz`), `:292,401,430` (`interval_s = 1.0/rate_hz`) |

---

## Contexte

Le CR du 19 mai 2026 demande d'« évaluer la latence pour estimer combien de données on peut
streamer sur la carte en même temps, par modèle » et « étudier l'impact du stride S sur
latence perçue et charge CPU ». `sensor_stream.py` expose déjà `--rate-hz` (confirmé
ligne 552, boucle de rate-limiting lignes 292/401/430) mais **aucune étude de saturation**
débit/latence n'existe : ce module formalise les formules du CR.

---

## Spec

```python
# src/evaluation/streaming_model.py

def debit_max(latence_inf_s: float) -> float:
    """Debit_max (Hz) = 1 / latence_inf — borne supérieure de fréquence d'acquisition
    soutenable par un modèle donné, sans accumulation de retard.
    """
    return 1.0 / latence_inf_s

def debit_streaming(f_acq_hz: float, stride: int, window: int) -> float:
    """Debit_streaming (Hz) = f_acq x stride/window — fréquence effective de production
    de nouvelles fenêtres d'inférence.
    """
    return f_acq_hz * stride / window

def marge_temps_reel(debit_streaming_hz: float, debit_max_hz: float) -> dict:
    """{"ok": debit_streaming <= debit_max, "marge_pct": (debit_max - debit_streaming)/debit_max}"""
    ...

def budget_buffer_bytes(window: int, sizeof_sample: int) -> int:
    """W x sizeof(sample) — à comparer à la SRAM disponible (configs/streaming_profile.yaml)."""
    return window * sizeof_sample

def check_sram_budget(buffer_bytes: int, sram_bytes: int) -> bool:
    return buffer_bytes <= sram_bytes
```

```yaml
# configs/streaming_profile.yaml
streaming:
  f_acq_hz: <à_renseigner>        # fréquence d'acquisition capteur réelle
  window_w: <à_renseigner>        # taille de fenêtre (échantillons)
  stride_s: <à_renseigner>        # pas entre fenêtres consécutives
  sizeof_sample_bytes: 4          # float32 par défaut
  sram_budget_bytes: 65536        # 64 Ko — budget Gap 2 de référence (pas 256 Ko total)
```

**Règles** :
- `W`, `S`, `f_acq`, `sizeof(sample)`, taille SRAM → **toujours** dans
  `configs/streaming_profile.yaml`, jamais en dur dans le code (règle CLAUDE.md).
- `latence_inf_s` provient des mesures DWT réelles déjà produites par les sprints
  précédents (`profiling.c`), pas d'une estimation analytique — réutiliser les chiffres
  mesurés, ne pas les recalculer.

---

## Vérification

```bash
python -c "from src.evaluation.streaming_model import debit_max, debit_streaming, marge_temps_reel; \
dm = debit_max(0.000233); ds = debit_streaming(100, 1, 5); print(marge_temps_reel(ds, dm))"

pytest tests/test_streaming_model.py -v   # S3409 : débit_max/streaming, contrainte SRAM
```
