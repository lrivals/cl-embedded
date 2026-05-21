# S1806 — Config YAML profiling : seuils RAM, latence max, format sortie

| Champ | Valeur |
|-------|--------|
| **ID** | S1806 |
| **Sprint** | Sprint 18 — 25 mai – 1er juin 2026 |
| **Priorité** | 🟡 Secondaire |
| **Durée estimée** | 1h |
| **Dépendances** | S1804 (firmware profiling) |
| **Fichiers cibles** | `configs/profiling_config.yaml` |
| **Statut** | ✅ Implémenté |

---

## Objectif

Centraliser les seuils d'alerte et les paramètres de profiling dans un fichier YAML versionnable, lu par `profiling_reader.py` et `board_dataset_builder.py`.

---

## Fichier : `configs/profiling_config.yaml`

```yaml
PROFILING_VERSION: 2

# Seuils d'alerte
LATENCY_ALERT_MS: 10.0      # Alerte si latence > 10 ms
RAM_ALERT_BYTES: 52000      # Alerte si RAM > 52 Ko
THROUGHPUT_MIN_IPS: 10      # Alerte si throughput < 10 ips

# Format de sortie JSON
OUTPUT_FIELDS:
  - latency_mean_us
  - latency_p50_us
  - latency_p99_us
  - ram_used_bytes
  - throughput_ips
  - inference_count
  - baud_rate
  - platform
  - firmware_version

# Fenêtre de moyennage pour le throughput
THROUGHPUT_WINDOW: 50

# UART
BAUD_RATE: 115200
UART_TIMEOUT_S: 2.0

# Répertoire de sauvegarde par défaut
DEFAULT_OUTPUT_DIR: "experiments/"

# Plateforme courante
PLATFORM: "nucleo_f439zi"
```

---

## Documentation des clés

### Seuils d'alerte

| Clé | Valeur | Justification |
|-----|--------|---------------|
| `LATENCY_ALERT_MS` | `10.0` | 10% du budget hardware de 100 ms (contrainte STM32N6). Une alerte à 10 ms laisse une marge ×10 avant la violation. |
| `RAM_ALERT_BYTES` | `52000` | 52 Ko = marge de sécurité de 12 Ko avant la contrainte absolue de 64 Ko. Le firmware Mahalanobis actuel utilise ~18 Ko. |
| `THROUGHPUT_MIN_IPS` | `10` | Minimum fonctionnel : 10 inférences/s = 100 ms/inférence = limite absolue. En dessous, le système ne peut pas tenir le budget temps réel. |

### Lien avec le Triple Gap

- `LATENCY_ALERT_MS` et `RAM_ALERT_BYTES` sont les sentinelles du **Gap 2** : tout dépassement doit être documenté dans le manuscrit avec mesure précise.
- Un `profiling.json` avec `gap2_compliant: true` signifie : `ram_peak_bytes < 64000` ET `latency_mean_ms < 100.0`.

### `PROFILING_VERSION`

Version du schéma de configuration. `profiling_reader.py` peut utiliser cette clé pour adapter son parsing lors de futures migrations.

### `THROUGHPUT_WINDOW`

Fenêtre de moyennage pour le calcul de throughput côté host (non utilisée côté firmware qui calcule la moyenne glissante depuis le boot). Réservée pour une future implémentation de fenêtre glissante dans `profiling_reader.py`.

### `PLATFORM`

Plateforme courante. Mise à jour manuellement lors du changement de board (NUCLEO-F439ZI → STM32N6 eval → Edge Spectrum). Propagée dans `profiling.json` pour la traçabilité des résultats.

---

## Lecture dans le code Python

```python
# profiling_reader.py — chargement avec fallback si fichier absent
def _load_profiling_config() -> dict:
    cfg_path = Path("configs/profiling_config.yaml")
    if cfg_path.exists():
        with open(cfg_path) as f:
            return yaml.safe_load(f)
    return {"LATENCY_ALERT_MS": 10.0, "RAM_ALERT_BYTES": 52000, "THROUGHPUT_MIN_IPS": 10}
```

---

## Évolutions prévues

| Horizon | Changement |
|---------|-----------|
| Phase 2b (STM32N6) | `PLATFORM: "stm32n6_eval"`, mise à jour `RAM_ALERT_BYTES` si NPU change l'empreinte |
| Phase 3 (Edge Spectrum) | Nouveau profil de seuils spécifique au matériel Edge Spectrum |
| INT8 (Gap 3) | Ajout d'un champ `INT8_LATENCY_TARGET_MS` pour comparer FP32 vs INT8 |

---

## Critères d'acceptation

- [ ] `profiling_reader.py` charge le fichier sans erreur depuis `configs/profiling_config.yaml`
- [ ] `alerts` vide pour un profiling en dry-run (lat~3µs, ram~200B, thr~333333 ips)
- [ ] Modifier `LATENCY_ALERT_MS: 0.001` → l'alerte latence apparaît en dry-run (valeur < 3µs = 0.003ms)
- [ ] `PROFILING_VERSION: 2` présent dans le fichier
