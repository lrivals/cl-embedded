# S1808 — Tests Python : streaming dry-run, mock serial, format CSV/JSON output

| Champ | Valeur |
|-------|--------|
| **ID** | S1808 |
| **Sprint** | Sprint 18 — 25 mai – 1er juin 2026 |
| **Priorité** | 🟡 Secondaire |
| **Durée estimée** | 3h |
| **Dépendances** | S1802 (`sensor_stream.py`), S1803 (`board_dataset_builder.py`) |
| **Fichiers cibles** | `tests/test_sensor_stream.py`, `tests/test_dataset_builder.py` |
| **Statut** | ✅ Implémenté — 21 tests existants |

---

## Objectif

Valider le protocole UART v2 et le pipeline de collecte de données sans board via des tests pytest couvrant :
- Construction et intégrité des trames v2 (magic, version, task_id, CRC)
- Comportement du streaming dry-run (count, task distribution, accuracy loopback)
- Calcul des statistiques agrégées
- Génération du CSV et du `results.json` au format Phase 1

---

## Commande d'exécution

```bash
pytest tests/test_sensor_stream.py tests/test_dataset_builder.py -v
```

---

## `tests/test_sensor_stream.py` — 12 tests

### Fixtures

```python
@pytest.fixture
def simple_features() -> np.ndarray:
    """Features f32 de taille 5 pour les tests de trame."""
    return np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)

@pytest.fixture
def mock_xy() -> tuple[np.ndarray, np.ndarray]:
    """100 samples × 5 features, labels binaires (seed=42)."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 5)).astype(np.float32)
    y = rng.integers(0, 2, size=100).astype(np.int64)
    return X, y
```

La fixture `mock_xy` est réutilisable pour futurs tests de streaming (S1809, S1810).

### Tests protocole v2

| Test | Assertion |
|------|-----------|
| `test_frame_magic` | `struct.unpack("<H", frame, 0)[0] == 0xABCD` |
| `test_frame_version` | `frame[2] == 0x02` |
| `test_frame_task_id` | `frame[3] == task_id` pour tid ∈ {0, 1, 2, 3} |
| `test_frame_crc_valid` | `crc8(frame[:-1]) == frame[-1]` |
| `test_frame_crc_detects_corruption` | Flip d'un bit → CRC mismatch |
| `test_frame_length` | `len(frame) == 9 + N×4 + 2 + 1` |
| `test_frame_all_features_preserved` | Roundtrip f32 exact à 6 décimales |

### Tests dry-run

| Test | Assertion |
|------|-----------|
| `test_dry_run_returns_correct_count` | `len(results) == n_samples` |
| `test_dry_run_all_status_ok` | `r["status"] & STATUS_CRC_ERR == 0` pour tout r |
| `test_dry_run_task_ids_distributed` | Exactement 3 task_id distincts avec n_tasks=3 |
| `test_dry_run_perfect_accuracy` | Accuracy = 1.0 (dry-run loopback : pred = true) |

### Tests statistiques

| Test | Assertion |
|------|-----------|
| `test_compute_stats_fields` | Présence des 9 champs obligatoires dans le dict stats |
| `test_compute_stats_empty` | `stats["n_samples"] == 0` pour une liste vide |

---

## `tests/test_dataset_builder.py` — 9 tests

### Fixture principale

```python
@pytest.fixture
def mock_results() -> list[dict]:
    """20 résultats fictifs couvrant 2 tâches, ~90% accuracy."""
```

Les champs requis : `task_id, ts_ms, true, pred, confidence, latency_us, ram_bytes, throughput_ips, status`.

### Tests CSV

| Test | Assertion |
|------|-----------|
| `test_save_csv_creates_file` | Fichier `dataset.csv` existe après `_save_csv()` |
| `test_save_csv_correct_row_count` | `len(rows) == len(mock_results)` (20) |
| `test_save_csv_required_columns` | Colonnes `task_id, true, pred, latency_us, ram_bytes` présentes |
| `test_save_csv_empty` | Liste vide → fichier CSV non créé |

### Tests `results.json`

| Test | Assertion |
|------|-----------|
| `test_results_json_required_fields` | 6 champs Phase 1 présents (`acc_final`, `avg_forgetting`, `backward_transfer`, `ram_peak_bytes`, `inference_latency_ms`, `n_params`) |
| `test_results_json_acc_range` | `0.0 ≤ acc_final ≤ 1.0` |
| `test_results_json_ram_positive` | `ram_peak_bytes > 0` |
| `test_results_json_latency_ms` | `0 < inference_latency_ms < 100.0` (Gap 2) |
| `test_results_json_empty` | Liste vide → `results_json == {}` |

---

## Couverture et lacunes identifiées

### Couvert ✅

- Protocole v2 complet (magic, version, task_id, CRC, longueur, features)
- Dry-run sans dépendance pyserial
- Format CSV (colonnes, count, fichier absent si liste vide)
- Format results.json (champs Phase 1, types, plages)

### Non couvert (extensions possibles)

| Cas | Fichier à créer |
|-----|-----------------|
| Mock serial pour test mode `--port` (pyserial mocké) | `tests/test_sensor_stream.py` |
| Test `config_snapshot.yaml` (champs présents, protocol_version=2) | `tests/test_dataset_builder.py` |
| Test `profiling_reader.py` depuis CSV | `tests/test_profiling_reader.py` |
| Test seuil d'alerte profiling (RAM > ALERT → alerts non vide) | `tests/test_profiling_reader.py` |

---

## Intégration CI

Ces tests s'exécutent **sans board** (dry-run uniquement) et sont inclus dans le pipeline CI GitHub Actions :

```yaml
# .github/workflows/firmware.yml (section tests Python)
- name: Tests Python Sprint 18
  run: pytest tests/test_sensor_stream.py tests/test_dataset_builder.py -v
```

---

## Critères d'acceptation

- [ ] `pytest tests/test_sensor_stream.py tests/test_dataset_builder.py -v` → 21 tests PASSED
- [ ] Aucun test ne nécessite `pyserial` installé
- [ ] `test_frame_crc_detects_corruption` détecte un flip d'un seul bit
- [ ] `test_results_json_latency_ms` passe : 0 < lat < 100 ms (Gap 2 check)
