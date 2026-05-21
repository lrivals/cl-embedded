# S1809 — Expérience E18-01 : stream 500 samples CWRU → CSV + profiling JSON auto

| Champ | Valeur |
|-------|--------|
| **ID** | S1809 |
| **Sprint** | Sprint 18 — 25 mai – 1er juin 2026 |
| **Priorité** | 🟡 Secondaire |
| **Durée estimée** | 3h |
| **Dépendances** | S1801–S1806 (pipeline complet opérationnel) |
| **Fichiers cibles** | `experiments/exp_S18_01/` |
| **Statut** | ⬜ À faire — critère de succès du sprint |

---

## Objectif

Valider le pipeline bout-en-bout en streamant 500 échantillons CWRU vers le firmware et en collectant automatiquement un dataset CSV + un profiling JSON.

C'est le **critère de succès principal du sprint** : produire `experiments/exp_S18_01/results.json` en moins de 5 minutes via dry-run.

---

## Contexte — Dataset CWRU

Le **Case Western Reserve University Bearing Dataset** est le Dataset 2 du projet, utilisé en scénario `by_fault_type` (3 tâches CL) :

| Tâche | Type de défaut | Sévérité |
|-------|---------------|----------|
| 0 | Ball fault | 0.007" |
| 1 | Inner race fault | 0.014" |
| 2 | Outer race fault | 0.021" |

Features utilisées (9 statistiques) : `max, min, mean, sd, rms, skewness, kurtosis, crest, form`.

---

## Commandes d'exécution

### Dry-run (sans board) — critère de succès sprint

```bash
python scripts/board_dataset_builder.py \
    --dataset cwru \
    --dry-run \
    --n-samples 500 \
    --n-tasks 3 \
    --output experiments/exp_S18_01
```

Durée attendue : < 30 secondes (dry-run, pas de UART).

### Génération du profiling JSON

```bash
python scripts/profiling_reader.py \
    --from-csv experiments/exp_S18_01/dataset.csv \
    --save experiments/exp_S18_01/profiling.json
```

### Avec board NUCLEO-F439ZI (quand disponible)

```bash
python scripts/board_dataset_builder.py \
    --dataset cwru \
    --port /dev/ttyACM0 \
    --n-samples 500 \
    --n-tasks 3 \
    --rate-hz 20 \
    --update \
    --platform nucleo_f439zi \
    --output experiments/exp_S18_01_board
```

---

## Fichiers produits : `experiments/exp_S18_01/`

```
exp_S18_01/
├── dataset.csv          ← 500 lignes, colonnes task_id/true/pred/latency_us/ram_bytes/...
├── results.json         ← métriques Phase 1 (acc_final, ram_peak_bytes, ...)
├── profiling.json       ← statistiques latence/RAM/throughput + gap2_compliant
└── config_snapshot.yaml ← config complète de l'expérience
```

---

## Valeurs attendues (dry-run)

| Métrique | Valeur dry-run | Valeur board (cible) |
|----------|---------------|----------------------|
| `acc_final` | 1.0 (loopback) | ~0.85–0.95 |
| `ram_peak_bytes` | 200 (simulé) | ~18 000–20 000 B |
| `inference_latency_ms` | ~0.003 ms (simulé) | ~0.4–2.0 ms |
| `gap2_compliant` | `true` | `true` (si < 64 Ko, < 100 ms) |
| `n_samples` | 500 | 500 |
| `n_tasks` | 3 | 3 |
| `crc_errors` | 0 | 0 (si liaison correcte) |

---

## Tableau résultats (à remplir après exécution)

| Mode | Date | `acc_final` | `ram_peak_bytes` | `latency_mean_ms` | `latency_p99_ms` | `throughput_mean_ips` | `gap2_compliant` | Durée session |
|------|------|------------|-----------------|------------------|-----------------|----------------------|-----------------|--------------|
| dry-run | — | — | — | — | — | — | — | — |
| NUCLEO-F439ZI | — | — | — | — | — | — | — | — |

---

## Vérification du résultat

```python
import json, pathlib

# Chargement résultats
r = json.loads(pathlib.Path("experiments/exp_S18_01/results.json").read_text())

# Critère de succès sprint
assert r["n_samples"] == 500
assert r["n_tasks"] == 3
assert r["acc_final"] is not None
assert r["ram_peak_bytes"] is not None
assert r["inference_latency_ms"] is not None

# Critère Gap 2 (board uniquement — dry-run valeurs simulées)
# assert r["ram_peak_bytes"] < 64000
# assert r["inference_latency_ms"] < 100.0

print("✅ Critères E18-01 validés")
print(f"  acc = {r['acc_final']:.3f}")
print(f"  RAM = {r['ram_peak_bytes']} B")
print(f"  lat = {r['inference_latency_ms']} ms")
```

---

## Critères d'acceptation

- [ ] `python scripts/board_dataset_builder.py --dataset cwru --dry-run --n-samples 500 --output experiments/exp_S18_01` complète en < 5 minutes
- [ ] `experiments/exp_S18_01/results.json` présent avec `n_samples=500` et `n_tasks=3`
- [ ] `experiments/exp_S18_01/dataset.csv` contient exactement 500 lignes
- [ ] `experiments/exp_S18_01/profiling.json` présent avec `gap2_compliant: true`
- [ ] `experiments/exp_S18_01/config_snapshot.yaml` présent avec `protocol_version: 2`

---

## Lien avec les expériences Phase 1

Cette expérience suit le format Phase 1 (`results.json` avec les 6 métriques CL obligatoires), permettant l'intégration dans `evaluate_all.py` pour une comparaison cross-dataset avec les expériences PC `exp_001` à `exp_148`.
