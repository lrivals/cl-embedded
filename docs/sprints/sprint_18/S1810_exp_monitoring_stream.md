# S1810 — Expérience E18-02 : stream Monitoring, compare latence Mahalanobis vs dummy

| Champ | Valeur |
|-------|--------|
| **ID** | S1810 |
| **Sprint** | Sprint 18 — 25 mai – 1er juin 2026 |
| **Priorité** | 🟢 Optionnel |
| **Durée estimée** | 3h |
| **Dépendances** | S1809 (E18-01 validé) |
| **Fichiers cibles** | `experiments/exp_S18_02/` |
| **Statut** | ⬜ À faire |

---

## Objectif

Valider le pipeline sur le **Dataset Monitoring** (Industrial Equipment Monitoring, 3 tâches par type d'équipement) et mesurer la différence de latence entre :
- **Mahalanobis** (firmware actuel) : inférence + mise à jour incrémentale
- **Dummy** (baseline) : réponse fixe sans calcul, pour calibrer la latence UART résiduelle

---

## Contexte — Dataset Monitoring

Le dataset Industrial Equipment Monitoring est le Dataset 2 du projet :

| Tâche | Équipement | Samples |
|-------|-----------|---------|
| 0 | Pump | ~2 534 |
| 1 | Turbine | ~2 565 |
| 2 | Compressor | ~2 573 |

Features (4 colonnes numériques) : `temperature, pressure, vibration, humidity`.

Label binaire : `faulty` (0 = normal, 1 = défaut).

---

## Commandes d'exécution

### Dry-run Monitoring (sans board)

```bash
python scripts/board_dataset_builder.py \
    --dataset monitoring \
    --dry-run \
    --n-samples 300 \
    --n-tasks 3 \
    --output experiments/exp_S18_02
```

### Profiling depuis CSV

```bash
python scripts/profiling_reader.py \
    --from-csv experiments/exp_S18_02/dataset.csv \
    --save experiments/exp_S18_02/profiling.json
```

### Avec board — Mahalanobis (firmware actuel)

```bash
python scripts/board_dataset_builder.py \
    --dataset monitoring \
    --port /dev/ttyACM0 \
    --n-samples 300 \
    --n-tasks 3 \
    --rate-hz 20 \
    --update \
    --platform nucleo_f439zi \
    --output experiments/exp_S18_02_mahal
```

### Avec board — Dummy baseline

Le dummy baseline est le firmware compilé avec `DUMMY_MODE` défini : retourne immédiatement `pred=0, conf=0.5` sans calcul. Permet de mesurer la latence UART incompressible.

```bash
# Compiler firmware en mode dummy (à implémenter dans pipeline.c)
# make -C firmware/stm32f4_blink DUMMY_MODE=1

python scripts/board_dataset_builder.py \
    --dataset monitoring \
    --port /dev/ttyACM0 \
    --n-samples 300 \
    --platform nucleo_f439zi \
    --output experiments/exp_S18_02_dummy
```

---

## Tableau comparaison Mahalanobis vs Dummy (à remplir)

| Mode firmware | `latency_mean_ms` | `latency_p99_ms` | `throughput_mean_ips` | `ram_peak_bytes` | `acc_final` |
|--------------|------------------|-----------------|----------------------|-----------------|------------|
| Dry-run | ~0.003 | ~0.003 | ~333333 | ~200 | 1.0 |
| Dummy (NUCLEO) | — | — | — | — | ~0.5 |
| Mahalanobis (NUCLEO) | — | — | — | — | — |

**Latence Mahalanobis** = latence mesurée − latence dummy → temps pur de l'inférence Mahalanobis.

---

## Analyse attendue

### Décomposition de la latence

```
Latence totale = Latence UART (transmission) + Latence inférence
                = (taille_trame / baud_rate) + temps_calcul_mahalanobis

Trame N=4 features : 9 + 4×4 + 3 = 28 B envoyés + 14 B reçus = 42 B
À 115200 baud : 42 × 10 bits / 115200 ≈ 3.65 ms (avec bits de stop)
→ Budget inférence restant = 100 ms - 3.65 ms ≈ 96 ms
```

### Comparaison CWRU vs Monitoring

| Dataset | N features | Taille trame | Latence UART théorique |
|---------|-----------|-------------|----------------------|
| CWRU | 9 | 9 + 36 + 3 = 48 B + 14 B | ~4.3 ms |
| Monitoring | 4 | 9 + 16 + 3 = 28 B + 14 B | ~3.6 ms |

---

## Fichiers produits : `experiments/exp_S18_02/`

```
exp_S18_02/
├── dataset.csv
├── results.json
├── profiling.json
└── config_snapshot.yaml
```

---

## Vérification du résultat

```python
import json, pathlib

r = json.loads(pathlib.Path("experiments/exp_S18_02/results.json").read_text())
p = json.loads(pathlib.Path("experiments/exp_S18_02/profiling.json").read_text())

assert r["dataset"] == "monitoring"
assert r["n_tasks"] == 3
assert p["gap2_compliant"] is not None

print(f"Monitoring — acc={r['acc_final']:.3f}")
print(f"Latence : mean={p['latency_mean_ms']} ms, P99={p['latency_p99_ms']} ms")
print(f"Gap 2 : {p['gap2_compliant']}")
```

---

## Critères d'acceptation

- [ ] `board_dataset_builder.py --dataset monitoring --dry-run --n-samples 300 --output experiments/exp_S18_02` complète sans erreur
- [ ] `profiling.json` présent avec `gap2_compliant` renseigné
- [ ] `results.json` : `dataset == "monitoring"` et `n_tasks == 3`
- [ ] (Board) latence Mahalanobis − dummy < 2 ms (inférence Mahalanobis pure rapide)

---

## Questions ouvertes

- `TODO(arnaud)` : Quelle fréquence d'échantillonnage cible pour le streaming continu (10 Hz / 100 Hz) ? Impacte directement `--rate-hz`.
- `TODO(fred)` : Le format de données capteurs réels Edge Spectrum est-il compatible avec les 4 features Monitoring, ou nécessite-t-il un mapping ?
