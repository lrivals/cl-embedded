# S2106–S2109 — Expériences E21-03 et E21-04 : Pronostia sur board + comparaison

| Champ | Valeur |
|-------|--------|
| **Sprint** | 21 |
| **Priorité** | S2106–S2107 🔴 / S2108–S2109 🟡 |
| **Statut** | ✅ Terminé (board live 2026-05-28, protocole S2113 × 3 rép.) |
| **Durée estimée** | 2h + 3h + 1h + 2h |
| **Dépendances** | S2101 ✅ (feature_subset) · S2102 ✅ (`--dataset pronostia`) · S2103 ✅ (`board_pronostia.yaml`) |
| **Fichiers cibles** | `experiments/exp_S21_03/`, `experiments/exp_S21_04/`, `experiments/comparison_sprint21.json` |
| **Gap** | FIXME(gap1) — première validation Pronostia sur board |

---

## Contexte

Pronostia est le dataset de roulements industriels (FEMTO IEEE PHM 2012).  
C'est le seul dataset du projet avec **profils de dégradation réels** (RUL progressif).  
Il a été validé en PC Phase 1 (Sprints 11–15), mais jamais sur board.

**Sprint 21 = première validation Pronostia sur board** → contribue directement à Gap 1.

---

## E21-03 — Mahalanobis sur Pronostia (3 conditions)

### Commande

```bash
# Dry-run
python scripts/board_experiment_recorder.py \
    --config configs/board_pronostia.yaml \
    --model mahalanobis \
    --exp-id mahalanobis_pronostia \
    --dry-run --output experiments/exp_S21_03

# Séquence CL complète dry-run via sensor_stream directement
python scripts/sensor_stream.py --dataset pronostia \
    --cl-sequence cond1:200,cond2:200,cond3:200 \
    --update --protocol-version 3 --dry-run \
    --output experiments/exp_S21_03

# Board live
python scripts/sensor_stream.py --dataset pronostia \
    --cl-sequence cond1:200,cond2:200,cond3:200 \
    --update --protocol-version 3 \
    --port /dev/ttyACM0 \
    --output experiments/exp_S21_03
```

### Ce qui est mesuré

Mahalanobis = détection d'anomalie one-class sur les 5 features sélectionnées.  
- **Inférence** : distance Mahalanobis vs seuil adaptatif (EMA alpha=0.05)
- **Mise à jour** : EMA de la moyenne et matrice de précision (FLAG_UPDATE actif)
- **CL** : le modèle doit s'adapter aux 3 conditions sans oublier la cond1

### Résultats board live (2026-05-28) — protocole S2113, 3 répétitions warm run

| Métrique | **Moyenne ± σ** | Gap 2 |
|----------|:---------------:|:-----:|
| `acc_final` | **0.094 ± 0.007** | — |
| `avg_forgetting` | **0.000 ± 0.000** | — |
| `ram_peak_bytes` | **200 ± 0** | ✅ << 64 Ko |
| `inference_latency_ms` | **0.004 ± 0.000** | ✅ << 100 ms |

> ⚠️ **acc = 0.094 — cold start** : même problème que E21-01. Le détecteur Mahalanobis démarre sans distribution de référence et classe tout comme anomalie. Sur Pronostia, la majorité des frames sont en régime de dégradation (faulty) → acc ≈ 1 - fraction_faulty ≈ 0.10. Voir FIXME(gap1) dans E21-01.

---

## E21-04 — EWC sur Pronostia (scénario CL cond1→cond2→cond3, λ=400)

### Commande

```bash
# Dry-run avec simulation CL réaliste (même logique que E19-02)
python scripts/board_experiment_recorder.py \
    --config configs/board_pronostia.yaml \
    --model ewc \
    --exp-id ewc_pronostia_l400 \
    --dry-run --update \
    --output experiments/exp_S21_04

# Board live
python scripts/board_experiment_recorder.py \
    --config configs/board_pronostia.yaml \
    --model ewc \
    --exp-id ewc_pronostia_l400 \
    --update --consolidate-on-task-change \
    --port /dev/ttyACM0 \
    --output experiments/exp_S21_04
```

### Ce qui est mesuré

EWC Online MLP (5→32→16→2) entraîné séquentiellement sur 3 conditions Pronostia.  
À chaque frontière de tâche : `ewc_consolidate()` (Fisher EMA + snapshot θ*).  
Comparaison prévue : E21-04 (λ=400, EWC actif) vs baseline λ=0 (catastrophic forgetting).

### Résultats board live (2026-05-28) — protocole S2113, 3 répétitions warm run

| Métrique | **EWC λ=400 (moy ± σ)** | **Baseline λ=0 (moy ± σ)** | Ref. E19-02 Monitoring |
|----------|:-----------------------:|:-------------------------:|:----------------------:|
| `acc_final` | **0.886 ± 0.023** | **0.852 ± 0.011** | 0.896 ± 0.003 |
| `avg_forgetting` | **0.146 ± 0.025** | **0.204 ± 0.017** | 0.010 ± 0.012 |
| `backward_transfer` | -0.146 ± 0.025 | -0.204 ± 0.017 | -0.010 |
| `ram_peak_bytes` | 9 728 ✅ | 9 728 ✅ | 9 728 ✅ |
| `inference_latency_ms` | **0.251 ± 0.000** ✅ | **0.250 ± 0.001** ✅ | 0.249 ± 0.001 ✅ |

> **Propriété EWC vérifiée sur board** : avg_forgetting(λ=400) = 0.146 < avg_forgetting(λ=0) = 0.204.
>
> Note : la latence mesurée sur board (0.25 ms) est ~22× plus rapide que la simulation dry-run (5.49 ms). La simulation utilisait un modèle Python CPU ; le firmware C sur Cortex-M4 à 180 MHz est nettement plus efficace.
>
> Note `ram_peak_bytes` : le config `board_pronostia.yaml` expose `RAM_MAHA_BYTES=200` en premier ; la valeur réelle pour EWC est 9 728 B (constante `_EWC_RAM_BYTES` dans le recorder). Corriger dans une prochaine itération de `board_pronostia.yaml`.

### Expérience baseline λ=0 (optionnel mais recommandé)

```bash
python scripts/board_experiment_recorder.py \
    --config configs/board_pronostia.yaml \
    --model ewc \
    --override lambda_ewc=0.0 \
    --exp-id ewc_pronostia_baseline \
    --dry-run --update \
    --output experiments/exp_S21_04_baseline
```

---

## S2108 — RAM profiling board_pronostia.yaml

### Commande

```bash
# Recompiler le firmware si nécessaire
make -C firmware/stm32f4_blink/ all

# RAM profiling (identique au budget Sprint 20 — N_FEATURES=5 inchangé)
python scripts/parse_map_file.py \
    --map firmware/stm32f4_blink/build/stm32f4_blink.map \
    --budget 65536

# Vérification rapide
python scripts/parse_map_file.py \
    --map firmware/stm32f4_blink/build/stm32f4_blink.map \
    --budget 65536 --output experiments/exp_S21_03/ram_profile.json
```

### Résultat attendu

Aucun changement firmware → RAM identique Sprint 20 : **~15.7 Ko** / 64 Ko.

| Composant | RAM .bss |
|-----------|:-------:|
| Mahalanobis | 200 B |
| EWC | 9 728 B |
| TinyOL | 5 800 B |
| Métriques + pipeline | ~400 B |
| **Total** | **~16.1 Ko** |

---

## S2109 — comparison_sprint21.json

Tableau comparatif **3 datasets × 3 modèles** pour le manuscrit.

### Format

```json
{
  "generated": "2026-05-27",
  "sprint": 21,
  "datasets": ["cwru", "monitoring", "pronostia"],
  "models": ["mahalanobis", "ewc", "tinyol"],
  "results": {
    "cwru": {
      "mahalanobis": {"exp": "exp_S18_01", "acc_final": 0.932, "avg_forgetting": 0.012, "latency_ms": 0.004, "ram_bytes": 200},
      "tinyol":      {"exp": "exp_S19_01", "acc_final": 0.887, "avg_forgetting": 0.031, "latency_ms": 0.004, "ram_bytes": 5800},
      "ewc":         {"exp": null, "note": "non testé CWRU board"}
    },
    "monitoring": {
      "ewc":         {"exp": "exp_S19_02", "acc_final": 0.897, "avg_forgetting": 0.009, "latency_ms": 0.004, "ram_bytes": 9728},
      "mahalanobis": {"exp": "exp_S21_01", "acc_final": null,  "note": "Sprint 21 pending"},
      "tinyol":      {"exp": "exp_S21_02", "acc_final": null,  "note": "Sprint 21 pending"}
    },
    "pronostia": {
      "mahalanobis": {"exp": "exp_S21_03", "acc_final": null,  "note": "Sprint 21 pending"},
      "ewc":         {"exp": "exp_S21_04", "acc_final": null,  "note": "Sprint 21 pending"},
      "tinyol":      {"exp": null,         "note": "non prévu Sprint 21"}
    }
  },
  "gap2_summary": {
    "ram_budget_bytes": 65536,
    "all_compliant": true,
    "ram_max_observed_bytes": 9728
  }
}
```

### Commande de génération

```bash
python scripts/compare_experiments.py \
    --exps experiments/exp_S21_01 experiments/exp_S21_02 \
           experiments/exp_S21_03 experiments/exp_S21_04 \
           experiments/exp_S19_01 experiments/exp_S19_02 \
           experiments/exp_S18_01 \
    --output experiments/comparison_sprint21.json
```

---

## Vérification finale sprint

```bash
# 1. Dry-run complet des 4 expériences
for exp in 01 02 03 04; do
    python -c "import json,pathlib; r=json.loads(pathlib.Path(f'experiments/exp_S21_${exp}/results.json').read_text()); \
        assert r.get('gap2_latency_compliant'), 'latency KO'; \
        print(f'E21-0${exp}: {r[\"model\"]}@{r[\"dataset\"]} acc={r[\"acc_final\"]} OK')"
done

# 2. Vérification comparison JSON
python -c "
import json
d = json.loads(open('experiments/comparison_sprint21.json').read())
assert set(d['datasets']) == {'cwru', 'monitoring', 'pronostia'}
print('comparison_sprint21.json OK —', len(d['results']), 'datasets')
"
```

---

## Questions ouvertes

- `FIXME(gap1)` : E21-04 EWC Pronostia constitue la première mesure CL formelle sur données industrielles de roulements **sur board**. Mentionner explicitement dans le chapitre Gap 1 du manuscrit.
- `TODO(arnaud)` : EWC Pronostia vs EWC Monitoring — comparer AF et acc_final pour évaluer la généralisation du modèle CL entre datasets.
- `TODO(fred)` : Pronostia conditions 1/2/3 correspondent-elles à des équipements différents dans le parc Edge Spectrum ? Si oui, E21-04 est directement exploitable pour P2-06.
