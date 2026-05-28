# S2005 — Expérience E19-02 : EWC head, 3 tâches Monitoring, λ sweep

| Champ | Valeur |
|-------|--------|
| **Sprint** | 20 |
| **Priorité** | 🔴 Critique |
| **Statut** | ✅ Terminé |
| **Durée estimée** | 5h |
| **Dépendances** | S2001 (ewc_consolidate), S2002 (protocol v3) |
| **Fichiers cibles** | `experiments/exp_S19_02/` |
| **Référence** | `docs/sprints/sprint_19/S1912_exp_ewc.md`, `configs/board_ewc.yaml` |

---

## Contexte

L'expérience E19-02 a été définie en Sprint 19 avec un dry-run framework prêt.
Il faut maintenant l'exécuter réellement une fois `ewc_consolidate()` et le protocole v3 disponibles.

**Hypothèse à valider** : EWC avec λ=400 réduit significativement le catastrophic forgetting vs λ=0 (baseline).

---

## Configuration

| Paramètre | Valeur |
|-----------|--------|
| Modèle | EWC head MLP (5→32→16→2) |
| Dataset | Industrial Equipment Monitoring (3 types : pump → turbine → compressor) |
| Tâches | 3 (domain-incremental) |
| Samples / tâche | ~167 (total 500) |
| Conditions | λ=400 (EWC actif) vs λ=0 (catastrophic forgetting baseline) |
| Platform | NUCLEO-F439ZI @ 180 MHz (ou dry-run) |
| Fisher decay α | 0.9 (`configs/board_ewc.yaml:fisher_decay`) |
| SGD lr | 0.01 (`configs/board_ewc.yaml:learning_rate`) |

---

## Métriques attendues

| Condition | acc_final | avg_forgetting | inference_latency_ms | ram_peak_bytes |
|-----------|:---------:|:--------------:|:--------------------:|:--------------:|
| λ=400 (EWC) | ≥ 0.75 | **≤ 0.10** | ≤ 100 ms | ~9.7 Ko |
| λ=0 (baseline) | ≥ 0.60 | **≥ 0.25** | ≤ 100 ms | ~9.7 Ko |

> La latence back-prop Cortex-M4 FP32 est estimée 10–50 ms — à mesurer précisément via DWT.

---

## Commandes

```bash
# λ=400 (EWC actif)
python scripts/board_experiment_recorder.py \
    --config configs/board_ewc.yaml \
    --exp-id S19_02_ewc \
    --dry-run   # retirer pour board réel

# λ=0 (catastrophic forgetting)
python scripts/board_experiment_recorder.py \
    --config configs/board_ewc.yaml \
    --override lambda=0.0 \
    --exp-id S19_02_baseline \
    --dry-run
```

---

## Livrables

- `experiments/exp_S19_02/results_ewc.json` — λ=400, 6 métriques obligatoires
- `experiments/exp_S19_02/results_baseline.json` — λ=0
- `experiments/exp_S19_02/config_snapshot.yaml` — copie board_ewc.yaml au moment de l'exp

---

## Vérification

- [ ] `avg_forgetting(λ=400) < avg_forgetting(λ=0)` — propriété EWC de base
- [ ] `inference_latency_ms ≤ 100` — critère Gap 2
- [ ] `ram_peak_bytes ≤ 64*1024` — critère Gap 2
- [ ] `results.json` contient les 6 métriques obligatoires CLAUDE.md
- [ ] Comparer avec résultats Phase 1 Python (`experiments/exp_S12_XX/`) pour `backward_transfer`

---

## Questions ouvertes

- `TODO(arnaud)` : Inclure les courbes d'accuracy par tâche (acc_task_0 pendant tâche 1, 2) ou acc globale suffit ?
- `TODO(arnaud)` : Valeur de référence Phase 1 Python pour `avg_forgetting` EWC sur Monitoring ?
