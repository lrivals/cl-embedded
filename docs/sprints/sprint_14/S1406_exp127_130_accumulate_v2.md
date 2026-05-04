# S14-06 — exp_127–130 : accumulate v2 pour HDC / TinyOL AE / KMeans / Mahalanobis

| Champ | Valeur |
|-------|--------|
| **ID** | S14-06 |
| **Sprint** | Sprint 14 |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 2h |
| **Dépendances** | — (modèles déjà implémentés, exp_086–089 refit déjà faites) |
| **Fichiers cibles** | `experiments/exp_127/`, `experiments/exp_128/`, `experiments/exp_129/`, `experiments/exp_130/` |

---

## Objectif

Produire les variantes **accumulate** manquantes pour les 4 modèles déjà exécutés en refit (exp_086–089). Ces expériences complètent le tableau Monitoring by_equipment pour permettre la comparaison refit vs accumulate dans le notebook 6 modèles.

---

## Rappel résultats existants (refit)

| Exp | Modèle | AUROC moyen | RAM |
|-----|--------|-------------|-----|
| exp_086 | HDC | 0.945 | — |
| exp_087 | TinyOL AE | 0.972 | — |
| exp_088 | KMeans | 0.984 | — |
| exp_089 | Mahalanobis | 0.988 | — |

---

## Expériences à lancer

### exp_127 — HDC Monitoring by_equipment accumulate

```bash
python scripts/run_anomaly_detection.py \
    --model hdc \
    --dataset monitoring \
    --scenario by_equipment \
    --strategy accumulate \
    --config configs/hdc_config.yaml \
    --exp_id exp_127
```

### exp_128 — TinyOL AE Monitoring by_equipment accumulate

```bash
python scripts/run_anomaly_detection.py \
    --model tinyol_ae \
    --dataset monitoring \
    --scenario by_equipment \
    --strategy accumulate \
    --config configs/tinyol_config.yaml \
    --exp_id exp_128
```

### exp_129 — KMeans Monitoring by_equipment accumulate

```bash
python scripts/run_anomaly_detection.py \
    --model kmeans \
    --dataset monitoring \
    --scenario by_equipment \
    --strategy accumulate \
    --config configs/unsupervised_config.yaml \
    --exp_id exp_129
```

### exp_130 — Mahalanobis Monitoring by_equipment accumulate

```bash
python scripts/run_anomaly_detection.py \
    --model mahalanobis \
    --dataset monitoring \
    --scenario by_equipment \
    --strategy accumulate \
    --config configs/unsupervised_config.yaml \
    --exp_id exp_130
```

---

## Critères d'acceptation

- [ ] exp_127–130 : `metrics_anomaly.json` présents dans chaque répertoire
- [ ] `config_snapshot.yaml` présents
- [ ] Les AUROC accum sont comparables aux AUROC refit (pas d'effondrement)
- [ ] `avg_forgetting` ≤ valeur refit pour les modèles avec rétention naturelle (KMeans accum : distance centroïde joint)

## Statut

⬜ À faire
