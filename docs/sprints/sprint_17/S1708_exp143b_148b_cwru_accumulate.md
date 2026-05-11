# S17-08 — exp_143b–148b : 6 modèles CWRU accumulate

| Champ | Valeur |
|-------|--------|
| **ID** | S17-08 |
| **Sprint** | Sprint 17 |
| **Priorité** | 🟢 Nice-to-have (si temps restant) |
| **Durée estimée** | 2h |
| **Dépendances** | S17-04 (exp_143–148 refit terminées) |
| **Fichiers cibles** | `experiments/exp_143b/` → `experiments/exp_148b/` |

---

## Objectif

Exécuter les 6 modèles d'anomaly detection sur CWRU en stratégie **accumulate** (le modèle est réentraîné à chaque tâche sur l'ensemble des données vues jusqu'ici). Permet de comparer refit vs accumulate sur CWRU et de compléter le tableau cross-stratégie du notebook S17-06.

Cette tâche est conditionnelle : ne démarrer que si S17-04, S17-05, S17-06, S17-07, et S17-09 sont terminées et qu'il reste du budget temps.

---

## Expériences

| Exp | Modèle | Stratégie | Config | Statut |
|-----|--------|-----------|--------|--------|
| exp_143b | HDC | accumulate | `configs/hdc_config.yaml` | ⬜ |
| exp_144b | TinyOL AE | accumulate | `configs/tinyol_config.yaml` | ⬜ |
| exp_145b | KMeans | accumulate | `configs/unsupervised_config.yaml` | ⬜ |
| exp_146b | Mahalanobis | accumulate | `configs/unsupervised_config.yaml` | ⬜ |
| exp_147b | DBSCAN | accumulate | `configs/unsupervised_config.yaml` | ⬜ |
| exp_148b | EWC one-class | accumulate | `configs/ewc_oneclass_config.yaml` | ⬜ |

---

## Commande (batch)

```bash
SCENARIO=$(grep SPLIT_STRATEGY configs/unsupervised_config.yaml | awk '{print $2}')

declare -A MODELS=(
    [exp_143b]=hdc
    [exp_144b]=tinyol_ae
    [exp_145b]=kmeans
    [exp_146b]=mahalanobis
    [exp_147b]=dbscan
    [exp_148b]=ewc_oneclass
)

for exp_id in "${!MODELS[@]}"; do
    model=${MODELS[$exp_id]}
    python scripts/run_anomaly_detection.py \
        --model $model \
        --dataset cwru \
        --scenario $SCENARIO \
        --strategy accumulate \
        --exp_id $exp_id
done
```

---

## Points d'attention

- **Accumulate sur CWRU** : le pool de données normales croît à chaque tâche (t1 : ~77, t2 : ~154, t3 : ~231). Contraste attendu avec refit où chaque tâche repart de zéro.
- **Mémoire** : vérifier `ram_peak_bytes` — la stratégie accumulate stocke toutes les données vues, ce qui peut dépasser le budget 64 Ko embarqué. Documenter si c'est le cas.
- **DBSCAN (exp_147b)** : le cluster grow avec l'accumulation — min_samples et eps peuvent nécessiter un ajustement différent de la version refit.
- **KMeans (exp_145b)** : avec N_CLUSTERS=1 et accumulate, le centroid se déplace vers la moyenne globale des normaux → stabilité améliorée attendue vs refit.

---

## Format `metrics_anomaly.json` (CWRU accumulate)

```json
{
  "model": "hdc",
  "dataset": "cwru",
  "scenario": "by_severity",
  "strategy": "accumulate",
  "normal_ratio": 0.10,
  "n_train_normal_per_task": [77, 154, 231],
  "auroc_per_task": [0.0, 0.0, 0.0],
  "auroc_mean": 0.0,
  "avg_forgetting": 0.0,
  "ram_peak_bytes": 0,
  "inference_latency_ms": 0.0
}
```

> `n_train_normal_per_task` est une liste croissante pour la stratégie accumulate (contrairement au scalaire fixe du refit).

---

## Critères d'acceptation

- [ ] exp_143b–148b : `metrics_anomaly.json` présents avec `auroc_per_task` de longueur 3
- [ ] `n_train_normal_per_task` est une liste de 3 valeurs croissantes dans chaque JSON
- [ ] `config_snapshot.yaml` présents dans les 6 répertoires
- [ ] `ram_peak_bytes` documenté — noter si accumulate dépasse le budget embarqué 64 Ko

## Statut

⬜ En attente S17-04 (et budget temps disponible)
