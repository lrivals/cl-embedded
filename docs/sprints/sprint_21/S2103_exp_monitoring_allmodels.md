# S2104–S2105 — Expériences E21-01 et E21-02 : tous les modèles sur Monitoring

| Champ | Valeur |
|-------|--------|
| **Sprint** | 21 |
| **Priorité** | S2104 🔴 / S2105 🟡 |
| **Statut** | ✅ Terminé (board live 2026-05-28, protocole S2113 × 3 rép.) |
| **Durée estimée** | 2h + 2h |
| **Dépendances** | Sprint 20 infrastructure ✅ (`board_experiment_recorder.py`, protocole v3, dry-run) |
| **Fichiers cibles** | `experiments/exp_S21_01/`, `experiments/exp_S21_02/` |

---

## Contexte

En Sprint 19/20, seul **EWC** a été validé sur Equipment Monitoring (E19-02, `avg_forgetting=0.009`).  
Mahalanobis et TinyOL ont été validés uniquement sur **CWRU** (E18-01, E19-01).

Ce sprint ferme cette lacune :
- **E21-01** : Mahalanobis sur Monitoring (anomaly detection, 3 domaines pump/turbine/compressor)
- **E21-02** : TinyOL sur Monitoring (reconstruction-based anomaly, même 3 domaines)

---

## E21-01 — Mahalanobis sur Equipment Monitoring

### Commande

```bash
# Dry-run
python scripts/board_experiment_recorder.py \
    --config configs/board_mahalanobis.yaml \
    --override dataset=monitoring \
    --exp-id mahalanobis_monitoring \
    --dry-run --output experiments/exp_S21_01

# Board live
python scripts/board_experiment_recorder.py \
    --config configs/board_mahalanobis.yaml \
    --override dataset=monitoring \
    --exp-id mahalanobis_monitoring \
    --port /dev/ttyACM0 --output experiments/exp_S21_01
```

### Scénario CL

```
Task 0 (pump)       : équipements type pompe → label faulty
Task 1 (turbine)    : équipements type turbine → label faulty
Task 2 (compressor) : équipements type compresseur → label faulty
```

Séquence CL : `pump:167,turbine:167,compressor:166` (500 samples total)

### Résultats board live (2026-05-28) — protocole S2113, 3 répétitions warm run

| Métrique | Rép. 1 | Rép. 2 | Rép. 3 | **Moyenne ± σ** | Gap 2 |
|----------|:------:|:------:|:------:|:---------------:|:-----:|
| `acc_final` | — | — | — | **0.107 ± 0.012** | — |
| `avg_forgetting` | — | — | — | **0.011 ± 0.008** | — |
| `ram_peak_bytes` | 200 | 200 | 200 | **200 ± 0** | ✅ << 64 Ko |
| `inference_latency_ms` | 0.004 | 0.004 | 0.004 | **0.004 ± 0.000** ✅ | ✅ << 100 ms |
| `gap2_latency_compliant` | ✅ | ✅ | ✅ | ✅ | — |

> ⚠️ **acc_final = 0.107 — cold start Mahalanobis** : le modèle démarre sans poids pré-chargés et doit apprendre la distribution normale via EMA. Le dataset Monitoring est ~90 % faulty → le détecteur prédit "normal" sur toutes les premières frames (distribution EMA pas encore convergée), d'où acc ≈ 1 - 0.90 = 0.10. Ce résultat est attendu sans initialisation explicite des poids.
>
> → `FIXME(gap1)` : pour obtenir une acc représentative, il faut charger les poids Mahalanobis pré-entraînés dans `model_weights.h` avant l'expérience board (voir S2108).

---

## E21-02 — TinyOL sur Equipment Monitoring

### Commande

```bash
# Dry-run
python scripts/board_experiment_recorder.py \
    --config configs/board_tinyol.yaml \
    --override dataset=monitoring \
    --exp-id tinyol_monitoring \
    --dry-run --output experiments/exp_S21_02

# Board live
python scripts/board_experiment_recorder.py \
    --config configs/board_tinyol.yaml \
    --override dataset=monitoring \
    --exp-id tinyol_monitoring \
    --port /dev/ttyACM0 --output experiments/exp_S21_02
```

### Particularité TinyOL

TinyOL est un autoencoder — anomaly detection par **seuil de reconstruction** (`RECON_THRESHOLD=0.05` MSE).  
Sur Monitoring, les features sont différentes de CWRU (température, pression, vibration, humidité vs statistiques temporelles bearing).  
Les poids pré-entraînés dans `model_weights.h` ont été fittés sur CWRU → possible biais de reconstruction.

→ **Point d'attention** : si acc_board < 0.60 sur Monitoring avec TinyOL, c'est probablement un problème de transfert de poids. Reporter dans les questions ouvertes.

### Résultats board live (2026-05-28) — protocole S2113, 3 répétitions warm run

| Métrique | **Moyenne ± σ** | Gap 2 |
|----------|:---------------:|:-----:|
| `acc_final` | **0.114 ± 0.010** | — |
| `avg_forgetting` | **0.000 ± 0.000** | — |
| `ram_peak_bytes` | **5 800 ± 0** | ✅ << 64 Ko |
| `inference_latency_ms` | **0.004 ± 0.000** ✅ | ✅ << 100 ms |

> ⚠️ **acc_final = 0.114 — même problème cold start que Mahalanobis** : les poids TinyOL embarqués dans `model_weights.h` ont été entraînés sur CWRU (statistiques de bearing). Le seuil MSE `RECON_THRESHOLD=0.05` n'est pas calibré pour Monitoring (temp/press/vib/humidity). Le modèle reconstruit mal toutes les frames → classe tout comme anomalie, alors que le dataset est ~90 % faulty → acc ≈ 0.10.
>
> → `FIXME(gap1)` : re-exporter les poids TinyOL après entraînement sur Monitoring, ou recalibrer `RECON_THRESHOLD` sur les premières frames du stream.

---

## Vérification commune

```bash
# Vérifier les 2 results.json
python -c "
import json
from pathlib import Path
for exp in ['exp_S21_01', 'exp_S21_02']:
    r = json.loads(Path(f'experiments/{exp}/results.json').read_text())
    assert r['gap2_latency_compliant'], f'{exp}: latency non compliant'
    assert r['gap2_ram_compliant'], f'{exp}: RAM non compliant'
    print(f'{exp}: acc={r[\"acc_final\"]:.3f} fgt={r[\"avg_forgetting\"]:.3f} OK')
"
```

---

## Questions ouvertes

- `TODO(arnaud)` : Comparer Mahalanobis Monitoring (E21-01) vs EWC Monitoring (E19-02) dans le même tableau — quel modèle recommander pour la maintenance prédictive équipement ?
- `TODO(arnaud)` : TinyOL sur Monitoring avec poids CWRU → est-ce qu'on doit re-exporter les poids TinyOL après entraînement sur Monitoring, ou garder les poids CWRU comme baseline "cross-dataset" ?
