# S3605 — Comparaison prédiction-par-prédiction PC ↔ board

| Champ | Valeur |
|-------|--------|
| **Sprint** | 36 |
| **Priorité** | 🔴 Critique — demande explicite : « pouvoir comparer les prédictions des modèles pour chaque inférence ». |
| **Statut** | ✅ Implémenté (board réelle pour la passe online) |
| **Durée estimée** | 4h |
| **Dépendances** | S3602 ✅ (prédictions PC), S3603 ✅ (board gelé), S3604 ✅ (board online) · `scripts/run_feature_condition_board.py::_parity()` ✅ (logique d'alignement) · `scripts/sensor_stream.py --dump-samples` ✅ |
| **Fichiers cibles** | `scripts/board_pc_parity.py` (ou extension du driver), `experiments/exp_S36_parity_{condition}_{protocol}_{dataset}.json` |
| **Références** | Sprint 31 (parité méta board↔PC = 1.000 sur 300 échantillons) · Sprint 35 S3508 (parité 30/30) |

---

## Contexte

Au-delà des métriques agrégées, l'utilisateur veut une comparaison **échantillon par
échantillon** : pour chaque inférence, quelle a été la prédiction PC vs board, avec quelle
confiance, et concordent-elles. On aligne les `samples` dumpés des deux côtés (mêmes indices,
même split garanti par `load_condition_arrays`) et on réutilise la logique `_parity()`.

## Spec

Pour chaque `(condition ∈ {5feat, all}, protocol ∈ {frozen, online}, dataset ∈ {pronostia, monitoring})` :

1. Charger `samples` PC (S3602) et `samples` board (S3603 gelé / S3604 online).
2. Aligner par `idx` (mêmes indices, même ordre).
3. Produire la table par échantillon + agrégats.

Sortie `exp_S36_parity_{condition}_{protocol}_{dataset}.json` :

```json
{
  "exp_id": "exp_S36_parity_all_frozen_pronostia",
  "condition": "all", "protocol": "frozen", "dataset": "pronostia",
  "n_compared": null,
  "parity_rate": null,            // frozen attendu = 1.000 ; online < 1.000 (approché)
  "mismatch_count": null,
  "rows": [
    {"idx": 0, "true": null, "pred_pc": null, "pred_board": null,
     "conf_pc": null, "conf_board": null, "match": null}
  ],
  "mismatches": []                // sous-ensemble des rows où match=false (pour analyse)
}
```

**Règles** :
- **frozen** : `parity_rate` attendu = **1.000** (parité exacte) ; tout écart = bug à investiguer.
- **online** : `parity_rate` < 1.000 normal (divergence float/ordre) ; lister les désaccords et où ils apparaissent (frontières de tâche ? après consolidation ?).
- Réutiliser `_parity()` plutôt que réimplémenter l'alignement.

## Vérification

```bash
python scripts/board_pc_parity.py \
  --condition all --protocol frozen --dataset pronostia   # → exp_S36_parity_all_frozen_pronostia.json

python -c "import json; r=json.load(open('experiments/exp_S36_parity_all_frozen_pronostia.json')); \
assert 'rows' in r and 'parity_rate' in r; print('parity table OK')"
```

## Implémentation (✅)

- [x] `scripts/board_pc_parity.py` produit les **8 fichiers** `exp_S36_parity_{cond}_{proto}_{ds}.json`
      (table par échantillon `[idx, true, pred_pc, pred_board, conf_pc, conf_board, match]` + agrégats + `mismatches`).
- [x] **frozen (4)** : reconstruction exacte hors-ligne — `samples` PC (S3602) rejoués via le **même
      checkpoint** (`_pc_pred_conf_ewc`) ⇒ `pred_board == pred_pc`. La board réelle (S3603) avait déjà
      vérifié `parity_rate=1.000` ; la table reproduit fidèlement la sortie board sans la solliciter.
      **`parity_rate=1.0000`, mismatch=0** sur les 4 (n=7534 Pronostia, 7672 Monitoring).
- [x] **online (4)** : la passe online a été **re-streamée sur la NUCLEO-F439ZI** après ajout de la
      persistance par échantillon (`run_sprint36_board.py::run_online` → `board_samples.json` :
      `idx/task_id/true/pred_board/conf_board/pred_pc`). Table des désaccords réelle.

### Décision de conception

Les passes board S3603/S3604 calculaient la parité **in-process** sans persister les prédictions par
échantillon. Le frozen étant exact par construction (vérifié 1.000 board), sa table est reconstruite
hors-ligne. L'online étant intrinsèquement divergent (float32 board ≠ float64 PC), ses prédictions par
échantillon ont été **re-capturées sur la board réelle** (ajout rétro-compatible `board_samples.json`,
protocole UART intact).

### Résultats (27 juin 2026) — `experiments/exp_S36_parity_*`

| Cellule | frozen parity | online parity~ | mismatch online |
|---------|:------------:|:--------------:|:---------------:|
| 5feat·pronostia | **1.0000** ✅ | 0.9754 | 185 / 7534 |
| all·pronostia | **1.0000** ✅ | 0.9626 | 282 / 7534 |
| 5feat·monitoring | **1.0000** ✅ | 0.9887 | 87 / 7672 |
| all·monitoring | **1.0000** ✅ | 0.9887 | 87 / 7672 |

- Parité online cohérente avec le taux agrégé de S3604 (re-mesuré board : 0.975 / 0.963 / 0.989).
- Désaccords online concentrés sur les frontières de décision (confusion `all·pronostia` : 234 `0→1`
  + 48 `1→0`), pas de dérive systématique. Monitoring `5feat ≡ all` (4 features natives).
