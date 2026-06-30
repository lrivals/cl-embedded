# S3604 — Runs board, passe ONLINE (latence inférence + MAJ CL)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 36 |
| **Priorité** | 🔴 Critique — seule cette passe mesure la **latence inférence + mise à jour CL** (l'enjeu Gap 2 réel pour l'apprentissage en ligne). |
| **Statut** | ✅ Implémenté (board réelle) |
| **Durée estimée** | 5h |
| **Dépendances** | S3603 ✅ (même build) · `scripts/sensor_stream.py` ✅ (`--update`, `--consolidate-on-task-change`, `--proto 3`) · firmware EWC update + `ewc_consolidate()` + `PROTO_FLAG_UPDATE`/`PROTO_FLAG_CONSOLIDATE` (`firmware/stm32f4_blink/inc/pipeline.h`) ✅ · `src/evaluation/online_metrics.py` ✅ (acc/AUROC/forgetting online) · `src/models/ewc/ewc_mlp.py` ✅ (séquence online PC miroir) |
| **Fichiers cibles** | `experiments/exp_S36_board_online_{condition}_ewc_{dataset}/` (réutilise `sensor_stream.py`) |
| **Références** | Sprint 26 (latences séparées 130 µs inférence / 403 µs inférence+update) · point de vigilance parité ↔ MAJ (S3600) |

---

## Contexte

Avec `--update`, le firmware exécute **forward + backprop EWC** par échantillon, et
`ewc_consolidate()` aux frontières de tâche (`--consolidate-on-task-change`). La latence
mesurée par DWT couvre alors **inférence + MAJ CL** — à comparer à la latence inférence
seule (S3603), exactement comme Sprint 26 (130 µs vs 403 µs).

**Divergence assumée** : poids non gelés ⇒ PC (float64, PyTorch) et board (float32, C)
divergent au fil des updates. La parité online est donc **approchée** (taux de concordance,
pas égalité). Pour la rendre interprétable, PC exécute la **même séquence online** (mêmes
échantillons, même ordre, même seed) — cf. `TODO(arnaud)` déterminisme.

## Spec

Par `(condition, dataset)`, **même build qu'en S3603** :

1. **Board online** : `sensor_stream.py --condition ... --proto 3 --update --consolidate-on-task-change --dump-samples` (split test complet).
   - Latence DWT = **inférence + MAJ CL** → `latency_us_p50/p99`.
   - Métriques online v3 : `acc`, `auroc`, `forgetting` (firmware) + F1 dérivé hôte.
2. **PC online miroir** : EWC mis à jour échantillon par échantillon dans le **même ordre**, métriques via `online_metrics.py`.
3. **Parité approchée** : taux de concordance `pred_board` vs `pred_pc` online (documenté comme approché, pas exact).

Sortie `exp_S36_board_online_{condition}_ewc_{dataset}/results.json` :

```json
{
  "exp_id": "exp_S36_board_online_all_ewc_pronostia",
  "platform": "nucleo_f439zi", "model": "ewc", "dataset": "pronostia",
  "condition": "all", "stream_mode": "online (--update + consolidate)",
  "latency_us_p50": null, "latency_us_p99": null,
  "latency_inference_only_us_p50": null,   // repris de S3603 pour le delta inférence vs inf+MAJ
  "online_accuracy": null, "online_auroc": null, "online_forgetting": null,
  "f1_faulty": null, "f1_macro": null,
  "parity_class": "approx", "parity_rate": null, "n_compared": null,
  "gap2_latency_compliant": null
}
```

**Règles** :
- Latence ici = **inférence + MAJ CL** ; reporter aussi le delta vs S3603 (inférence seule).
- Parité `approx` (jamais annoncée comme exacte).
- Chiffres « à mesurer » tant que non exécuté.

## Vérification

```bash
python scripts/sensor_stream.py --port /dev/ttyACM0 \
  --dataset pronostia --condition all --proto 3 \
  --update --consolidate-on-task-change --dump-samples \
  --out experiments/exp_S36_board_online_all_ewc_pronostia/

python -c "import json; r=json.load(open('experiments/exp_S36_board_online_all_ewc_pronostia/results.json')); \
assert r['parity_class']=='approx' and 'online_forgetting' in r; print('board online OK')"
```

## Implémentation (✅)

- [x] Board online streamée (4 cellules) via `scripts/run_sprint36_board.py --pass online`
      (`_stream_cl_sequence`, `request_update=True`, `consolidate=True`, séquence 3 tâches contiguës
      couvrant le split complet).
- [x] Séquence online **PC miroir** (`_pc_online_mirror`) : modèle initialisé depuis le checkpoint
      flashé, rejoue la même séquence (prédire → 1 pas SGD single-sample → `consolidate` aux frontières).
- [x] Delta latence **inférence seule (S3603)** vs **inférence+MAJ (S3604)** calculé et stocké.
- [x] Parité online **approchée** documentée + Gap 2 confirmé (inf+MAJ ≪ 100 ms).

### Correctif outil (additif, rétro-compatible)

`_stream_cl_sequence` n'acceptait pas `model_flags` → la passe online tournait **sans
`FRAME_FLAGS_EWC_MODE`** (mode firmware par défaut ≠ EWC) : latence aberrante 17 µs < inf-seule.
Ajout d'un paramètre `model_flags: int = 0` à `_stream_cl_sequence` (OR dans les flags de trame) +
branchement dans `main()` (chemin `--cl-sequence`). Défaut `0` ⇒ comportement inchangé pour les
appels existants (protocole UART intact, `sensor_stream.py` reste la source unique).

### Résultats board réelle NUCLEO-F439ZI (12 juin 2026) — `experiments/exp_S36_board_online_*/`

| Cellule | k | lat inf+MAJ P50 | (inf seule S3603) | Δ MAJ | parité~ | F1_faulty | forgetting PC | Gap 2 |
|---------|---|-----------------|-------------------|-------|---------|-----------|---------------|-------|
| 5feat·monitoring | 4 | 239 µs | 48 µs | +191 µs | 0.989 | 0.902 | −0.003 | ✅ |
| all·monitoring | 4 | 239 µs | 48 µs | +191 µs | 0.989 | 0.902 | −0.003 | ✅ |
| 5feat·pronostia | 5 | 251 µs | 50 µs | +201 µs | 0.975 | 0.929 | −0.003 | ✅ |
| all·pronostia | 13 | 340 µs | 65 µs | +275 µs | 0.963 | 0.878 | +0.005 | ✅ |

- **Surcoût MAJ CL** +191…+275 µs (cohérent Sprint 26 : 130 µs inf vs 403 µs inf+MAJ). Latence totale
  inf+MAJ 239–340 µs **≪ 100 ms** ⇒ Gap 2 préservé pour l'apprentissage en ligne.
- **Parité approchée** 0.963–0.989 (jamais annoncée exacte) : poids non gelés ⇒ PC (float64) et board
  (float32) divergent au fil des updates. Le miroir PC rejoue la même séquence/ordre/seed pour rendre
  le taux interprétable.
- Métriques online firmware (proto v3 `acc/auroc/forgetting`) également consignées par cellule
  (`online_*_firmware`). Forgetting ≈ 0 (init depuis checkpoint convergé + 3 tâches courtes).
