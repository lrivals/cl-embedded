# S4-03 — Expérience exp_004 : buffer UINT8 vs FP32 — delta précision

| Champ | Valeur |
|-------|--------|
| **ID** | S4-03 |
| **Sprint** | Sprint 4 — Semaine 4 (6–13 mai 2026) |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 2h |
| **Dépendances** | S4-02 (buffer UINT8 opérationnel) · exp_003 (baseline TinyOL FP32 disponible) |
| **Fichiers cibles** | `scripts/train_tinyol.py` (extension `--uint8` flag) · `experiments/exp_004_tinyol_uint8/` |
| **Statut** | ✅ Terminé |

---

## Objectif

Mesurer l'impact du buffer UINT8 sur la précision (delta AA vs FP32) et la RAM (bytes économisés) pour prouver que la quantification des activations est acceptable pour le déploiement MCU.

Cette expérience adresse directement **Gap 3** : quantification pendant l'entraînement incrémental, avec chiffres précis.

**Critère de succès** : `experiments/exp_004_tinyol_uint8/results/metrics.json` produit avec `delta_aa ≤ 0.005` (seuil acceptable pour justification manuscrit).

---

## Commande d'exécution

```bash
python scripts/train_tinyol.py \
    --config configs/tinyol_config.yaml \
    --exp_id exp_004_tinyol_uint8 \
    --uint8_buffer true \
    --compare_fp32 true   # exécute aussi le run FP32 pour comparaison directe
```

**Résultats persistés dans** `experiments/exp_004_tinyol_uint8/` :
```
experiments/exp_004_tinyol_uint8/
├── config_snapshot.yaml
└── results/
    ├── metrics.json           ← métriques CL FP32 + UINT8
    ├── memory_report.json     ← RAM FP32 vs UINT8 buffer
    └── acc_matrix_uint8.npy  ← matrice accuracy UINT8
```

---

## Format des résultats attendus

### `results/metrics.json`

```json
{
  "exp_id": "exp_004_tinyol_uint8",
  "dataset": "pump_maintenance",
  "model": "tinyol",
  "seed": 42,
  "config": {
    "use_uint8_buffer": true,
    "buffer_size": 50,
    "buffer_replay_ratio": 0.2
  },
  "fp32_baseline": {
    "aa": 0.5586,
    "af": 0.0084,
    "bwt": -0.0084
  },
  "uint8_buffer": {
    "aa": null,
    "af": null,
    "bwt": null
  },
  "delta_aa": null,
  "gap3_target_met": null
}
```

### `results/memory_report.json`

```json
{
  "fp32_buffer_bytes": 1800,
  "uint8_buffer_bytes": 450,
  "compression_ratio": 4.0,
  "total_ram_fp32_bytes": null,
  "total_ram_uint8_bytes": null,
  "within_budget_64ko": null,
  "within_budget_256ko": null
}
```

---

## Métriques à comparer

| Métrique | TinyOL FP32 (exp_003) | TinyOL UINT8 (exp_004) | Delta |
|---------|:---------------------:|:----------------------:|:-----:|
| AA | 0.5586 | ? | Δ AA |
| AF | 0.0084 | ? | Δ AF |
| BWT | -0.0084 | ? | Δ BWT |
| RAM buffer | 1 800 B | 450 B | −1 350 B (75%) |
| RAM totale update | 6 425 B | ? | ? |
| Latence update | ? | ? | ? |

**Seuil de succès Gap 3** : `|delta_aa| ≤ 0.005` (0.5 point de précision maximal sacrifié pour 4× de gain RAM).

---

## Extension du script `train_tinyol.py`

```python
# Ajout du flag --uint8_buffer et de la logique de comparaison
parser.add_argument("--uint8_buffer", type=str, default="false",
                    help="Active le buffer UINT8 (true/false)")
parser.add_argument("--compare_fp32", type=str, default="false",
                    help="Exécute aussi le run FP32 pour comparaison directe")

# Dans la boucle principale
if args.uint8_buffer.lower() == "true":
    config["oto_head"]["use_uint8_buffer"] = True

# Export du delta AA en fin d'expérience
if compare_fp32:
    metrics["delta_aa"] = metrics["uint8_buffer"]["aa"] - metrics["fp32_baseline"]["aa"]
    metrics["gap3_target_met"] = abs(metrics["delta_aa"]) <= 0.005
```

---

## Critères d'acceptation

- [ ] `experiments/exp_004_tinyol_uint8/results/metrics.json` existe avec toutes les clés
- [ ] `results/memory_report.json` contient `compression_ratio: 4.0`
- [ ] `delta_aa` calculé et stocké (signé — positif = UINT8 améliore légèrement)
- [ ] `gap3_target_met` est `true` ou `false` avec explication dans le rapport
- [ ] `within_budget_256ko` est calculé (budget NUCLEO-F439ZI, pas 64 Ko)
- [ ] `config_snapshot.yaml` contient les paramètres UINT8 utilisés
- [ ] Run reproductible avec seed=42

---

## Analyse attendue

Si `delta_aa > 0.005` (trop de dégradation) :
1. Augmenter `buffer_size` de 50 → 100
2. Passer `buffer_replay_ratio` de 0.2 → 0.5
3. Si insuffisant : quantification per-channel (n_bits=8, per_channel=True)
4. Escalade : `TODO(dorra)` pour avis expert

Si `delta_aa ≤ 0.005` :
- Documenter comme preuve Gap 3 dans le manuscrit
- Reporter dans `roadmap_phase1.md` : `FIXME(gap3)` → ✅ résolu partiellement

---

## Questions ouvertes

- `FIXME(gap3)` : Le seuil de 0.005 est arbitraire — valider avec Arnaud si une dégradation plus élevée est acceptable sur Dataset 1 (AA ≈ 0.56 de base, peu de marge).
- `TODO(arnaud)` : Pour le manuscrit, doit-on reporter le delta AA absolu ou relatif (delta_aa / aa_fp32) ?
