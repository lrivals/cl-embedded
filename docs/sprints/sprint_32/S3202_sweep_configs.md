# S3202 — Génération des configs de balayage `configs/sweep/`

| Champ | Valeur |
|-------|--------|
| **Sprint** | 32 |
| **Priorité** | 🔴 Critique |
| **Statut** | ✅ Implémenté |
| **Durée estimée** | 2h |
| **Dépendances** | S3201 (loaders paramétrés) |
| **Résultat** | `scripts/generate_threshold_sweep_configs.py` → **15 configs** `configs/sweep/{dataset}_thr{XX}.yaml`. Base Pronostia = `pronostia_config.yaml` (mode binaire, décision utilisateur) + injection `label_mode: rul_threshold`, **pas** `pronostia_rul_config.yaml` (régression, sans section `data:`). |
| **Fichiers cibles** | `scripts/generate_threshold_sweep_configs.py`, `configs/sweep/*.yaml` |
| **Références** | `configs/cmapss_config.yaml`, `configs/battery_config.yaml`, `configs/pronostia_rul_config.yaml` |

---

## Contexte

Produire 15 configs (5 seuils × 3 datasets) à partir des configs de base, en n'injectant **que** le champ seuil — règle CLAUDE.md : aucun hyperparamètre modifié dans le code source, tout passe par YAML.

### Mapping des seuils (fractions du RUL_CAP)

| Dataset | Champ injecté | Seuils |
|---------|---------------|--------|
| CMAPSS | `data.faulty_threshold` | 10, 20, 30, 40, 50 |
| Pronostia | `data.label_mode: rul_threshold` + `data.faulty_threshold` | 24, 48, 72, 96, 120 |
| Battery | `data.rul_failure_threshold` | 67, 133, 200, 267, 333 |

---

## Spec

```python
# scripts/generate_threshold_sweep_configs.py
SWEEPS = {
    "cmapss":    ("configs/cmapss_config.yaml",       "faulty_threshold",      [10, 20, 30, 40, 50]),
    "pronostia": ("configs/pronostia_rul_config.yaml","faulty_threshold",      [24, 48, 72, 96, 120]),
    "battery":   ("configs/battery_config.yaml",      "rul_failure_threshold", [67, 133, 200, 267, 333]),
}
# pour chaque (dataset, seuil) : charger base, set data[champ]=seuil
# (+ data.label_mode="rul_threshold" pour pronostia), écrire
# configs/sweep/{dataset}_thr{seuil}.yaml — seul le champ seuil diffère.
```

---

## Vérification

```bash
python scripts/generate_threshold_sweep_configs.py
ls configs/sweep/*.yaml | wc -l   # attendu : 15
# diff doit ne porter que sur le champ seuil :
diff <(yq 'del(.data.faulty_threshold)' configs/sweep/cmapss_thr10.yaml) \
     <(yq 'del(.data.faulty_threshold)' configs/sweep/cmapss_thr50.yaml)   # vide
```
