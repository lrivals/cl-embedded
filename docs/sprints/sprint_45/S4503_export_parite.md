# S4503 — Export des paramètres & parité board↔PC

| Champ | Valeur |
|-------|--------|
| **Sprint** | 45 |
| **Priorité** | 🔴 Critique — garantit que le board décide comme le PC (parité par construction) et pilote la mesure. |
| **Statut** | 📝 Doc — spec ; implémentation à venir. |
| **Durée estimée** | 7h |
| **Dépendances** | S4502 ✅ (firmware) · `scripts/export_weights_c.py` ✅ (précédent `--drift-thresholds`, S3803) · `scripts/board_pc_parity38.py` ✅ (gabarit parité) · `scripts/sensor_stream.py` ✅ (streaming UART) |
| **Fichiers cibles** | `scripts/export_weights_c.py` (`--drift-methods`), `firmware/.../inc/drift_methods_params.h` (généré), `scripts/run_sprint45_board.py`, `scripts/board_pc_parity45.py`, `experiments/exp_S45_parity_*` |
| **Références** | S3802/S3806 (référence PC + parité Sprint 38) · CLAUDE.md § « ne jamais éditer les headers générés à la main » |

---

## Contexte

Pour que le board produise **le même verdict** que le Python, il doit recevoir **exactement** les mêmes
paramètres calibrés (seuils Page-Hinkley δ/λ, DDM, bornes/référence PSI). Cette tâche génère le header de
paramètres, orchestre le cycle board (train→export→build→flash→stream) et **mesure la parité** échantillon
par échantillon, sur le modèle éprouvé de Sprint 38.

## Spec

### 1. Export — `export_weights_c.py --drift-methods`

Lit les paramètres calibrés (issus de la référence PC S44 sur le segment d'enrôlement) et émet
`inc/drift_methods_params.h` : `PAGE_HINKLEY_DELTA`, `PAGE_HINKLEY_LAMBDA`, `DDM_WARN_SIGMA`,
`DDM_DRIFT_SIGMA`, `PSI_BIN_EDGES[]`, `PSI_REF_COUNTS[]`, `PSI_THRESHOLD`, `DRIFT_WINDOW_SIZE` + garde
`DRIFT_METHODS_PARAMS_PROVIDED`. **Jamais édité à la main** (règle CLAUDE.md). Header vide par défaut →
fallback neutre (seuils n'activant rien) → 0 régression.

### 2. Driver board — `scripts/run_sprint45_board.py`

Par cellule `(détecteur, dataset)` : calibre sur l'enrôlement (miroir exact PC) → `export_weights_c.py
--drift-methods` → `make EXTRA_CFLAGS="-DDRIFT_DETECT -DDRIFT_METHOD=<m>"` → flash → **stream** le split
test (`sensor_stream.py`, sans `--update`, `--proto 3`). Récupère verdict + latence DWT + `.bss` par
échantillon. `assemble_result` source unique ; N/A honnête (`metric_value=null` + `na_reason`) pour les
datasets sans ground-truth ponctuelle.

### 3. Parité — `scripts/board_pc_parity45.py`

Rejoue **la même séquence, même ordre, même seed** côté PC (détecteur S44) et compare au verdict board →
`experiments/exp_S45_parity_{detector}_{dataset}.json` : table par échantillon `[idx, score, verdict_pc,
verdict_board, match]` + `mismatches` + `verdict_parity` (fraction). Attendu : **parité exacte 1.000**
sur les détecteurs déterministes à paramètres identiques (comme `drift_detector.c` S3803).

## Contraintes

- Header **généré uniquement** ; le board consomme les mêmes paramètres que le PC → **parité par
  construction** (pas d'ajustement board indépendant).
- Streaming sans `--update` (on mesure la détection, pas l'adaptation du modèle de faute).
- `« à mesurer »` / `null` tant que non flashé (aucun chiffre inventé).
- `sensor_stream.py` **inchangé** au niveau wire (verdict via champ snapshot réinterprété, S4501/S4502).

## Vérification

```bash
python scripts/run_sprint45_board.py --detector page_hinkley --dataset gas_sensor_drift --port /dev/ttyACM0
python scripts/board_pc_parity45.py --detector page_hinkley --dataset gas_sensor_drift
```
- `inc/drift_methods_params.h` régénéré porte `DRIFT_METHODS_PARAMS_PROVIDED` + les valeurs calibrées.
- `exp_S45_parity_page_hinkley_gas_sensor_drift.json` : `verdict_parity == 1.000` (déterministe,
  paramètres identiques) ; mismatches = 0.
