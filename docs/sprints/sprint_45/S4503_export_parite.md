# S4503 — Export des paramètres & parité board↔PC

| Champ | Valeur |
|-------|--------|
| **Sprint** | 45 |
| **Priorité** | 🔴 Critique — garantit que le board décide comme le PC (parité par construction) et pilote la mesure. |
| **Statut** | ✅ Implémenté — export + driver board + parité ; cellule de validation mesurée board réelle (**parité 1.000**). |
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

---

## Résolution (implémentée)

**Fichiers** : `export_weights_c.py::export_drift_methods_to_c` + arg `--drift-methods` (génère
`inc/drift_methods_params.h`, `DRIFT_METHODS_PARAMS_PROVIDED`, jamais édité à la main) ;
`scripts/run_sprint45_board.py` (driver train→export→build→flash→**stream chronologique**) ;
`scripts/board_pc_parity45.py` (parité par échantillon) ; `tests/test_sprint45_board.py`.

**Points de conception clés** :
- **Streaming chronologique** — `sensor_stream._stream_uart` **mélange** les échantillons par tâche
  (`np.random.choice`), ce qui détruirait l'ordre du drift. Le driver écrit sa **propre boucle
  ordonnée** réutilisant les primitives wire (`build_frame_v2`/`parse_response`) → `sensor_stream.py`
  **inchangé**.
- **Parité par construction** — la réplique PC dérive le signal du **modèle de référence board
  exporté** (précédent S38 `_pc_gate_replay`) : `pred_pc = argmax EWCMlpMulticlass(features)` (tête
  exportée) → `error = 1[pred_pc ≠ true]` → Page-Hinkley Python (mêmes seuils) → `verdict_pc`.
- **Dims minimales au build** — seul le chemin EWC est exécuté ⇒ `EWC_IN=k` (+`PROTO_MAX_N=k` si
  k>16, +`MAHA_DIM=k` pour PSI). **Ne pas** gonfler `HDC_N_FEATURES`/`TINYOL_IN` (la projection HDC
  `k·HDC_DIM` déborderait la SRAM à k=128).

**Cellule de validation mesurée — board réelle NUCLEO-F439ZI** (`page_hinkley × gas_sensor_drift`,
128 features, 13 910 échantillons, seed 42) :

| Métrique | Valeur |
|----------|--------|
| **parité verdict board↔PC** | **1.000** (0 mismatch / 13 910) |
| parité prédiction EWC board↔PC | **1.000** |
| latence DWT (P50 = P99) | **270 µs** ≪ 100 ms (**Gap 2 ✅**) |
| `.bss` | 166 352 B (k=128 → tête EWC 128→32→16→2) |
| erreurs CRC | **0** |
| verdicts board | NORMAL 13 907 · WARNING 0 · DRIFT 3 |
| détection (F1 vs 9 drifts, tol 200) | **0.0** (honnête) |

**Honnêteté** : la parité (objet du sprint : *le board décide-t-il comme le PC ?*) est **exacte** ;
la **qualité de détection F1=0.0** est un chiffre réel non maquillé — Page-Hinkley (λ=50 littérature)
sur le flux d'erreur d'une tête EWC entraînée seulement sur l'enrôlement ne déclenche presque pas
sur ce dataset. L'amélioration de la détection (calibration λ/δ, choix du modèle de faute) est hors
périmètre du **portage** (elle relèverait d'un tuning PC S44). La chaîne board est validée bout-en-bout.

**Runbook (reste de la grille, non flashé — board 1 cellule par choix utilisateur)** : chaque
`(détecteur, dataset)` se mesure par
`python scripts/run_sprint45_board.py --detector <d> --dataset <ds> --port /dev/ttyACM0` puis
`board_pc_parity45.py`. Tant que non flashé : `metric_value` reste `null`/« à mesurer » (aucun chiffre
inventé). `electricity` → N/A honnête (pas de vérité-terrain ponctuelle).
