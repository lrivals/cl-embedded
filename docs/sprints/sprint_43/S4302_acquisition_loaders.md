# S4302 — Acquisition & loaders des datasets de drift

| Champ | Valeur |
|-------|--------|
| **Sprint** | 43 |
| **Priorité** | 🔴 Critique — sans loaders exposant le ground-truth de drift, aucune métrique de délai de détection n'est calculable. |
| **Statut** | ✅ Implémenté — 4 loaders (`gas_sensor_drift`/`hydraulic`/`electricity`/`synthetic_drift`) + module commun `src/data/drift_dataset.py` (`DriftDataset`+`freeze_zscore`) + 4 configs + `scripts/download_drift_datasets.py` (idempotent) + registre `src/data/__init__.py` (`DRIFT_LOADERS`/`DRIFT_CONFIGS`). INSECTS→Hydraulic, `river`→numpy (décisions utilisateur). |
| **Durée estimée** | 7h |
| **Dépendances** | S4301 ✅ (sélection) · `src/data/pump_dataset.py` ✅ (style de loader + normalisation figée) · `src/data/__init__.py` ✅ (registre de loaders) |
| **Fichiers cibles** | `scripts/download_drift_datasets.py`, `src/data/gas_sensor_drift_dataset.py`, `src/data/insects_dataset.py`, `src/data/electricity_dataset.py`, `src/data/synthetic_drift_dataset.py`, `configs/gas_sensor_drift_config.yaml`, `configs/insects_drift_config.yaml`, `configs/electricity_drift_config.yaml`, `configs/synthetic_drift_config.yaml` |
| **Références** | `docs/context/datasets.md` · CLAUDE.md § « pas de données brutes committées », § « constantes nommées dans configs » |

---

## Contexte

Les détecteurs de drift (S44) et leur portage board (S45) consomment des flux d'échantillons **avec des
points de drift connus**. Cette tâche rend les datasets S4301 disponibles via des **loaders alignés sur
le style projet** (normalisation figée sur le segment initial, découpage CL, seed 42) et **expose la
ground-truth de drift** (`drift_points`, `drift_type`) de façon uniforme pour le harnais d'évaluation.

## Spec

### 1. Script de téléchargement — `scripts/download_drift_datasets.py`

CLI `--dataset {gas_sensor_drift,insects,electricity,synthetic,all}` → télécharge les archives dans
`data/raw/<dataset>/` (créé si absent, **jamais committé**, cf. `.gitignore`). Vérifie l'intégrité
(checksum si disponible), décompresse, journalise la source/licence. Idempotent (skip si présent).
Les synthétiques (`river`) ne se téléchargent pas : le script journalise qu'ils sont générés à la volée.

### 2. Loaders — `src/data/<dataset>_dataset.py`

Interface commune (miroir des loaders existants) :

- `load(config_path) -> DriftDataset` où `DriftDataset` expose :
  - `X` (`np.ndarray [N, d]`), `y` (labels de faute/classe si dual-usage, sinon `None`),
  - **`drift_points`** (`list[int]` — indices de changement de distribution, vérité-terrain),
  - **`drift_type`** (`Literal["sudden","gradual","incremental","recurring"]` par segment),
  - `feature_names`, `segments` (découpage CL/temporel), `metadata` (source, licence).
- **Normalisation figée** sur le **segment initial** (Z-score ou MinMax selon config), appliquée aux
  segments suivants **sans re-fit** — cohérent avec `pump_dataset.py` et essentiel pour que les
  détecteurs de drift **voient** la dérive non renormalisée (sinon la normalisation glissante masquerait
  le drift).
- Ordre **chronologique** préservé (le drift est temporel — aucun shuffle global).
- `seed=42` pour tout sous-échantillonnage.

Cas particuliers :
- **Gas Sensor Drift** : `drift_points` = frontières des 10 batches ; `y` = 6 gaz ; `drift_type` =
  `incremental`.
- **INSECTS** : une variante par fichier ; `drift_points` = points documentés de la variante ;
  `drift_type` ← variante.
- **Synthétique** (`synthetic_drift_dataset.py`) : génère un flux via `river` avec **points de drift
  exacts** paramétrés en config (vérité-terrain parfaite pour calibrer S44). `y` = label du générateur.

### 3. Configs — `configs/<dataset>_drift_config.yaml`

Toutes les tailles/paramètres en **constantes nommées** (CLAUDE.md) : chemin brut, méthode de
normalisation, colonnes de features, découpage de segments, `drift_points` attendus (ou paramètres du
générateur synthétique), `window`/`stride` si features glissantes, `seed`.

## Contraintes

- **Aucune donnée brute committée** ; `data/raw/<dataset>/` en `.gitignore` (vérifier l'entrée).
- Aucun hardcode de dimension/fenêtre dans le code — tout en config.
- Les loaders **n'inventent pas** de `drift_points` : soit ils viennent de la ground-truth du dataset,
  soit (synthétique) ils sont **imposés** au générateur et donc exacts. Pour un dataset sans ground-truth
  ponctuelle (Electricity/NOAA), `drift_points=None` et `drift_type` documenté — le harnais S44 le gère
  (métriques de délai non calculables, seules FAR/stabilité le sont).
- Enregistrer chaque loader dans `src/data/__init__.py` (registre existant).

## Vérification

```bash
python scripts/download_drift_datasets.py --dataset gas_sensor_drift   # → data/raw/gas_sensor_drift/
python -c "from src.data.gas_sensor_drift_dataset import load; d=load('configs/gas_sensor_drift_config.yaml'); print(d.X.shape, d.drift_points, d.drift_type)"
```
- `d.X.shape` cohérent avec la doc S4301 ; `d.drift_points` non vide (batches) ; normalisation figée
  vérifiable (moyenne du segment 0 ≈ 0 en Z-score, segments suivants ≠ 0 → drift visible).
- Synthétique : les `drift_points` retournés == ceux imposés en config (vérité-terrain exacte).
