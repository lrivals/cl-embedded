# S4604 — Expériences TinyOL : 3-way × {Monitoring, Pronostia}

| Champ | Valeur |
|-------|--------|
| **Sprint** | 46 |
| **Priorité** | 🟠 Important — étend le message 3-way au second modèle à fake-quant. |
| **Statut** | 📝 Doc — spec complète ; implémentation à venir |
| **Durée estimée** | 3h |
| **Dépendances** | S4602 ✅ (harnais) · `src/models/tinyol/tinyol_int8.py` ✅ (`TinyOLAutoencoderInt8`, `OtOHeadInt8`) · loaders Monitoring/Pronostia ✅ |
| **Fichiers cibles** | `configs/quant_moment/tinyol_{monitoring,pronostia}.yaml`, `experiments/exp_S46_tinyol/` |
| **Références** | S2804 (tinyol_int8) · S4602 · S4603 (symétrie EWC) |

---

## Contexte

TinyOL possède, comme EWC, un chemin de fake-quant dans sa boucle (l'update online `OtOHeadInt8.update_int8`
applique des poids fake-quant, gradient FP32 straight-through) plus une calibration PTQ des activations
(`calibrate_int8`). L'axe **before/after/both** s'y applique donc, avec la nuance que TinyOL est un
autoencodeur + tête OtO : sa métrique naturelle est l'**erreur de reconstruction** (anomaly detection),
convertie en F1 via seuil quand un label est disponible.

## Spec

### 1. Mapping des moments sur TinyOL

| Moment | Chemin TinyOL |
|--------|---------------|
| **fp32** | `TinyOLAutoencoder` + `OtOHead` FP32 |
| **before** | `TinyOLAutoencoderInt8` (fake-quant poids + activations UINT8) entraîné/mis à jour avec quant-in-the-loop ; `OtOHeadInt8.update_int8` |
| **after** | autoencodeur FP32 entraîné → `calibrate_int8(X_calib)` → `forward_int8` (PTQ activations, poids fake-quant à l'inférence) |
| **both** | autoencodeur INT8 (before) → recalibration PTQ des activations sur l'ensemble représentatif → inférence entière |

> Nuance honnête : TinyOL n'a pas de noyau firmware entier per-canal comme EWC (`int8_c_emulation` cible
> la tête EWC 5→32→16→2). Pour TinyOL, `after`/`both` reposent sur la calibration UINT8 par-tenseur de
> `tinyol_int8.py` — c'est le format disponible. Ce point est signalé dans les résultats (pas de
> per-canal TinyOL).

### 2. Configs

`configs/quant_moment/tinyol_{monitoring,pronostia}.yaml` :

```yaml
model: tinyol
dataset: monitoring          # / pronostia
extends: tinyol_config
quant_moment: both
after_scheme: per_tensor_calib   # UINT8 activation calibration (TinyOL n'a pas de per-canal)
seed: 42
metric: recon_error          # + f1 si seuil label disponible
```

### 3. Exécution

```bash
for ds in monitoring pronostia; do
  python scripts/run_s46_quant_moment.py --model tinyol --dataset $ds \
    --moment all --after-scheme per_tensor_calib \
    --config configs/quant_moment/tinyol_$ds.yaml \
    --output experiments/exp_S46_tinyol/${ds}_all.json
done
```

## Format de sortie

`experiments/exp_S46_tinyol/{monitoring,pronostia}_all.json` — schéma S4602, `metric_name` =
`recon_error` (ou `f1`). Table de synthèse (valeurs `pending`) :

| Dataset | fp32 | before | after | both | RAM ratio | Note |
|---------|:---:|:---:|:---:|:---:|:---:|------|
| Monitoring | pending | pending | pending | pending | pending | UINT8 par-tenseur (pas de per-canal) |
| Pronostia | pending | pending | pending | pending | pending | idem |

## Contraintes

- Métrique cohérente entre les 4 moments d'un même dataset (recon_error → même seuil pour F1).
- Signaler l'absence de per-canal TinyOL (limite de format, pas de résultat manquant).
- `seed=42`, `config_snapshot` déposé.

## Vérification

```bash
ls experiments/exp_S46_tinyol/monitoring_all.json experiments/exp_S46_tinyol/pronostia_all.json
python -c "import json; d=json.load(open('experiments/exp_S46_tinyol/monitoring_all.json')); \
assert set(d['moments'])=={'fp32','before','after','both'}"
```

---

## Résolution (implémentée)

✅ **Implémenté et exécuté** (PC/émulé, seed 42). Configs
`configs/quant_moment/tinyol_{monitoring,pronostia}.yaml` (héritant de
`tinyol_int8_{monitoring,pronostia}.yaml` via `extends`) + `run_tinyol_moments` ajouté au
harnais S4602 (`scripts/run_s46_quant_moment.py`). **Décision utilisateur** : axe =
**erreur de reconstruction** (fidèle à la spec). Réutilise `TinyOLAdapter`
(`scripts/benchmark_int8_fp32.py`) : FP32 = `TinyOLAnomalyDetector`, INT8 =
`TinyOLAutoencoderInt8` (fake-quant poids INT8 + activations UINT8 par-tenseur).

**2 JSON** `experiments/exp_S46_tinyol/{monitoring,pronostia}_all.json` (+ `config_snapshot.yaml`).
Métrique = **AUROC sur l'erreur de reconstruction** (normal-vs-fault).

### Table de synthèse (AUROC recon-error, seed 42)

| Dataset | fp32 | before (QAT) | after (PTQ) | both | RAM ratio | Δboth vs fp32 | Gap 3 |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Monitoring (D2) | 0.8771 | 0.6923 | 0.6923 | 0.6923 | ×3.53 (1456→412 B) | −0.1848 | métrique ❌ / RAM ✅ |
| Pronostia (D4)  | 0.6654 | 0.6594 | 0.6594 | 0.6594 | ×3.83 (4276→1117 B) | −0.0061 | métrique ✅ / RAM ✅ |

### Constat honnête (collapse before ≈ after ≈ both)

Comme anticipé (`na_note` dans le JSON), TinyOL n'a **ni noyau entier per-canal ni vraie
boucle QAT sur l'autoencodeur** → les trois moments empruntent le **même** forward INT8
par-tenseur appliqué au **même** AE FP32 entraîné : `before` = `after` = `both`
numériquement. Ce **collapse est reporté tel quel** (aucune cellule artificielle). Le seul
axe réellement à fake-quant-en-boucle serait la **tête OtO** (`OtOHeadInt8.update_int8`),
non retenu comme métrique (l'erreur de reconstruction ne dépend que de l'autoencodeur).

- **Monitoring** : la fake-quant UINT8 par-tenseur **dégrade** l'AUROC recon (0.877 → 0.692,
  Gap 3 métrique ❌) — grande sensibilité de l'erreur de reconstruction à la quantification
  des activations sur ce dataset (per-canal absent = limite de format, pas un bug de portage).
- **Pronostia** : dégradation négligeable (Δ = −0.006, Gap 3 métrique ✅).
- **RAM ✅ sur les 2 datasets** (poids ÷≈3.5–3.8, INT8 + overhead scales/zero-points).

`before` = **borne haute** (fake-quant inférence) et `both` = **déploiement** sont documentés
dans les `note` du JSON ; ici ils coïncident faute de kernel entier TinyOL distinct.

### Vérification

```
$ python -c "import json;d=json.load(open('experiments/exp_S46_tinyol/monitoring_all.json'));\
  assert set(d['moments'])=={'fp32','before','after','both'}"   # OK
```
