# S3905 — Configs des schémas de quantification intermédiaires

| Champ | Valeur |
|-------|--------|
| **Sprint** | 39 |
| **Priorité** | 🔴 Critique — paramètres du sweep S3906 (jamais en dur dans le code) |
| **Statut** | ✅ Implémenté (1er juillet 2026) — 25 configs générées, `yaml.safe_load` OK |
| **Durée estimée** | 2h |
| **Dépendances** | S3904 (facteurs identifiés) · `configs/ewc_int8_*.yaml`, `configs/mahalanobis_q15_*.yaml` (patrons) |
| **Fichier cible** | `configs/quant_intermediate/*.yaml` |
| **Références** | `configs/mahalanobis_q15_pronostia.yaml` (clé `quantization:`) · `src/utils/int8_c_emulation.py` (`QuantConfig`) |

---

## Contexte

Conformément à la règle projet (« jamais d'hyperparamètre en dur, toujours via `configs/` »), chaque schéma
intermédiaire balayé par S3906 doit avoir un fichier de config. On réutilise la clé `quantization:` déjà
introduite par le Mahalanobis Q15 (Sprint 34) et on l'étend aux têtes neuronales.

## Schémas à configurer (× 5 datasets)

| Config | `quantization` | weight_scale | act_repr | acc | RAM |
|--------|:--------------:|:------------:|:--------:|:---:|:---:|
| `fp32_{ds}.yaml` | `fp32` | — | — | — | ×1 |
| `int8_legacy_{ds}.yaml` | `int8_legacy` | fixed_128 | q7_fixed | int16 | ×4 |
| `int8_perchannel_{ds}.yaml` | `int8_perchannel` | per_channel | q7_calib | int32 | ×4 |
| `q15_{ds}.yaml` | `q15` | per_channel | q15 | int32 | ×2 |
| `mixed_{ds}.yaml` | `mixed_int8w_q15act` | per_channel(int8) | q15 | int32 | ×4 poids |

## Gabarit de config

```yaml
# configs/quant_intermediate/q15_pronostia.yaml
model: ewc
dataset: pronostia
condition: 5feat
quantization: q15            # ← mappe vers QuantConfig.q15()
seed: 42
metric: f1_faulty
# Les schémas réutilisent les loaders/normalisations board existants.
```

> La correspondance `quantization:` → `QuantConfig` se fait dans `run_s39_quant_sweep.py` (S3906) via une
> table de mapping unique (pas de duplication). Les modèles non-neuronaux (Mahalanobis) réutilisent leurs
> configs `mahalanobis_q15_*` existantes ; HDC reste exact (INT8==FP32).

## Vérification

```bash
ls configs/quant_intermediate/                      # 5 schémas × 5 datasets
python -c "import yaml,glob; [yaml.safe_load(open(f)) for f in glob.glob('configs/quant_intermediate/*.yaml')]"
```
