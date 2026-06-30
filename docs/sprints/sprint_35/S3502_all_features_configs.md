# S3502 — Configs condition `all` (dims natives)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 35 |
| **Priorité** | 🟡 Important — alimente la condition `all` (S3503, S3508) |
| **Statut** | ✅ Implémenté |
| **Durée estimée** | 2h |
| **Dépendances** | loaders `src/data/` ✅ (dims natives), `configs/*_feature_subset.yaml` (format de référence) |
| **Fichiers cibles** | `configs/all_features/{dataset}.yaml` |

---

## Contexte

La condition `all` utilise **toutes les features natives** de chaque dataset, contrairement aux
subsets top-5. Il faut une config explicite par dataset (jamais de dim en dur) listant la dimension
native et l'ordre des features, alignée sur ce que produisent les loaders `src/data/`.

## Spec

```yaml
# configs/all_features/{dataset}.yaml — condition `all` (dims natives)
dataset: cwru
condition: all
n_features: 9
feature_names: [max, min, mean, sd, rms, skewness, kurtosis, crest, form]
source_loader: src/data/cwru_dataset.py
```

Dimensions natives à confirmer depuis les loaders (à vérifier en S3502, ne pas supposer) :

| Dataset | Loader | n_features natif (à confirmer) |
|---------|--------|:------------------------------:|
| cwru | `cwru_dataset.py` | 9 |
| monitoring | `monitoring` loader | 4 |
| pronostia | `pronostia_dataset.py` | 13 |
| cmapss | `cmapss_loader.py` | 21 sensors (⚠️ > `PROTO_MAX_N=16`) |
| paderborn | `paderborn_loader.py` | à confirmer |

**Note board** : CMAPSS `all`=21 > `PROTO_MAX_N=16` → dépend de la décision S3506
(relever `PROTO_MAX_N` ou restreindre `all` côté board). Côté PC, aucune contrainte.

## Vérification

```bash
ls configs/all_features/   # 5 fichiers
python -c "import yaml; d=yaml.safe_load(open('configs/all_features/cwru.yaml')); assert d['n_features']==len(d['feature_names'])"
```
