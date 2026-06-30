# S2214–S2216 — Paderborn : Expériences CL PC

| Champ | Valeur |
|-------|--------|
| **Sprint** | 22 |
| **Priorité** | 🔴 (S2214) / 🟡 (S2215, S2216) |
| **Statut** | ✅ Terminé (S2214/S2215 ✅ 2026-06-01 — S2217bis/S2218bis ✅ 2026-06-01 — biais structurel confirmé sur les 4 modèles) |
| **Durée estimée** | 1h + 30 min + 2h + 45 min + 45 min = 5h |
| **Dépendances** | S2210–S2213 ✅ (`paderborn_loader.py` + `paderborn_config.yaml` opérationnels) |
| **Fichiers cibles** | `experiments/exp_S22_05/`, `experiments/exp_S22_06/`, `experiments/exp_S22_07/`, `experiments/exp_S22_08/`, `notebooks/results_paderborn_cl.ipynb` |
| **Référence** | `scripts/train_ewc.py`, `evaluation/metrics.py`, `notebooks/results_cmapss_cl.ipynb` (pattern) |

---

## Contexte

Quatre expériences CL sur Paderborn (sain K001 → OR fault KA04 → IR fault KI04) couvrant les trois modèles du projet + la baseline Mahalanobis. EWC est le modèle de référence (S2214). Mahalanobis sert de baseline légère non-neuronale (S2215). TinyOL (S2217bis) et HDC (S2218bis) ont été ajoutés suite aux résultats insuffisants d'EWC (`acc_final = 0.667`) — la comparaison 4 modèles est nécessaire pour statuer sur le biais de classe structurel (2 tâches fault vs 1 tâche saine) et choisir l'approche à retenir pour le manuscrit (Gap 1).

---

## S2214 — exp_S22_05 : EWC / Paderborn

```bash
python scripts/train_ewc.py \
    --config configs/paderborn_config.yaml \
    --exp-id exp_S22_05 \
    --output experiments/exp_S22_05/
```

**Structure de sortie attendue** :
```
experiments/exp_S22_05/
├── config_snapshot.yaml
├── results.json
├── training_curves.png
└── ewc_weights_final.pt
```

**Métriques cibles** (critères manuscrit) :

| Métrique | Valeur cible | Justification |
|----------|:------------:|---------------|
| `acc_final` | ≥ 0.80 | Paderborn = 3 tâches seulement, signal plus propre que CMAPSS |
| `avg_forgetting` | ≤ 0.10 | Même justification + régularisation EWC efficace sur 2→3 tâches |

**Résultats obtenus (2026-06-01)** :

| Métrique | Valeur cible | Résultat | Statut |
| -------- | :----------: | :------: | :----: |
| `acc_final` | ≥ 0.80 | **0.6667** | ❌ |
| `avg_forgetting` | ≤ 0.10 | **0.5000** | ❌ |
| `ram_peak_bytes` | — | 1 171 B | ✅ << 64 Ko |
| `inference_latency_ms` | ≤ 100 ms | 0.034 ms | ✅ |

> ⚠️ **Effondrement binaire — biais de classe structurel** : le modèle apprend K001 (label=0, sain) parfaitement après T1 (acc=1.00), puis oublie totalement après T2/T3 (acc=0.00). La cause est un déséquilibre de classe 2:1 : 2 tâches "défaut" (label=1) pour 1 tâche "sain" (label=0). Le biais vers label=1 dépasse la régularisation EWC (λ=1000). **Le joint training donne également 0.6667** → problème de formulation du problème, pas d'hyperparamètre.
>
> `per_task_acc = [0.00, 1.00, 1.00]` — le modèle prédit tout comme défaut après T2.
>
> → `TODO(arnaud)` : Option A : reformuler en 3 classes (K001=0, KA04=1, KI04=2). Option B : utiliser AUROC comme métrique primaire. Option C : ajouter `pos_weight` dans BCELoss.

**Note** : la cible `acc_final ≥ 0.80` présupposait un signal plus propre, mais n'anticipait pas le déséquilibre structurel 2 tâches fault pour 1 tâche saine.

---

## S2217bis — exp_S22_07 : TinyOL / Paderborn

```bash
python scripts/train_tinyol.py \
    --config configs/tinyol_config.yaml \
    --data_config configs/paderborn_config.yaml \
    --exp_id exp_S22_07 \
    --exp_dir experiments/exp_S22_07/
```

**Structure de sortie attendue** :
```
experiments/exp_S22_07/
├── config_snapshot.yaml
├── results.json
└── training_curves.png
```

**Métriques cibles** (assouplies — biais de classe connu) :

| Métrique | Valeur cible | Résultat | Statut |
| -------- | :----------: | :------: | :----: |
| `acc_final` | ≥ 0.65 | **0.6667** | ❌ |
| `avg_forgetting` | ≤ 0.15 | **0.5000** | ❌ |
| `ram_peak_bytes` | — | 4 393 B | ✅ << 64 Ko |
| `inference_latency_ms` | ≤ 100 ms | 0.009 ms | ✅ |

> ⚠️ **Même effondrement binaire que EWC** : TinyOL apprend K001 (label=0) parfaitement (acc=1.00) puis oublie totalement après T2/T3 (`per_task_acc = [0.00, 1.00, 1.00]`). L'architecture OtO n'apporte pas de protection contre le biais 2 tâches fault vs 1 tâche saine — le backbone gelé compresse K001 vers un embedding distinct, mais la tête OtO converge vers label=1 après T2.

---

## S2218bis — exp_S22_08 : HDC / Paderborn

```bash
python scripts/train_hdc.py \
    --config configs/hdc_config.yaml \
    --data_config configs/paderborn_config.yaml \
    --exp_id exp_S22_08 \
    --exp_dir experiments/exp_S22_08/
```

**Structure de sortie attendue** :

```text
experiments/exp_S22_08/
├── config_snapshot.yaml
├── results.json
└── training_curves.png
```

**Métriques cibles** (assouplies — biais de classe connu) :

| Métrique | Valeur cible | Résultat | Statut |
| -------- | :----------: | :------: | :----: |
| `acc_final` | ≥ 0.65 | **0.3333** | ❌ |
| `avg_forgetting` | ≤ 0.15 | **0.0000** | ✅ |
| `ram_peak_bytes` | — | 14 504 B | ✅ << 64 Ko |
| `inference_latency_ms` | ≤ 100 ms | 0.053 ms | ✅ |

> ⚠️ **Biais inverse — HDC reste bloqué sur le prototype K001** : `per_task_acc = [1.00, 0.00, 0.00]` — HDC prédit tout comme sain (label=0) après T1. L'accumulation additive des prototypes ne déplace pas le classifieur binary vers label=1 : les prototypes fault (T2, T3) sont écrasés par le prototype K001 dominant. `avg_forgetting = 0.000` confirme l'absence d'oubli catastrophique — mais au prix d'un refus d'apprendre les nouvelles classes (forgetting structurel du label=1). Résultat cohérent avec Mahalanobis (0.356).

---

## S2215 — exp_S22_06 : Mahalanobis / Paderborn

```bash
python scripts/train_mahalanobis.py \
    --config configs/paderborn_config.yaml \
    --exp-id exp_S22_06 \
    --output experiments/exp_S22_06/
```

**Métriques cibles** :

| Métrique | Valeur cible | Résultat | Statut |
| -------- | :----------: | :------: | :----: |
| `acc_final` | ≥ 0.65 | **0.3556** | ❌ |
| `avg_forgetting` | ≤ 0.20 | **0.0023** | ✅ |
| `ram_peak_bytes` | — | 1 532 B | ✅ << 64 Ko |
| `inference_latency_ms` | ≤ 100 ms | 0.007 ms | ✅ |

> ⚠️ **acc_final = 0.356 — même biais de classe que EWC** : le seuil Mahalanobis est calibré sur K001 (sain), donc le détecteur prédit "normal" pour K001 (correct) mais aussi pour toutes les faults → acc ≈ 0/3 sur KA04/KI04. La faible forgetting (0.002) confirme que le modèle ne bouge pas — il reste bloqué sur la distribution K001.
>
> → `TODO(arnaud)` : Mahalanobis CL nécessite une reformulation du seuil par tâche, ou une distance de Mahalanobis multi-classe.

**Rôle** : Mahalanobis est une baseline de détection d'anomalie non-supervisée. Sur Paderborn (3 conditions très distinctes spectralement), le seuil calibré sur T0 ne généralise pas aux conditions fault.

---

## S2216 — Notebook `notebooks/results_paderborn_cl.ipynb` ✅ Créé

### Sections requises

1. **Chargement résultats** : lecture de `exp_S22_05/results.json`, `exp_S22_06/results.json`, `exp_S22_07/results.json` et `exp_S22_08/results.json`

2. **Tableau comparatif 4 modèles** :

| Modèle | acc_final | avg_forgetting | BWT | RAM (Ko) |
| ------ | :-------: | :------------: | :------: | :------: |
| EWC | 0.6667 | 0.5000 | -0.5000 | 1.1 |
| Mahalanobis | 0.3556 | 0.0023 | -0.0023 | 1.5 |
| TinyOL | 0.6667 | 0.5000 | -0.5000 | 4.3 |
| HDC | 0.3333 | 0.0000 | +0.0005 | 14.2 |

3. **Courbes AUROC séquentielles** : après tâche K001, après KA04, après KI04 (pour EWC uniquement)

4. **Comparaison avec CWRU** (même famille de défaut roulement) :

```python
# Charger résultats CWRU depuis experiments existantes
import json
cwru_ewc = json.load(open("experiments/exp_XXX/results.json"))  # remplacer XXX
paderborn_ewc = json.load(open("experiments/exp_S22_05/results.json"))

comparison = {
    "CWRU": {"acc_final": cwru_ewc["acc_final"], "af": cwru_ewc["avg_forgetting"]},
    "Paderborn": {"acc_final": paderborn_ewc["acc_final"], "af": paderborn_ewc["avg_forgetting"]},
}
# Barplot côte à côte
```

5. **Interprétation Gap 1** : pourquoi les résultats Paderborn diffèrent-ils de CWRU malgré la similarité physique ? (signal courant vs vibration, conditions stationnaires vs transitoires)

```python
# Cellule obligatoire — Gap 1
print("Gap 1 — Paderborn apporte :")
print("  1. Signal courant moteur (non disponible dans CWRU/Pronostia)")
print("  2. Conditions stationnaires : signal plus propre → acc_final attendu > CWRU")
print("  3. 3e dataset industriel indépendant pour la généralisation CL")
```

---

## Vérification end-to-end

```bash
# Lancer les 4 expériences
python scripts/train_ewc.py --config configs/paderborn_config.yaml --exp-id exp_S22_05 --output experiments/exp_S22_05/
python scripts/train_mahalanobis.py --config configs/paderborn_config.yaml --exp-id exp_S22_06 --output experiments/exp_S22_06/
python scripts/train_tinyol.py --config configs/tinyol_config.yaml --data_config configs/paderborn_config.yaml --exp_id exp_S22_07 --exp_dir experiments/exp_S22_07/
python scripts/train_hdc.py --config configs/hdc_config.yaml --data_config configs/paderborn_config.yaml --exp_id exp_S22_08 --exp_dir experiments/exp_S22_08/

# Vérifier présence
for d in experiments/exp_S22_0{5,6,7,8}/; do
    [ -f "$d/results.json" ] && echo "$d OK" || echo "$d MANQUANT"
done

# Vérifier métriques EWC (cibles originales — résultats déjà documentés)
python -c "
import json
r = json.load(open('experiments/exp_S22_05/results.json'))
assert r['acc_final'] >= 0.80, f\"acc_final {r['acc_final']} < 0.80\"
assert r['avg_forgetting'] <= 0.10, f\"forgetting {r['avg_forgetting']} > 0.10\"
print('exp_S22_05 EWC Paderborn — critères manuscrit OK')
"

# Vérifier métriques TinyOL + HDC (cibles assouplies)
python -c "
import json
for exp, model in [('exp_S22_07', 'TinyOL'), ('exp_S22_08', 'HDC')]:
    r = json.load(open(f'experiments/{exp}/results.json'))
    assert r['acc_final'] >= 0.65, f\"{model} acc_final {r['acc_final']} < 0.65\"
    assert r['avg_forgetting'] <= 0.15, f\"{model} forgetting {r['avg_forgetting']} > 0.15\"
    print(f'{exp} {model} Paderborn — critères OK')
"
```

---

## Questions ouvertes

- ~~`TODO(arnaud)` : HDC et TinyOL à ajouter si résultats EWC insuffisants~~ → résolu : les 2 modèles ajoutés (S2217bis/S2218bis) suite à `acc_final EWC = 0.667`.
- `FIXME(gap1)` : Le notebook doit inclure une section "Diversité datasets Gap 1" résumant CWRU + Pronostia + Monitoring + CMAPSS + Paderborn avec une ligne par dataset dans `docs/datasets_analysis.md`.
