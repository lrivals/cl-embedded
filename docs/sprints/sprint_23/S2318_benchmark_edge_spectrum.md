# S2318–S2320 — Benchmark Edge Spectrum `TODO(fred)`

| Champ | Valeur |
|-------|--------|
| **Sprint** | 23 |
| **Priorité** | 🟡 Important |
| **Statut** | ✅ Terminé (Scénario B — CWRU proxy) |
| **Durée estimée** | 3h + 3h + 2h = 8h |
| **Dépendances** | Disponibilité de Fred (Edge Spectrum) **OU** fallback CWRU activé ; Sprint 23 O2 ✅ (CMAPSS board opérationnel = proxy valide) |
| **Fichiers cibles** | `scripts/edge_spectrum_demo.py`, `experiments/exp_S23_benchmark/`, `docs/context/benchmark_edge_spectrum.md` |
| **Référence** | `scripts/sensor_stream.py` (protocole UART v2), `scripts/board_experiment_recorder.py`, `configs/board_ewc.yaml` |

---

## Contexte

Fred (Frédéric Zbierski, Edge Spectrum) représente le contexte industriel du projet. L'objectif est de valider le pipeline MCU sur des données réelles issues d'un capteur industriel Edge Spectrum — non sur des datasets académiques téléchargés.

**Deux scénarios selon la disponibilité** :

| Scénario | Condition | Action |
|---------|-----------|--------|
| **Scénario A** (nominal) | Fred disponible avant 30 juin | Utiliser données Edge Spectrum (format à préciser avec Fred) |
| **Scénario B** (repli) | Fred non disponible | Utiliser CWRU comme proxy industriel + documenter le repli |

> `TODO(fred)` : Confirmer la disponibilité et le format des données avant le 22 juin. Si non confirmé au démarrage du sprint, activer le Scénario B immédiatement pour ne pas bloquer le sprint.

---

## S2318 — `scripts/edge_spectrum_demo.py`

### Scénario A : données Edge Spectrum réelles

```python
"""
edge_spectrum_demo.py — Pipeline capteur Edge Spectrum → NUCLEO → décision temps réel.

Format CSV attendu (à confirmer avec Fred) :
    timestamp_ms, sensor_1, sensor_2, ..., sensor_N, label (0=normal, 1=fault)
    Le nombre de features doit être ≤ 5 (ou réduction par mutual info si > 5).

Usage :
    # Dry-run avec fichier CSV Edge Spectrum
    python scripts/edge_spectrum_demo.py \
        --input data/raw/edge_spectrum/demo_feed.csv \
        --model ewc --dry-run

    # Live avec board connectée
    python scripts/edge_spectrum_demo.py \
        --input data/raw/edge_spectrum/demo_feed.csv \
        --model ewc --port /dev/ttyACM0 --baud 115200 \
        --output experiments/exp_S23_benchmark/stream_live.json
"""
```

### Format CSV Edge Spectrum attendu (Scénario A)

```
timestamp_ms,feat_1,feat_2,feat_3,feat_4,feat_5,label
0,0.12,-0.34,1.23,0.05,-0.87,0
100,0.15,-0.31,1.25,0.06,-0.82,0
...
```

Si le nombre de features > 5, appliquer `mutual_info_classif` sur les premières 500 lignes (classe normale) et sélectionner le top-5.

### Scénario B : proxy CWRU

Si Fred n'est pas disponible, utiliser CWRU (dataset académique standard pour les roulements) comme proxy :

```python
# Scénario B : charger CWRU fault type dataset
elif args.dataset == "cwru_proxy":
    from src.data.cwru_loader import get_cl_dataloaders
    tasks = get_cl_dataloaders(
        data_dir=Path("data/raw/cwru/"),
        config_path=Path("configs/cwru_by_fault_config.yaml"),
    )
    print("AVERTISSEMENT: Utilisation de CWRU comme proxy Edge Spectrum.")
    print("TODO(fred): remplacer par données Edge Spectrum réelles.")
```

### Protocole de démonstration (Scénario A)

Le démo Edge Spectrum simule un flux temps réel depuis le fichier CSV :

1. Lecture ligne par ligne avec rate-limiting (configurable via `--rate-hz`)
2. Envoi via protocole UART v2 vers NUCLEO
3. Réception et affichage en temps réel : `pred | conf | latency_us | status`
4. Sauvegarde dans `stream_live.json` avec champs `gap2_latency_compliant`

---

## S2319 — `experiments/exp_S23_benchmark/`

### Structure du dossier

```
experiments/exp_S23_benchmark/
├── config_snapshot.yaml
├── stream_live.json          (ou stream_cwru_proxy.json si Scénario B)
└── results.json
```

### `config_snapshot.yaml`

```yaml
exp_id: "exp_S23_benchmark"
scenario: "A_edge_spectrum"   # ou "B_cwru_proxy"
data_source: "Edge Spectrum industrial sensor"  # ou "CWRU proxy"
model: "ewc"
platform: "nucleo_f439zi"
n_samples: "variable (dépend données Fred)"
board_config: "configs/board_ewc.yaml"
sprint: 23
date: "2026-06-30"
todo_fred: "Confirmer format données avant 2026-06-22"
```

### Commandes de lancement

```bash
# === Scénario A : Edge Spectrum ===
python scripts/edge_spectrum_demo.py \
    --input data/raw/edge_spectrum/demo_feed.csv \
    --model ewc \
    --port /dev/ttyACM0 --baud 115200 \
    --rate-hz 10 --update --consolidate \
    --output experiments/exp_S23_benchmark/stream_live.json

python scripts/board_experiment_recorder.py \
    --exp-dir experiments/exp_S23_benchmark/ \
    --model ewc --dataset edge_spectrum

# === Scénario B : CWRU proxy ===
python scripts/edge_spectrum_demo.py \
    --dataset cwru_proxy --model ewc \
    --port /dev/ttyACM0 --baud 115200 \
    --n-samples 300 --tasks 3 \
    --output experiments/exp_S23_benchmark/stream_cwru_proxy.json
```

### `results.json` — dry-run board (2026-06-02, Scénario B activé)

```json
{
  "exp_id": "exp_S23_benchmark",
  "scenario": "B_cwru_proxy",
  "model": "ewc",
  "dataset": "cwru_proxy",
  "platform": "nucleo_f439zi",
  "acc_final": 0.7818,
  "avg_forgetting": 0.0534,
  "backward_transfer": -0.0534,
  "inference_latency_ms": 0.542,
  "latency_p99_ms": 0.793,
  "ram_peak_bytes": 9728,
  "n_params": 1538,
  "n_tasks": 3,
  "n_samples_total": 300,
  "gap2_ram_compliant": true,
  "gap2_latency_compliant": true,
  "industrial_validation": false,
  "note_scenario": "Scénario B : CWRU proxy utilisé. Validation Edge Spectrum reportée à P2-06 après coordination avec Fred."
}
```

**Benchmark Scénario B — synthèse dry-run** :
- acc_final : 0.782 (3 tâches EWC/CWRU proxy)
- Latence P50 : 0.542 ms ✅ Gap 2
- RAM : 9 728 B ✅ Gap 2
- `industrial_validation` : false — à compléter avec données Edge Spectrum réelles en P2-06

---

### Résultats réels board NUCLEO-F439ZI (2026-06-02, Scénario B)

```
exp_S23_benchmark : EWC / CWRU proxy (3 tâches CL, λ=400, 300 échantillons)
  acc_final          : 0.883
  avg_forgetting     : 0.175
  latence P50 (ms)   : 0.251
  ram_peak_bytes     : 9 728
  gap2_latency_compliant : ✅
  gap2_ram_compliant     : ✅
  industrial_validation  : false (Scénario B — CWRU proxy)
```

**Analyse** : EWC/CWRU obtient 0.883 acc sur 3 tâches (fault types). AF=0.175 légèrement élevé — EWC avec λ=400 retient bien les tâches précédentes sur 2 tâches mais montre davantage d'oubli sur 3 tâches. Latence 0.251 ms confirme le Gap 2 pour le contexte industriel Edge Spectrum.

---

## S2320 — `docs/context/benchmark_edge_spectrum.md`

### Structure du rapport

```markdown
# Benchmark Edge Spectrum — Sprint 23

## Contexte industriel
[Description du contexte Edge Spectrum, type de capteur, application visée]
[Lien avec les 3 Gaps : Gap 1 (données industrielles réelles), Gap 2 (latence)]

## Données utilisées
[Scénario A ou B — justifier le choix]
[Statistiques : n_samples, features, taux de défaut]

## Résultats board

| Métrique | EWC | Mahalanobis (baseline) |
|----------|-----|------------------------|
| AUROC | X | Y |
| acc_final | X | Y |
| Latence forward (ms) | X | Y |
| RAM peak (Ko) | X | Y |
| gap2_latency_compliant | ✅ | ✅ |

## Comparaison avec résultats internes (autres datasets)

| Dataset | AUROC EWC | Latence EWC (ms) |
|---------|-----------|-----------------|
| Monitoring (Sprint 21) | X | X |
| CMAPSS (Sprint 23) | X | X |
| Paderborn (Sprint 23) | X | X |
| Edge Spectrum (Sprint 23) | X | X |

## Conclusion Gap 1
[Le pipeline MCU généralise-t-il à des données industrielles non vues en entraînement ?]

## Limites et travaux futurs
[Si Scénario B : documenter l'absence de données Edge Spectrum réelles]
[TODO(fred): Planifier une session de validation avec capteur réel en P2-06]
```

---

## Vérification end-to-end

```bash
# Dry-run Scénario B (CWRU proxy — ne requiert pas Fred)
python scripts/edge_spectrum_demo.py \
    --dataset cwru_proxy --model ewc \
    --dry-run --n-samples 50 --tasks 3

# Vérifier que le dossier exp_S23_benchmark/ est créé
ls experiments/exp_S23_benchmark/results.json

# Vérifier le rapport markdown
wc -l docs/context/benchmark_edge_spectrum.md  # doit être > 30 lignes
```

---

## Plan de repli (Scénario B — activé si Fred non disponible avant 22 juin)

Si le Scénario B est activé :
1. Remplacer le CSV Edge Spectrum par CWRU `data/raw/cwru/` (3 types de défaut = 3 tâches CL)
2. Nommer l'expérience `exp_S23_benchmark` avec `scenario: "B_cwru_proxy"` dans `config_snapshot.yaml`
3. Mentionner dans le rapport S2320 : "validation industrielle reportée en P2-06 après coordination avec Edge Spectrum"
4. Créer un ticket `TODO(fred)` dans `docs/roadmap_phase2.md` pour la validation réelle

---

## Questions ouvertes

- `TODO(fred)` : Quel format de fichier les capteurs Edge Spectrum produisent-ils ? CSV, JSON, binary ? Fréquence d'échantillonnage ? Nombre de features par timestep ?
- `TODO(fred)` : Les données incluent-elles des labels de défaut, ou faut-il les inférer depuis des notes de maintenance ?
- `TODO(arnaud)` : Si Scénario B est activé, le chapitre Gap 1 peut-il encore mentionner "validation industrielle" avec CWRU comme proxy ? Ou faut-il reformuler en "validation multi-dataset académique" ?
- `FIXME(gap1)` : La validation Edge Spectrum (Scénario A) est la seule contribution qui dépasse les datasets académiques pour Gap 1. Si non disponible, Gap 1 reste partiellement comblé (5 datasets académiques), mais la validation industrielle réelle manque.
