# S2324–S2325 — Tests + Documentation finale Sprint 23

| Champ | Valeur |
|-------|--------|
| **Sprint** | 23 |
| **Priorité** | 🟡 Important |
| **Statut** | ✅ Terminé — 2026-06-02 |
| **Durée estimée** | 1h + 30 min = 1h30 |
| **Dépendances** | S2301–S2323 ✅ (toutes les expériences et implémentations terminées) |
| **Fichiers cibles** | `tests/` (tests CMAPSS + Paderborn stream), `docs/roadmap_phase2.md` |
| **Référence** | `tests/test_cmapss_loader.py`, `tests/test_paderborn_loader.py` (Sprint 22, S2223), `docs/roadmap_phase2.md` actuel |

---

## Contexte

Ces deux tâches clôturent le sprint 23 : vérification que les tests existants sur les loaders CMAPSS et Paderborn (créés en Sprint 22) couvrent aussi le streaming board, puis mise à jour de la roadmap pour refléter la complétion du sprint et l'état final des 3 Gaps.

---

## S2324 — `pytest tests/ -k "cmapss or paderborn"` vert

### Tests de streaming à ajouter

Sprint 22 a créé `tests/test_cmapss_loader.py` et `tests/test_paderborn_loader.py` qui couvrent le chargement des données. Sprint 23 a étendu `sensor_stream.py` avec `--dataset cmapss` et `--dataset paderborn`. Il faut vérifier que le streaming dry-run fonctionne.

#### Fichier `tests/test_cmapss_stream.py`

```python
"""Tests du streaming CMAPSS via sensor_stream.py."""

import pytest
from pathlib import Path
import subprocess
import sys


class TestCmapssStream:
    """Tests dry-run du streaming CMAPSS — ne requièrent pas de board."""

    def test_cmapss_stream_dryrun_runs(self, tmp_path):
        """sensor_stream.py --dataset cmapss --dry-run ne lève pas d'exception."""
        result = subprocess.run(
            [
                sys.executable, "scripts/sensor_stream.py",
                "--dataset", "cmapss",
                "--model", "ewc",
                "--dry-run",
                "--n-samples", "10",
                "--tasks", "2",
                "--output", str(tmp_path / "stream_test.json"),
            ],
            capture_output=True, text=True, timeout=60
        )
        assert result.returncode == 0, f"Erreur stream CMAPSS : {result.stderr}"

    @pytest.mark.skipif(
        not Path("configs/cmapss_feature_subset.yaml").exists(),
        reason="cmapss_feature_subset.yaml non généré (lancer S2305 d'abord)"
    )
    def test_cmapss_feature_subset_loaded(self):
        """Le fichier cmapss_feature_subset.yaml est chargé sans erreur."""
        import yaml
        d = yaml.safe_load(Path("configs/cmapss_feature_subset.yaml").read_text())
        assert "selected_features" in d
        assert len(d["selected_features"]) == 5

    def test_cmapss_stream_hdc_dryrun(self, tmp_path):
        """sensor_stream.py --model hdc --dataset cmapss --dry-run (S2304 intégré)."""
        result = subprocess.run(
            [
                sys.executable, "scripts/sensor_stream.py",
                "--dataset", "cmapss",
                "--model", "hdc",
                "--dry-run",
                "--n-samples", "10",
                "--tasks", "2",
                "--output", str(tmp_path / "stream_hdc.json"),
            ],
            capture_output=True, text=True, timeout=60
        )
        assert result.returncode == 0, f"Erreur stream HDC CMAPSS : {result.stderr}"
```

#### Fichier `tests/test_paderborn_stream.py`

```python
"""Tests du streaming Paderborn via sensor_stream.py."""

import pytest
from pathlib import Path
import subprocess
import sys


class TestPaderbornStream:

    def test_paderborn_stream_dryrun_runs(self, tmp_path):
        """sensor_stream.py --dataset paderborn --dry-run ne lève pas d'exception."""
        result = subprocess.run(
            [
                sys.executable, "scripts/sensor_stream.py",
                "--dataset", "paderborn",
                "--model", "ewc",
                "--dry-run",
                "--n-samples", "10",
                "--tasks", "3",
                "--output", str(tmp_path / "stream_pad.json"),
            ],
            capture_output=True, text=True, timeout=60
        )
        assert result.returncode == 0, f"Erreur stream Paderborn : {result.stderr}"

    @pytest.mark.skipif(
        not Path("configs/paderborn_feature_subset.yaml").exists(),
        reason="paderborn_feature_subset.yaml non généré (lancer S2311 d'abord)"
    )
    def test_paderborn_feature_subset_loaded(self):
        import yaml
        d = yaml.safe_load(Path("configs/paderborn_feature_subset.yaml").read_text())
        assert "selected_features" in d
        assert len(d["selected_features"]) == 5
```

### Commandes de vérification

```bash
# Tests sans données (toujours passent)
pytest tests/test_cmapss_loader.py tests/test_paderborn_loader.py -v

# Tests stream (requièrent feature_subset.yaml mais pas de board)
pytest tests/test_cmapss_stream.py tests/test_paderborn_stream.py -v

# Commande sprint — critère de clôture
pytest tests/ -k "cmapss or paderborn" -v
# Attendu : tous les tests verts (les tests @skipif passent si données absentes)

# Rapport coverage (optionnel)
pytest tests/ -k "cmapss or paderborn" --tb=short -q
```

---

## S2325 — Mise à jour `docs/roadmap_phase2.md`

### Modifications à apporter

1. **Sprint 23** : passer de `⬜ À démarrer` → `✅ Terminé` (ou `🔄 En cours` si partiel) avec la date de clôture réelle
2. **Tableau Triple Gap** : mettre à jour les statuts Gap 1, Gap 2, Gap 3
3. **Section Sprint 23** : insérer le bilan des 7 expériences board

### Format de la section Sprint 23 dans la roadmap

```markdown
### Sprint 23 — 22 juin – 5 juillet 2026 ✅ Terminé

**Livrables** :
- `firmware/stm32f4_blink/src/hdc.c` complet (binarize + retrain) + tests Unity ≥10 ✅
- `firmware/stm32f4_blink/src/ewc_head_int8.c` intégré dans pipeline ✅
- `configs/board_cmapss.yaml` + `configs/board_paderborn.yaml` ✅
- 7 expériences board : exp_S23_01–06 + exp_S23_INT8 ✅
- `experiments/comparison_sprint23.json` (5 datasets × 4 modèles) ✅
- Notebook `board_benchmark_all_datasets.ipynb` + figure manuscrit ✅
- `docs/context/benchmark_edge_spectrum.md` ✅

**Résultats clés board** :
| Exp | Modèle | Dataset | lat_ms | acc_final | Gap 2 |
|-----|--------|---------|:------:|:---------:|:-----:|
| exp_S23_01 | EWC | CMAPSS | TBD | TBD | ✅ |
| exp_S23_02 | TinyOL | CMAPSS | TBD | TBD | ✅ |
| exp_S23_03 | Maha | CMAPSS | TBD | TBD | ✅ |
| exp_S23_04 | HDC | CMAPSS | TBD | TBD | ✅ |
| exp_S23_05 | EWC | Paderborn | TBD | TBD | ✅ |
| exp_S23_06 | Maha | Paderborn | TBD | TBD | ✅ |
| exp_S23_INT8 | EWC INT8 | CMAPSS | TBD | TBD | — |
```

### Tableau Triple Gap — statut final

```markdown
## Triple Gap — Statut final après Sprint 23

| Gap | Description | Statut |
|-----|-------------|:------:|
| **Gap 1** | Validation sur 5 datasets industriels (CWRU + Monitoring + Pronostia + CMAPSS + Paderborn) | ✅ |
| **Gap 2** | Latence < 100 ms sur NUCLEO-F439ZI, 4 modèles × 5 datasets, RAM ≤ 64 Ko | ✅ |
| **Gap 3** | Quantification INT8 pendant l'entraînement incrémental, Δ AUROC < 0.02 | ⚠️ Partiel (voir exp_S23_INT8) |

**Gap 3** : si `latency_int8 < latency_fp32` → ✅. Sinon → ⚠️ (réduction RAM ×2.7 documentée, pas d'accélération latence sur Cortex-M4 FPU — résultat négatif honnête).

> **Rédaction manuscrit** : ces trois gaps forment le fil directeur du chapitre 4 (Évaluation Expérimentale). Chaque tableau de résultats doit référencer le gap comblé.
```

---

## Vérification end-to-end Sprint 23

```bash
# 1. Tests Unity HDC (S2303)
gcc -O0 -I firmware/stm32f4_blink/inc \
    -I firmware/stm32f4_blink/tests/unity/src \
    firmware/stm32f4_blink/tests/unity/src/unity.c \
    firmware/stm32f4_blink/tests/test_hdc.c \
    firmware/stm32f4_blink/src/hdc.c \
    -lm -o /tmp/test_hdc && /tmp/test_hdc
# Attendu : >= 10/10 PASS

# 2. Tests Python CMAPSS + Paderborn
pytest tests/ -k "cmapss or paderborn" -v --tb=short
# Attendu : tous verts

# 3. Vérifier les 7 dossiers experiments
for exp in exp_S23_01 exp_S23_02 exp_S23_03 exp_S23_04 exp_S23_05 exp_S23_06 exp_S23_INT8; do
    test -f "experiments/$exp/results.json" && echo "$exp OK" || echo "$exp MANQUANT"
done

# 4. Vérifier le JSON de comparaison
python -c "
import json
d = json.load(open('experiments/comparison_sprint23.json'))
print('Datasets :', list(d['results'].keys()))
assert len(d['results']) >= 4
print('comparison_sprint23.json OK')
"

# 5. Critères de succès sprint (S2300)
python -c "
import json, glob
exps = glob.glob('experiments/exp_S23_*/results.json')
print(f'{len(exps)} expériences board trouvées (attendu: 7)')
compliant = [e for e in exps
             if json.load(open(e)).get('gap2_latency_compliant', False)]
print(f'{len(compliant)} gap2_latency_compliant=true')
"
```

---

## Résultats (2026-06-02)

### S2324 — Tests streaming ✅

```text
pytest tests/ -k "cmapss or paderborn" -v
31 passed, 439 deselected in 70.53s
```

| Fichier | Tests | Résultat |
| ------- | :---: | :------: |
| `tests/test_cmapss_stream.py` | 3 | ✅ PASS |
| `tests/test_paderborn_stream.py` | 2 | ✅ PASS |
| `tests/test_cmapss_loader.py` | 10 | ✅ PASS |
| `tests/test_paderborn_loader.py` | 16 | ✅ PASS |

**Correction apportée** : `--tasks N` → `--n-tasks N` (flag réel de `sensor_stream.py` — la spec utilisait le mauvais nom d'argument).

### S2325 — Roadmap ✅

- Sprint 23 : `🔄 EN COURS` → `✅ TERMINÉ` dans header, vue macro, section sprint et ligne O7
- Section Livrables + tableau 7 expériences board ajoutés sous la table objectifs
- Triple Gap renommé "Statut final après Sprint 23", Gap 1 étendu à 5 datasets, Gap 3 → ⚠️ Partiel avec note honnête

---

## Questions ouvertes

- `TODO(arnaud)` : Le tableau Triple Gap dans la roadmap doit-il être visible dans le README du dépôt pour le rapport de stage ? Ou uniquement dans la roadmap interne ?
- `FIXME(gap3)` : La formulation "Gap 3 partiel" doit être discutée avec Arnaud — dans certains cas, une réduction RAM × 2.7 sans accélération latence peut quand même être présentée comme contribution originale si aucun travail précédent ne l'a mesurée sur MCU.
