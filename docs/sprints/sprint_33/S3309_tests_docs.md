# S3309 — Tests + documentation (Sprint 33)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 33 |
| **Priorité** | 🟢 Bas (mais jamais absente) |
| **Statut** | ⬜ À démarrer |
| **Durée estimée** | 2h |
| **Dépendances** | S3301–S3308 |
| **Fichiers cibles** | `tests/test_compute_cost.py`, `tests/test_hw_cost_model.py`, `tests/test_autonomy.py`, `docs/roadmap_phase2.md`, `CLAUDE.md` |
| **Références** | `tests/` (conventions pytest existantes), `skills/graphify_sprint_update.md` |

---

## Contexte

Verrouille les nouvelles métriques de coût/énergie par des tests et clôture le sprint
(roadmap, statut `CLAUDE.md`, graphe de connaissance).

---

## Spec

```python
# tests/test_compute_cost.py
def test_flops_equals_2x_macs():
    # compute_flops(macs) == 2 * macs, pour chaque modèle existant

def test_bops_fp32_vs_int8_ratio():
    # compute_bops(macs, 32) / compute_bops(macs, 8) == 16   (32**2 / 8**2)

def test_macs_non_regression():
    # valeurs macs_*() identiques avant/après extension S3301 (snapshot des valeurs actuelles)

# tests/test_hw_cost_model.py
def test_efficacite_bounds():
    # efficacite in [0.1, 0.6] respectée par la config par défaut

def test_flops_per_watt_positive():
    ...

def test_throughput_consistent_with_latency():
    # throughput(t) == 1/t, cohérence avec estimate_inference_time

# tests/test_autonomy.py
def test_average_current_matches_manual_calc():
    ...

def test_autonomy_hours_decreases_with_higher_current():
    ...
```

- Tests Unity firmware **restent verts** (`make test`) — S3304 (marqueurs GPIO) ne doit
  introduire aucune régression sous compilation standard (sans `ENERGY_MARKERS`).
- MAJ `docs/roadmap_phase2.md` (ligne Sprint 33) + statut sprint dans `CLAUDE.md` + volet
  énergie dans `docs/triple_gap.md` (lien Gap 3).
- Invoquer le skill `graphify_sprint_update` (évalue la pertinence d'un update du graphe).

---

## Vérification

```bash
pytest tests/ -k "compute_cost or hw_cost or autonomy or energy_capture" -v
make -C firmware/stm32f4_blink test   # 0 nouvelle régression
```

---

## Complétion — couverture `energy_capture.py`

Nouveau `tests/test_energy_capture.py` (**16 PASS**) : intégration µJ (`integrate_energy_uj`),
parsing CSV tolérant (`_load_csv`, avec/sans header, colonnes sync/tension présentes ou
absentes), déduction des fenêtres de phase depuis le signal PA8 (`derive_phase_windows`),
segmentation (`segment_by_phase`, phases vides/inconnues, refus sans tension), export JSON
(chemin placeholder `"à mesurer"` vs mesuré) et **chaîne bout-en-bout** `_capture_one` sur CSV
synthétique. Total suite énergie : **47 PASS** (16 energy_capture + 16 compute_cost +
7 hw_cost_model + 8 autonomy).
