# S3305 / S3306 — Capture énergie (LPM01A) + campagne board

| Champ | Valeur |
|-------|--------|
| **Sprint** | 33 |
| **Priorité** | 🔴 Critique |
| **Statut** | ⬜ À démarrer |
| **Durée estimée** | 4h (S3305) + 4h (S3306) |
| **Dépendances** | S3304 (marqueurs de phase) · PowerShield X-NUCLEO-LPM01A + STM32CubeMonitor-Power ✅ (validé utilisateur) |
| **Fichiers cibles** | `scripts/energy_capture.py`, `experiments/exp_S33_energy/` |
| **Références** | `scripts/sensor_stream.py` (pattern CLI / pilotage série), `scripts/board_experiment_recorder.py` (pattern enregistrement expérience) |

---

## Contexte

Le CR du 9 juin 2026 demande explicitement « utiliser STM32 Monitor Power pour le profilage
énergétique » et « comparer les métriques FP32/INT8 côté hardware ». Aucun script
d'acquisition énergie n'existe (confirmé). Ce doc couvre le pilote LPM01A **et** la campagne
de mesure qui l'utilise directement (4 modèles × 2 encodages), car l'un ne se valide qu'avec
l'autre.

---

## S3305 — `scripts/energy_capture.py`

```python
"""Pilote STM32CubeMonitor-Power / LPM01A (CLI ou export CSV), segmente le courant/tension
selon les marqueurs de phase S3304, intègre en µJ par phase, exporte un JSON normalisé.
"""

def capture_session(duration_s: float, sampling_rate_hz: float, output_csv: Path) -> Path:
    """Lance une capture LPM01A (via CLI STM32CubeMonitor-Power ou import CSV existant)."""
    ...

def segment_by_phase(csv_path: Path, phase_timestamps: list[tuple[str, float, float]]) -> dict:
    """Découpe la trace courant/tension selon les fenêtres [t_start, t_end] par EnergyPhase
    (corrélées aux marqueurs GPIO/DWT de S3304).
    """
    ...

def integrate_energy_uj(courant_a: np.ndarray, tension_v: np.ndarray, dt_s: float) -> float:
    """E (uJ) = somme(I x V x dt) x 1e6."""
    ...

def export_energy_json(phases_uj: dict, model: str, encoding: str, output_path: Path) -> None:
    """{"model": ..., "encoding": "fp32"|"int8", "phases_uj": {"startup":..., "acquisition":...,
    "inference":..., "idle":...}, "total_uj":..., "timestamp": ...}"""
    ...
```

## S3306 — Campagne board

Enregistrement `experiments/exp_S33_energy/{modele}_{fp32,int8}.json` pour
{EWC, HDC, TinyOL, Mahalanobis} × {FP32, INT8} + `summary.json` (table comparative FP32 vs
INT8 énergie, lien explicite avec le résultat Gap 3 — réduction RAM sans accélération
latence FPU, Sprint 29).

---

**Règles** :
- **Aucun chiffre inventé** : tant que la board + LPM01A n'ont pas réellement tourné, les
  champs JSON portent la valeur littérale `"à mesurer"` plutôt qu'un nombre placeholder.
- Reporter la fréquence d'échantillonnage et la calibration LPM01A comme `TODO(dorra)` dans
  le JSON ou la docstring si non encore définies.
- Suivre le pattern d'enregistrement de `board_experiment_recorder.py` (déjà existant) pour
  la structure de répertoire `experiments/exp_S33_energy/`.

---

## Vérification

```bash
python scripts/energy_capture.py --model ewc --encoding fp32 \
    --duration 10 --output experiments/exp_S33_energy/ewc_fp32.json

python -c "import json; d=json.load(open('experiments/exp_S33_energy/ewc_fp32.json')); \
assert 'phases_uj' in d"

ls experiments/exp_S33_energy/   # 8 JSON modèle x encodage + summary.json attendus
```

---

## Complétion — segmentation depuis CSV réel (sans LPM01A posé)

La chaîne `_load_csv → segment_by_phase → integrate_energy_uj → export_energy_json`
est désormais **débloquée et exercée bout-en-bout** (auparavant `_capture_one` levait
`NotImplementedError` sur un vrai CSV, faute de table de timestamps de phase).

- **Source des frontières de phase = colonne de sync PA8** dans le même CSV LPM01A.
  Le firmware émet déjà le signal GPIO PA8 (S3304, câblé dans `pipeline.c`) ; le LPM01A
  le capture en parallèle du courant. `_load_csv` reconnaît la colonne (`sync`/`pa8`/
  `gpio`/`digital`/`marker`) ; `derive_phase_windows(trace)` convertit les fronts en
  fenêtres `(phase, t_start, t_end)`. Alignement par construction, **aucune horloge
  hôte↔LPM01A à synchroniser**.
- **Limitation 1-bit (assumée, pas une invention)** : avec le schéma actuel, les phases
  `startup`/`acquisition`/`inference` partagent le même niveau HAUT → la déduction reporte
  honnêtement chaque plateau haut comme `inference` et chaque plateau bas comme `idle`.
  Une granularité 4-phases nécessiterait un encodage multi-bit côté firmware → évolution
  future, hors périmètre S33.
- **Règle « aucun chiffre inventé » préservée** : sans CSV (`--campaign`), les JSON restent
  en `"à mesurer"`. Aucun chiffre n'est écrit dans `experiments/exp_S33_energy/` tant que la
  sonde LPM01A n'a pas réellement tourné. Le CSV synthétique des tests reste cantonné à
  `tmp_path` (fixtures vérifiables à la main).
- **Tests** : `tests/test_energy_capture.py` (16 PASS) couvre intégration µJ, parsing
  tolérant, `derive_phase_windows`, segmentation, export placeholder/mesuré et le
  bout-en-bout `_capture_one` sur trace synthétique.
