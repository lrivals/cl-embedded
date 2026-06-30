# S3307 — `src/evaluation/autonomy.py` + RAM profiling

| Champ | Valeur |
|-------|--------|
| **Sprint** | 33 |
| **Priorité** | 🟡 Important |
| **Statut** | ⬜ À démarrer |
| **Durée estimée** | 2h |
| **Dépendances** | S3306 (µJ par phase mesurés) |
| **Fichiers cibles** | `src/evaluation/autonomy.py`, `experiments/exp_S33_energy/autonomy.json` |
| **Références** | `src/evaluation/memory_profiler.py:25` (`profile_forward_pass`), `:89` (`profile_cl_update`), `:142` (`full_memory_report`) |

---

## Contexte

Dernière étape de la chaîne énergie : dériver une **autonomie estimée** (heures) à partir
des µJ/phase mesurés en S3306 et d'une capacité de batterie typique, pour répondre à la
question du CR « combien d'accuracy perd-on pour gagner en RAM/autonomie ? ». Nouveau
module → RAM profiling obligatoire (règle CLAUDE.md : tout nouveau modèle/module mesuré).

---

## Spec

```python
# src/evaluation/autonomy.py

def average_current_ma(phases_uj: dict, phase_durations_s: dict, tension_v: float = 3.3) -> float:
    """I_moy (mA) = somme(I_phase x t_phase) / T_cycle, dérivé des µJ/phase et durées
    mesurées (S3306) : I_phase = (uJ_phase / 1e6) / (tension_v x t_phase).
    """
    ...

def autonomy_hours(capacite_mah: float, i_moy_ma: float) -> float:
    """Autonomie_h = Capacite_mAh / I_moy_mA."""
    return capacite_mah / i_moy_ma

def sweep_capacities(i_moy_ma: float, capacites_mah: list[float]) -> dict[float, float]:
    """Balayage de capacités batterie typiques (depuis une config), retourne {capacite: heures}."""
    ...
```

**Règles** :
- Capacités de batterie typiques (`capacites_mah`) lues depuis une config (pas en dur dans
  le code) — réutiliser `configs/hw_profile_f439zi.yaml` (S3302) ou un fichier dédié.
- **RAM profiling obligatoire** : appliquer `profile_forward_pass()` /
  `full_memory_report()` (pattern existant `memory_profiler.py:25,142`) au module
  `autonomy.py` lui-même, puisqu'il introduit du nouveau code mesuré.
- Sortie `experiments/exp_S33_energy/autonomy.json` : un par modèle×encodage + un balayage
  de capacités.

---

## Vérification

```bash
python -c "from src.evaluation.autonomy import average_current_ma, autonomy_hours; \
i = average_current_ma({'inference': 100.0}, {'inference': 0.0005}); print(autonomy_hours(220, i))"

python scripts/profile_memory.py --model autonomy   # RAM profiling du nouveau module

pytest tests/test_autonomy.py -v   # S3309 : I_moy / autonomie
```
