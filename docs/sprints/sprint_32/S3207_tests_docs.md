# S3207 — Tests + documentation

| Champ | Valeur |
|-------|--------|
| **Sprint** | 32 |
| **Priorité** | 🟢 Bas |
| **Statut** | ✅ Implémenté |
| **Durée estimée** | 4h |
| **Résultat** | `tests/test_threshold_sweep.py` **16/16 PASS** (defaults inchangés 3 loaders, opérateurs `<=`/`<`, configs sweep ne diffèrent que par le seuil, subset battery 5/7, `positive_ratio` monotone cmapss/pronostia/battery). Tests Unity firmware **94/96** (2 TinyOL préexistants hors périmètre, inchangés). Roadmap + CLAUDE.md MAJ ; `graphify_sprint_update` invoqué. |
| **Dépendances** | S3201–S3206 |
| **Fichiers cibles** | `tests/test_threshold_sweep.py`, `docs/roadmap_phase2.md`, `CLAUDE.md` |
| **Références** | `tests/test_cmapss_loader.py`, `tests/test_battery_dataset.py`, `skills/graphify_sprint_update.md` |

---

## Contexte

Verrouiller le paramétrage du seuil par des tests et clore le sprint (roadmap, CLAUDE.md, graphe).

---

## Spec

```python
# tests/test_threshold_sweep.py
def test_default_threshold_unchanged():
    # config sans champ seuil -> labels identiques à l'existant (les 3 loaders)

def test_threshold_param_applied():
    # faulty_threshold injecté -> binarisation conforme (CMAPSS '<=', Battery '<')

def test_positive_ratio_monotonic():
    # seuil restrictif (RUL élevé) -> plus de positifs ; ratio croît avec le seuil

def test_sweep_configs_only_threshold_differs():
    # 2 configs sweep d'un même dataset ne diffèrent que sur le champ seuil
```

- Tests Unity firmware **restent verts** (`make test`) — le seuil n'affecte pas le code C.
- MAJ `docs/roadmap_phase2.md` (ligne Sprint 32) + statut sprint dans `CLAUDE.md`.
- Invoquer le skill `graphify_sprint_update` (évalue si un update du graphe est pertinent).

---

## Vérification

```bash
pytest tests/test_threshold_sweep.py -v
make -C firmware/stm32f4_blink test   # 0 nouvelle régression
```
