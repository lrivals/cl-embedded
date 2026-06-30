# S3513 — Tests (Python + Unity)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 35 |
| **Priorité** | 🟢 Nice-to-have (mais jamais absent — règle `sprint_generation.md`) |
| **Statut** | ⬜ À démarrer |
| **Durée estimée** | 3h |
| **Dépendances** | S3501 (sélection), S3504 (F1), S3510 (heatmap builders), S3506 (dims firmware) |
| **Fichiers cibles** | `tests/test_feature_selection.py`, `tests/test_heatmap_builders.py`, (Unity) `firmware/stm32f4_blink/tests/test_pipeline.c` |

---

## Spec

- `tests/test_feature_selection.py` :
  - permutation importance déterministe (seed) sur cas connu ;
  - k\* optimisé sur F1 val cohérent (k\* ≤ n_features, F1[k\*] ≥ F1[k] règle de parcimonie) ;
  - F1 `faulty` correct sur matrice de confusion connue (S3504).
- `tests/test_heatmap_builders.py` :
  - le builder produit une matrice 5×4 par `(metric, condition, platform)` ;
  - cellules `pending`/`N/A` masquées, pas de NaN affiché en valeur ;
  - HDC×monitoring board ≠ 0.113 après fix (S3509).
- **Unity** (si S3506 a touché les dims) : `make test` vert, `test_pipeline.c`/`test_ewc_head.c`
  inchangés en condition 5-feat (non-régression) ; build aux dims `all`/`best` compile.

**Règle** : tests jamais 🔴, mais jamais absents.

## Vérification

```bash
pytest tests/test_feature_selection.py tests/test_heatmap_builders.py -v
cd firmware/stm32f4_blink && make test   # Unity vert (0 régression hors 2 TinyOL préexistants)
```
