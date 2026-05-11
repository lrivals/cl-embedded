# S19-02 — Vérification loader `get_pronostia_dataloaders_anomaly_detection()`

| Champ | Valeur |
|-------|--------|
| **ID** | S19-02 |
| **Sprint** | Sprint 19 |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 1h |
| **Dépendances** | S19-01 (scénario documenté) |
| **Fichier cible** | `src/data/pronostia_dataset.py` |

---

## Objectif

Vérifier que `get_pronostia_dataloaders_anomaly_detection()` — déjà implémentée en Sprint 15 — est compatible avec le scénario by_bearing_condition et le mode anomaly detection one-class. Adapter si nécessaire (ajout du paramètre `scenario`, vérification que train_loader ne contient que des normaux).

---

## Contexte

La fonction `get_pronostia_dataloaders_anomaly_detection()` a été implémentée lors du Sprint 15 pour l'anomaly detection Pronostia. Le Sprint 19 réutilise cette implémentation sans la réécrire. Cette tâche vérifie uniquement que l'interface est conforme aux attentes du Sprint 19.

---

## Interface attendue

```python
def get_pronostia_dataloaders_anomaly_detection(
    data_path: str | Path,
    scenario: Literal["by_bearing_condition"] = "by_bearing_condition",
    test_size: float = 0.2,
    batch_size: int = 32,
    seed: int = 42,
) -> list[tuple[int, str, DataLoader, DataLoader]]:
    """
    Retourne 3 tâches Pronostia pour anomaly detection one-class.

    Returns
    -------
    list of (task_id, task_name, train_loader_normal, test_loader_all)
        train_loader_normal : uniquement les échantillons normaux de la condition courante
        test_loader_all     : Normal (subset) + faulty de la condition courante
    """
```

---

## Vérifications à effectuer

1. **Retourne 3 tâches** : `len(loaders) == 3`
2. **train_loader ne contient que des normaux** : `unique(y_train) == {0}`
3. **test_loader contient les deux classes** : `0 in unique(y_test) and 1 in unique(y_test)`
4. **input_dim = 13** : `X.shape[1] == 13`
5. **Reproductibilité** : deux appels avec `seed=42` donnent les mêmes loaders

Si l'implémentation Sprint 15 ne couvre pas tous ces points, adapter `pronostia_dataset.py` en conséquence.

---

## Critères d'acceptation

- [ ] `get_pronostia_dataloaders_anomaly_detection()` retourne 3 tuples valides
- [ ] `X.shape[1] == 13` pour toutes les tâches
- [ ] `train_loader` ne contient que label=0 (normaux)
- [ ] `test_loader` contient label=0 et label=1
- [ ] Interface compatible avec `scripts/run_anomaly_detection.py --dataset pronostia`

## Statut

⬜ À vérifier (Sprint 15 peut déjà couvrir ce cas)
