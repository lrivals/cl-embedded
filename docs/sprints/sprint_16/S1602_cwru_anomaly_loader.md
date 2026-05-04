# S16-02 — Loader `get_cwru_dataloaders_anomaly_detection()`

| Champ | Valeur |
|-------|--------|
| **ID** | S16-02 |
| **Sprint** | Sprint 16 |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 2h |
| **Dépendances** | S16-01 (scénario décidé) |
| **Fichier cible** | `src/data/cwru_dataset.py` |

---

## Objectif

Implémenter `get_cwru_dataloaders_anomaly_detection()` : wrapper sur les loaders CWRU existants qui produit des DataLoaders train_normal / test_all pour le scénario d'anomaly detection one-class.

---

## Contexte CWRU anomaly detection

**Données normales disponibles** : uniquement `Time_Normal_1_098.mat` (1 fichier = ~230 fenêtres normales)
**Données faulty** : 9 fichiers (3 Ball + 3 Inner Race + 3 Outer Race = ~2 070 fenêtres)
**Ratio** : ~10% de données normales sur l'ensemble du dataset — cas défavorable pour le one-class learning

**Défi principal** : avec ~77 données normales par tâche (230 / 3), les modèles one-class peuvent manquer de données d'entraînement. Le loader doit signaler ce problème dans les logs.

---

## Interface

```python
def get_cwru_dataloaders_anomaly_detection(
    data_path: str | Path,
    scenario: Literal["by_fault_type", "by_severity"],  # décidé en S16-01
    test_size: float = 0.2,
    batch_size: int = 32,
    seed: int = 42,
) -> list[tuple[int, str, DataLoader, DataLoader]]:
    """
    Retourne 3 tâches CWRU pour anomaly detection one-class.

    Returns
    -------
    list of (task_id, task_name, train_loader_normal, test_loader_all)
        train_loader_normal : uniquement Time_Normal (réparti équitablement entre 3 tâches)
        test_loader_all     : Normal (subset) + faulty (tâche courante) pour évaluation AUROC
    """
```

---

## Logique de split

### Répartition des données normales

Les 230 fenêtres normales sont réparties en 3 tiers :
```
Tâche 0 : Normal[0:77]
Tâche 1 : Normal[77:154]
Tâche 2 : Normal[154:230]
```

> Cette répartition est déterministe (pas de shuffle pour la consistance cross-expériences).

### Scénario by_severity (si retenu)

```
Tâche 0 ("severity_007") :
    train = Normal_tâche0 (77 échantillons)
    test  = Normal_tâche0_test + B007 + IR007 + OR007

Tâche 1 ("severity_014") :
    train = Normal_tâche1 (77 échantillons)
    test  = Normal_tâche1_test + B014 + IR014 + OR014

Tâche 2 ("severity_021") :
    train = Normal_tâche2 (77 échantillons)
    test  = Normal_tâche2_test + B021 + IR021 + OR021
```

### Scénario by_fault_type (si retenu)

```
Tâche 0 ("ball") :
    train = Normal_tâche0 (77 échantillons)
    test  = Normal_tâche0_test + B007 + B014 + B021

Tâche 1 ("inner_race") :
    train = Normal_tâche1 (77 échantillons)
    test  = Normal_tâche1_test + IR007 + IR014 + IR021

Tâche 2 ("outer_race") :
    train = Normal_tâche2 (77 échantillons)
    test  = Normal_tâche2_test + OR007 + OR014 + OR021
```

---

## Warning à émettre

```python
import warnings
if len(X_train_normal) < 100:
    warnings.warn(
        f"Tâche {task_id} : seulement {len(X_train_normal)} échantillons normaux "
        f"d'entraînement (CWRU ~20% normal). Les détecteurs one-class peuvent être instables.",
        UserWarning,
    )
```

---

## Critères d'acceptation

- [ ] Retourne 3 tuples `(task_id, task_name, train_loader, test_loader)`
- [ ] `train_loader` ne contient que des échantillons avec label=0 (Normal)
- [ ] `test_loader` contient Normal + Faulty (vérifiable avec `unique(y)`)
- [ ] `X.shape[1] == 9` pour toutes les tâches
- [ ] Warning émis si < 100 échantillons normaux par tâche
- [ ] Les deux scénarios (by_fault_type, by_severity) fonctionnent

## Statut

⬜ En attente S16-01
