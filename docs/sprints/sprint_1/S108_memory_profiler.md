# S1-08 — Implémenter `memory_profiler.py` (tracemalloc)

| Champ | Valeur |
|-------|--------|
| **ID** | S1-08 |
| **Sprint** | Sprint 1 — Semaine 1 (15–22 avril 2026) |
| **Priorité** | 🟡 Normal |
| **Durée estimée** | 2h |
| **Dépendances** | S1-04 (`EWCMlpClassifier`) |
| **Fichier cible** | `src/evaluation/memory_profiler.py` |

---

## Objectif

Implémenter un profiler de mémoire RAM basé sur `tracemalloc` pour mesurer les chiffres de RAM nécessaires à la validation du **Gap 2** (opération sub-100 Ko avec chiffres précis mesurés).

Deux phases doivent être profilées séparément :
1. **Forward pass** (inférence seule, mode `eval()`) — proxy de la RAM MCU à l'inférence
2. **CL update** (forward + backward + optimizer step) — RAM nécessaire à la mise à jour incrémentale

> **Note méthodologique** : `tracemalloc` mesure l'allocateur Python/PyTorch, pas la RAM C native. Ces chiffres sont des **proxies PC**. Les mesures MCU réelles seront effectuées en Phase 2 (portage STM32N6). Les deux mesures doivent être reportées distinctement dans le manuscrit.

**Critère de succès** : `full_memory_report(model, (1, 6))` affiche les chiffres RAM et confirme `within_budget_64ko = True` pour `EWCMlpClassifier`.

> **Statut** : ✅ **Implémenté** — `src/evaluation/memory_profiler.py` présent et fonctionnel.

---

## Interface implémentée

### `profile_forward_pass(model, input_shape, n_runs=100, device="cpu")`

```python
def profile_forward_pass(
    model: nn.Module,
    input_shape: tuple,   # ex. (1, 6) pour un seul exemple
    n_runs: int = 100,
    device: str = "cpu",
) -> dict:
    """
    Retourne :
        ram_peak_bytes           : int — pic RAM Python (tracemalloc)
        ram_current_bytes        : int — RAM courante après le forward
        inference_latency_ms     : float — latence moyenne sur n_runs
        inference_latency_std_ms : float — écart-type latence
        n_params                 : int
        params_fp32_bytes        : int — estimation poids FP32
        params_int8_bytes        : int — estimation poids INT8
        within_budget_64ko       : bool — ram_peak_bytes < 65 536
    """
```

### `profile_cl_update(update_fn, input_shape, label_shape, n_runs=50, device="cpu")`

```python
def profile_cl_update(
    update_fn: Callable[[torch.Tensor, torch.Tensor], float],
    # update_fn encapsule : optimizer.zero_grad() + loss.backward() + optimizer.step()
    input_shape: tuple,
    label_shape: tuple = (1, 1),
    n_runs: int = 50,
    device: str = "cpu",
) -> dict:
    """
    Retourne :
        ram_peak_bytes_update      : int — pic RAM pendant la mise à jour
        update_latency_ms          : float — latence moyenne d'une mise à jour
        update_latency_std_ms      : float
        within_budget_64ko_update  : bool
    """
```

### `full_memory_report(model, input_shape, update_fn=None, model_name="Model")`

Génère un rapport consolidé (forward + update optionnel) et l'affiche sur stdout :

```
🔍 Profiling mémoire : EWCMlpClassifier
   Input shape : (1, 6)

   ┌─ Résultats ─────────────────────────────────────┐
   │  Paramètres    :        769 params
   │  RAM FP32      :      3,076 B (3.0 Ko)
   │  RAM INT8 est. :        769 B (0.8 Ko)
   │  RAM peak fwd  :      X,XXX B (X.X Ko) — X.X% budget
   │  Latence fwd   :      X.XXX ms (± X.XXX)
   │  STM32N6 64Ko  : ✅ DANS LE BUDGET
   └────────────────────────────────────────────────┘
```

### `compare_models_memory(reports)`

Tableau comparatif multi-modèles (utilisé en S2 pour EWC vs HDC).

---

## Critères d'acceptation

- [x] `from src.evaluation.memory_profiler import profile_forward_pass, full_memory_report` — aucune erreur d'import
- [x] `profile_forward_pass(EWCMlpClassifier(), (1, 6))` retourne `within_budget_64ko = True`
- [x] `full_memory_report()` affiche le tableau formaté sur stdout
- [x] `ram_peak_bytes` mesurable (> 0) pour un forward pass réel
- [x] `compare_models_memory()` retourne une string formatée sans erreur
- [x] `ruff check src/evaluation/memory_profiler.py` et `black --check` passent

---

## Sorties attendues à reporter dans le manuscrit

| Mesure | Valeur attendue (théorique) | Où reporter |
|--------|----------------------------|-------------|
| `n_params` EWC MLP | 769 | Table Gap 2, manuscrit §4.3 |
| `params_fp32_bytes` | 3 076 B (3 Ko) | Table Gap 2 |
| RAM EWC totale (modèle + Fisher + θ*) | ~9 Ko | Table Gap 2 |
| `inference_latency_ms` | < 100 ms attendu | Table Gap 2 |
| `within_budget_64ko` | `True` | Conclusion Gap 2 |

> **FIXME(gap2)** : distinguer `ram_peak_bytes` PC-side (tracemalloc) des mesures MCU réelles à venir en Phase 2.

---

## Questions ouvertes

- `TODO(dorra)` : les mesures tracemalloc PC sous-estiment-elles ou surestiment-elles la RAM MCU ? Facteur de correction empirique à valider sur STM32N6.
- `TODO(arnaud)` : inclure le profiling du calcul Fisher (`update_ewc_state`) dans S1-09 ou reporter à Phase 2 ?
