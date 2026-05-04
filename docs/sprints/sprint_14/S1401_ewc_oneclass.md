# S14-01 — Implémenter `EWCOneClassDetector`

| Champ | Valeur |
|-------|--------|
| **ID** | S14-01 |
| **Sprint** | Sprint 14 |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 4h |
| **Dépendances** | — |
| **Fichier cible** | `src/models/ewc/ewc_oneclass.py` |

---

## Objectif

Implémenter `EWCOneClassDetector` : un autoencoder MLP entraîné uniquement sur données normales, avec régularisation EWC appliquée à la loss de reconstruction MSE pour limiter l'oubli catastrophique entre tâches. Ce modèle expose l'API `fit_task` / `predict_score` / `on_task_end` compatible avec `run_anomaly_detection_scenario()`.

---

## Architecture

```
Entrée : x ∈ R^{input_dim}
    │
    ▼
Encodeur :
  fc_enc1 : input_dim → hidden_dim   (ReLU)  # MEM: hidden_dim*4 B @ FP32
  fc_enc2 : hidden_dim → latent_dim  (ReLU)  # MEM: latent_dim*4 B @ FP32
    │
    ▼
Décodeur :
  fc_dec1 : latent_dim → hidden_dim  (ReLU)  # MEM: hidden_dim*4 B @ FP32
  fc_dec2 : hidden_dim → input_dim   (—)     # MEM: input_dim*4 B @ FP32
    │
    ▼
Sortie : x_hat ∈ R^{input_dim}

Loss = MSE(x, x_hat) + lambda_ewc * Σ_i F_i * (θ_i - θ_i*)²
Score d'anomalie = MSE(x, x_hat)  [scalaire par échantillon]
Seuil = percentile(threshold_percentile, MSE_train_normal)
```

**Paramètres par défaut (Monitoring, input_dim=4)** :

| Paramètre | Valeur | Empreinte RAM |
|-----------|--------|---------------|
| `input_dim` | 4 | — |
| `hidden_dim` | 32 | 128 B @ FP32 |
| `latent_dim` | 8 | 32 B @ FP32 |
| `lambda_ewc` | 400 | — |
| `threshold_percentile` | 95 | — |
| `n_epochs` | 20 | — |
| `lr` | 1e-3 | — |

RAM totale estimée (modèle FP32) : ≈ 2 Ko (très inférieur à 64 Ko).

---

## Interface

```python
class EWCOneClassDetector:
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        latent_dim: int = 8,
        lambda_ewc: float = 400.0,
        threshold_percentile: float = 95.0,
        n_epochs: int = 20,
        lr: float = 1e-3,
        device: str = "cpu",
    ) -> None: ...

    def fit_task(self, X_normal: np.ndarray) -> None:
        """Entraîne l'autoencoder sur données normales de la tâche courante.
        Applique la régularisation EWC si fisher_ et params_star_ sont déjà calculés.
        Calcule le seuil threshold_ = percentile(threshold_percentile, MSE_train).
        """

    def on_task_end(self) -> None:
        """Calcule la matrice de Fisher (diagonale, empirique) et sauvegarde θ*.
        À appeler après fit_task(), avant la tâche suivante.
        """

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """Retourne le MSE de reconstruction par échantillon. Shape : (n_samples,)."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Retourne 1 (anomalie) si score > threshold_, 0 sinon."""

    def get_ram_bytes(self) -> int:
        """Retourne l'empreinte mémoire du modèle en octets (paramètres FP32)."""
```

---

## Implémentation EWC

La pénalité EWC est calculée sur **tous les paramètres** de l'autoencoder (encodeur + décodeur) :

```python
# Fisher diagonale empirique — calculé sur X_normal après entraînement
for x in X_normal_loader:
    loss = mse_loss(model(x), x)
    loss.backward()
    for name, p in model.named_parameters():
        fisher[name] += p.grad.data ** 2
fisher = {k: v / len(X_normal_loader) for k, v in fisher.items()}

# Pénalité lors de l'entraînement sur tâche t+1
ewc_penalty = sum(
    (fisher[n] * (p - params_star[n]) ** 2).sum()
    for n, p in model.named_parameters()
)
total_loss = mse_loss(x_hat, x) + lambda_ewc * ewc_penalty
```

---

## Annotations RAM obligatoires

Chaque couche doit porter son annotation `# MEM:` (convention projet) :

```python
self.fc_enc1 = nn.Linear(input_dim, hidden_dim)   # MEM: input_dim*hidden_dim*4 B @ FP32
self.fc_enc2 = nn.Linear(hidden_dim, latent_dim)  # MEM: hidden_dim*latent_dim*4 B @ FP32
self.fc_dec1 = nn.Linear(latent_dim, hidden_dim)  # MEM: latent_dim*hidden_dim*4 B @ FP32
self.fc_dec2 = nn.Linear(hidden_dim, input_dim)   # MEM: hidden_dim*input_dim*4 B @ FP32
```

---

## Critères d'acceptation

- [ ] `EWCOneClassDetector` importable depuis `src.models.ewc.ewc_oneclass`
- [ ] `fit_task(X_normal)` entraîne sans erreur sur `X.shape == (N, 4)` (Monitoring)
- [ ] `on_task_end()` peuple `self.fisher_` et `self.params_star_` correctement
- [ ] `predict_score(X)` retourne `np.ndarray` de shape `(N,)` avec valeurs ≥ 0
- [ ] `predict(X)` retourne `np.ndarray` de shape `(N,)` avec valeurs ∈ {0, 1}
- [ ] La pénalité EWC est non nulle à la tâche 2+ (vérifiable via `loss_components`)
- [ ] Annotations `# MEM:` présentes sur les 4 couches linéaires
- [ ] `get_ram_bytes()` retourne une valeur cohérente avec les paramètres (≤ 64 Ko pour input_dim=4)

## Statut

⬜ À faire
