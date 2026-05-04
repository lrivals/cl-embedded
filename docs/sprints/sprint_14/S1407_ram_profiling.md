# S14-07 — RAM profiling `EWCOneClassDetector` + annotations MEM

| Champ | Valeur |
|-------|--------|
| **ID** | S14-07 |
| **Sprint** | Sprint 14 |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 1h |
| **Dépendances** | S14-01 |
| **Fichiers cibles** | `src/models/ewc/ewc_oneclass.py`, `evaluation/memory_profiler.py` |

---

## Objectif

Mesurer l'empreinte RAM réelle de `EWCOneClassDetector` pour input_dim=4 (Monitoring) via `tracemalloc`, et s'assurer que toutes les couches portent leurs annotations `# MEM:`.

---

## Commande de profiling

```bash
python scripts/profile_memory.py \
    --model ewc_oneclass \
    --dataset monitoring \
    --config configs/ewc_oneclass_config.yaml
```

Ou directement dans un script de validation :

```python
import tracemalloc
import numpy as np
from src.models.ewc.ewc_oneclass import EWCOneClassDetector

tracemalloc.start()
detector = EWCOneClassDetector(input_dim=4, hidden_dim=32, latent_dim=8)
X_normal = np.random.randn(100, 4).astype(np.float32)
detector.fit_task(X_normal)
detector.on_task_end()
scores = detector.predict_score(X_normal)
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"RAM peak : {peak / 1024:.1f} Ko")
```

---

## Résultats attendus

| Composant | RAM estimée (théorique) |
|-----------|------------------------|
| Paramètres modèle (FP32) | ≈ 2 Ko |
| Fisher matrix (même taille que params) | ≈ 2 Ko |
| θ* sauvegardé (même taille que params) | ≈ 2 Ko |
| Buffer d'activations (batch=32, input=4) | ≈ 1 Ko |
| **Total estimé** | **≈ 7 Ko** |

Contrainte : total mesuré ≤ 64 Ko (large marge attendue pour input_dim=4).

---

## Annotations MEM à vérifier dans `ewc_oneclass.py`

```python
# Encodeur
self.fc_enc1 = nn.Linear(input_dim, hidden_dim)
# MEM: input_dim*hidden_dim*4 B @ FP32 / input_dim*hidden_dim B @ INT8
self.fc_enc2 = nn.Linear(hidden_dim, latent_dim)
# MEM: hidden_dim*latent_dim*4 B @ FP32 / hidden_dim*latent_dim B @ INT8

# Décodeur
self.fc_dec1 = nn.Linear(latent_dim, hidden_dim)
# MEM: latent_dim*hidden_dim*4 B @ FP32 / latent_dim*hidden_dim B @ INT8
self.fc_dec2 = nn.Linear(hidden_dim, input_dim)
# MEM: hidden_dim*input_dim*4 B @ FP32 / hidden_dim*input_dim B @ INT8
```

---

## Critères d'acceptation

- [ ] RAM peak mesurée et documentée dans un commentaire en tête de `ewc_oneclass.py`
- [ ] RAM peak ≤ 64 Ko pour input_dim=4
- [ ] Annotations `# MEM:` présentes sur les 4 couches linéaires (FP32 et INT8)
- [ ] `get_ram_bytes()` retourne une valeur cohérente (±10%) avec tracemalloc

## Statut

⬜ À faire
