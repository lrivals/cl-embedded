# Skill : Debug et Revue de Code

> **Usage** : Demander à Claude d'analyser une erreur, de réviser du code, ou de vérifier la conformité avec les contraintes du projet.  
> **Déclencheur** : "debug [erreur]" / "révise ce code" / "est-ce que ce code est compatible MCU ?"

---

## Checklist de revue de code (CL-Embedded)

Quand Claude révise du code, vérifier systématiquement :

### Conformité MCU
```
[ ] Pas d'Adam / AdamW (→ SGD uniquement)
[ ] Pas de BatchNorm / LayerNorm en ligne
[ ] Pas d'activations non-ReLU (GELU, SiLU...)
[ ] Pas d'allocations dynamiques dans le forward
[ ] Annotations # MEM: présentes sur chaque couche
[ ] Tailles constantes (depuis configs YAML, pas hardcodées)
[ ] torch.no_grad() autour du backbone gelé
```

### Conformité projet
```
[ ] Type hints sur toutes les fonctions publiques
[ ] Docstrings format NumPy
[ ] Config depuis YAML (pas d'hyperparamètres dans le code)
[ ] Sauvegarde checkpoints avec format standardisé
[ ] Métriques RAM mesurées (tracemalloc) et loggées
[ ] Seed fixé via set_seed()
[ ] Résultats sauvegardés dans experiments/ (pas dans src/)
```

### Reproductibilité
```
[ ] set_seed() appelé avant tout entraînement
[ ] Config snapshot sauvegardé dans experiments/exp_XXX/
[ ] Chemins relatifs (Path) jamais absolus (/home/user/...)
[ ] Données jamais committées (.gitignore)
```

---

## Erreurs fréquentes et corrections

### Erreur : gradient dans le backbone
```python
# ❌ Problème : gradient se propage dans le backbone
features = backbone(x)
loss = criterion(head(features), y)
loss.backward()  # gradient remonte jusqu'au backbone

# ✅ Correction
with torch.no_grad():
    features = backbone(x)
loss = criterion(head(features), y)
loss.backward()  # gradient s'arrête à head
```

### Erreur : normalisation en ligne
```python
# ❌ Problème : stats calculées sur le batch courant
def preprocess(x):
    return (x - x.mean()) / x.std()  # varie à chaque batch !

# ✅ Correction : stats offline depuis config
def preprocess(x, mean, std):
    return (x - mean) / std  # reproductible, portable MCU
```

### Erreur : hyperparamètre hardcodé
```python
# ❌ Problème
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# ✅ Correction
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=cfg["training"]["learning_rate"]
)
```

### Erreur : résultats sauvegardés dans le mauvais répertoire
```python
# ❌ Problème
torch.save(results, "results.json")

# ✅ Correction
exp_dir = Path(f"experiments/{exp_id}/results/")
exp_dir.mkdir(parents=True, exist_ok=True)
with open(exp_dir / "metrics.json", "w") as f:
    json.dump(results, f, indent=2)
```

---

## Format de rapport de revue

Quand Claude rapporte une revue de code :

```
REVUE : [nom_fichier.py]

✅ Points conformes :
  - [liste des bonnes pratiques observées]

⚠️ Points à corriger (non-bloquants) :
  - Ligne X : [description] → [correction suggérée]

❌ Points bloquants (incompatibilité MCU ou projet) :
  - Ligne Y : [description] → [correction obligatoire]

💡 Suggestions d'amélioration :
  - [optionnel]

Conformité MCU : [✅ Conforme / ⚠️ Partiellement / ❌ Non conforme]
Conformité projet : [✅ Conforme / ⚠️ Partiellement / ❌ Non conforme]
```
