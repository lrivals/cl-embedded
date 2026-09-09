# S5101 — Définition du score système composite

| Champ | Valeur |
|-------|--------|
| **Sprint** | 51 |
| **Priorité** | 🟢 Bas — fonde la métrique ; S5102 (implémentation) en découle. |
| **Statut** | 📝 Doc — spec complète ; implémentation à venir |
| **Durée estimée** | 3h |
| **Dépendances** | Sprint 49 (RAM totale) · Sprint 50 (énergie) · `compute_cost.py` ✅ |
| **Fichiers cibles** | `docs/context/system_score.md` (nouveau) |
| **Références** | CR §3 « Métrique globale d'évaluation système » |

## Contexte

Le CR demande un **indicateur composite** unique. Cette tâche fige les dimensions, la normalisation et le schéma
d'agrégation, avant implémentation (S5102).

## Spec

### 1. Dimensions (CR §3)

| Dimension | Métrique | Sens |
|-----------|----------|------|
| Mémoire | RAM totale (octets) | ↓ mieux |
| Calcul | Latence d'inférence (ms) | ↓ mieux |
| Énergie | Énergie/inférence (µJ) | ↓ mieux |
| Modèle | Params / MACs | ↓ mieux |
| Performance | Accuracy / F1 / AUROC | ↑ mieux |

### 2. Normalisation (obligatoire)

Unités incomparables → normaliser par dimension (min-max sur l'ensemble des configurations comparées, ou
z-score), en orientant « ↑ = mieux » (inverser les dimensions coût). Documenter la méthode retenue.

### 3. Agrégation

`score = Σ w_d · x̃_d` (moyenne pondérée des dimensions normalisées), `Σ w_d = 1`. Pondérations **par défaut**
égales, ajustables ; `TODO(arnaud)` pour valider les poids adaptés au contexte embarqué.

### 4. Sensibilité

Fournir le classement sous ≥ 2 jeux de poids (ex. « RAM-priorité » vs « énergie-priorité ») pour montrer la
robustesse/fragilité du rang. Le classement absolu n'est jamais présenté comme unique vérité.

### 5. Règle plateforme

Le score classe **à plateforme fixée** (board, ou PC) — jamais un score PC vs un score board (CR §1).

## Format de sortie

`docs/context/system_score.md` : dimensions, normalisation, agrégation, pondérations, sensibilité, règle plateforme.

## Contraintes

- Aucun chiffre de résultat (définition pure).
- Normalisation explicite (sinon la plus grande échelle domine).
- Renvoi aux sources S49/S50/`compute_cost.py`.

## Vérification

```bash
test -f docs/context/system_score.md
grep -i "normalis\|pondération\|sensibilité\|plateforme" docs/context/system_score.md
```
