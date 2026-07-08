# S4402 — Détecteurs de drift supervisés (DDM, EDDM, Page-Hinkley)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 44 |
| **Priorité** | 🔴 Critique — famille supervisée de l'axe « supervisé ∥ non-supervisé à parité ». |
| **Statut** | 📝 Doc — spec ; implémentation à venir. |
| **Durée estimée** | 6h |
| **Dépendances** | S4401 ✅ (interface `BaseDriftDetector`, config) · `src/models/ewc/ewc_mlp.py` ✅ ou `src/models/unsupervised/mahalanobis_detector.py` ✅ (source du flux d'erreur) |
| **Fichiers cibles** | `src/models/drift/base.py`, `src/models/drift/ddm.py`, `src/models/drift/eddm.py`, `src/models/drift/page_hinkley.py` |
| **Références** | Gama 2004 (DDM) · Baena-García 2006 (EDDM) · Page 1954 (Page-Hinkley/CUSUM) · `src/evaluation/drift_detector.py` (verdict enum précédent) |

---

## Contexte

Les détecteurs supervisés surveillent le **flux d'erreur** d'un modèle prédictif (0 = correct, 1 =
erreur) : quand le taux d'erreur croît significativement, la distribution a changé. Ils sont **très
portables MCU** (état O(1)) mais exigent un **retour de vérité** (label) — ce qui, sur une carte déployée
seule, correspond au scénario « active learning » (P2 de Sprint 38). Cette tâche les implémente et fixe
la source du flux d'erreur.

## Spec

### 1. Interface commune — `src/models/drift/base.py`

`BaseDriftDetector` (défini en S4401) : `update(value) -> DriftVerdict`, `set_params_from_reference`,
`reset`, `get_state_bytes`, `requires_label`. `DriftVerdict = Enum(NORMAL, WARNING, DRIFT)`.

### 2. Source du flux d'erreur

Le flux d'erreur `e_t = 1[ŷ_t ≠ y_t]` provient d'un **modèle de faute existant** (décision `TODO(arnaud)`
S4400) : EWC (tête binaire) ou Mahalanobis seuillé. Un helper `error_stream(model, X, y) -> np.ndarray`
produit la séquence à partir des prédictions échantillon par échantillon (réutilise l'inférence
existante, ne réimplémente pas le modèle).

### 3. Détecteurs

- **DDM** (`ddm.py`) : suit `p_t` (taux d'erreur en ligne) et `s_t = sqrt(p_t(1−p_t)/t)` ; mémorise
  `p_min`, `s_min` ; `WARNING` si `p_t+s_t ≥ p_min+2·s_min`, `DRIFT` si `≥ p_min+3·s_min` ; reset des
  minima au drift. État O(1) (`# MEM: ~24 B @ FP32`).
- **EDDM** (`eddm.py`) : suit la **distance moyenne entre erreurs** `p'_t` et son écart-type `s'_t` ;
  mémorise `p'_max`, `s'_max` ; seuils `α=0.95` (warning) / `β=0.90` (drift) sur
  `(p'_t+2·s'_t)/(p'_max+2·s'_max)`. Meilleur sur drift **graduel**. État O(1).
- **Page-Hinkley** (`page_hinkley.py`) : moyenne courante `x̄_t`, cumul `m_T = Σ(x_t − x̄_t − δ)`,
  `min_T = min m_t` ; `DRIFT` si `m_T − min_T > λ`. Paramètres `δ` (tolérance) et `λ` (seuil) ← config.
  État O(1). Applicable au flux d'erreur **ou** à une feature scalaire (frontière avec S4403).

### 4. Paramètres

Tous ← `configs/sprint44_drift_detection.yaml` (seuils DDM 2σ/3σ, EDDM α/β, Page-Hinkley δ/λ). Aucun
seuil dans le code.

## Contraintes

- **Annotations `# MEM:`** obligatoires (état borné — argument de portabilité S45).
- État strictement O(1) (pas de fenêtre) → différenciateur MCU vs S4403.
- `requires_label = True` pour les trois.
- Parité de comportement avec les définitions de référence (littérature / `river`) vérifiable en test
  (S4406) sur une séquence d'erreur connue.

## Vérification

```bash
pytest tests/test_drift_detectors.py -k "ddm or eddm or page_hinkley" -v
python -c "from src.models.drift.page_hinkley import PageHinkley; d=PageHinkley(delta=0.005,lambda_=50); print(d.get_state_bytes())"
```
- Sur une séquence d'erreur à saut connu (ex. 0.1 → 0.5 à t=1000), les trois signalent `DRIFT` **après**
  le saut, avec un délai cohérent (Page-Hinkley ≤ DDM sur drift soudain).
- `get_state_bytes()` constant (indépendant du nombre d'échantillons vus) → confirme O(1).
