# S4401 — Inventaire de référence des détecteurs de drift + config

| Champ | Valeur |
|-------|--------|
| **Sprint** | 44 |
| **Priorité** | 🔴 Critique — fige la taxonomie, l'interface commune et la config ; conditionne S4402–S4406. |
| **Statut** | 📝 Doc — spec ; implémentation à venir. |
| **Durée estimée** | 4h |
| **Dépendances** | Sprint 43 ✅ (datasets) · `src/evaluation/drift_detector.py` ✅ (baseline à cataloguer) |
| **Fichiers cibles** | `docs/context/drift_detectors.md` (source de vérité), `configs/sprint44_drift_detection.yaml` |
| **Références** | Sprint 42 S4202 (`quantization_strategies.md` — gabarit d'inventaire de référence) · Gama et al. 2014 · Bifet & Gavaldà 2007 (ADWIN) |

---

## Contexte

Avant d'implémenter, on **catalogue** les détecteurs retenus : formule, état mémoire, besoin de labels,
et **viabilité MCU** (état borné → portable board S45). Ce document est la **source de vérité textuelle**
des slides/manuscrit sur la détection de drift, miroir de `quantization_strategies.md` (S4202).

## Spec

Produire `docs/context/drift_detectors.md` avec, **par détecteur**, une fiche :

- **Famille** : statistique supervisé / statistique non-supervisé / test deux-échantillons / baseline.
- **Signal d'entrée** : flux d'erreur (0/1, labels requis) vs feature scalaire vs fenêtre de features vs
  score anomaly.
- **Principe & formule** : condition de déclenchement (warning/drift).
- **État mémoire** (annotation `# MEM:`) : O(1) / O(W) / O(log W) ; octets pour W typique.
- **Hyperparamètres** : seuils, tailles de fenêtre, δ/λ, etc. (tous en config).
- **Viabilité MCU** : ✅/🟡/⚠️ + justification (état borné, opérations, tri/allocation).
- **Réf. bibliographique**.

### Détecteurs catalogués

**Supervisés (flux d'erreur) — S4402 :**
- **DDM** (Drift Detection Method, Gama 2004) : suit `p_error + s`, warning à `p+s > p_min+2·s_min`,
  drift à `> p_min+3·s_min`. État O(1). ✅ MCU.
- **EDDM** (Early DDM, Baena-García 2006) : suit la distance entre erreurs (mieux sur drift graduel).
  État O(1). ✅ MCU.
- **Page-Hinkley** (test séquentiel CUSUM) : cumul `m_T = Σ(x_t − x̄_t − δ)`, drift si `m_T − min > λ`.
  État O(1). ✅ MCU excellente.

**Non-supervisés (features/score) — S4403 :**
- **ADWIN** (Adaptive Windowing, Bifet 2007) : fenêtre adaptative, coupe si deux sous-fenêtres diffèrent
  (borne de Hoeffding). Histogrammes exponentiels → O(log W). 🟡 MCU (variante à état fixe à étudier).
- **KSWIN** (Kolmogorov-Smirnov Windowing, Raab 2020) : KS entre fenêtre récente et réservoir. O(W) +
  tri. 🟡 MCU.
- **KS glissant deux-échantillons** : `ks_2samp(ref_window, cur_window)` périodique. O(W) + tri. ⚠️ MCU.
- **MMD** (Maximum Mean Discrepancy, noyau RBF) : distance entre fenêtres. O(W²) naïf / O(W) linéaire.
  ⚠️ MCU (variante linéaire requise).
- **PSI / Jensen-Shannon** sur histogrammes à bacs fixes : O(bins). ✅ MCU (histogramme borné).

**Baseline projet :**
- **`SlidingWindowDriftDetector`** (Sprint 9/38) : double seuil sur score Maha + persistance. O(W).
  ✅ déjà porté C (`drift_detector.c`).

### Interface commune (figée ici, implémentée S4402/S4403)

`src/models/drift/base.py::BaseDriftDetector` :
- `update(value) -> DriftVerdict` (`NORMAL`/`WARNING`/`DRIFT`) — un échantillon (erreur **ou** feature).
- `set_params_from_reference(reference_values)` — calibration (seuils/percentiles) sur segment d'enrôlement.
- `reset()` ; `get_state_bytes() -> int` (empreinte pour profilage) ; `requires_label -> bool`.

Verdict à **3 niveaux** (`WARNING` ajouté vs le `DriftVerdict` firmware binaire NORMAL/FAULT/DRIFT) pour
coller à DDM/EDDM ; mappé au binaire à l'export board (S45).

### Config — `configs/sprint44_drift_detection.yaml`

Tous les hyperparamètres en **constantes nommées** : par détecteur (seuils, fenêtres, δ/λ, bins), liste
des datasets, méthode de calibration (percentile enrôlement vs valeurs littérature), seed 42.

## Contraintes

- Aucune valeur hardcodée hors config.
- La fiche **viabilité MCU** doit être argumentée (c'est elle qui décide la sélection S45).
- Ne pas dupliquer `SlidingWindowDriftDetector` — le **cataloguer** comme baseline et le **réutiliser**.

## Critères d'acceptation

1. `docs/context/drift_detectors.md` couvre les ≥ 8 détecteurs + la baseline, chacun avec état mémoire et
   viabilité MCU argumentée.
2. `src/models/drift/base.py::BaseDriftDetector` (interface) est spécifié sans ambiguïté (S4402/S4403 s'y
   conforment).
3. `configs/sprint44_drift_detection.yaml` liste tous les hyperparamètres — aucun dans le code.
4. La classification supervisé/non-supervisé est explicite (`requires_label`).
