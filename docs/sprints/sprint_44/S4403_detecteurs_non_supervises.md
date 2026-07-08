# S4403 — Détecteurs de drift non-supervisés (ADWIN, KSWIN, KS, MMD, PSI/JS)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 44 |
| **Priorité** | 🔴 Critique — famille non-supervisée de l'axe « supervisé ∥ non-supervisé à parité » ; c'est celle réaliste pour une carte déployée seule (pas de labels). |
| **Statut** | 📝 Doc — spec ; implémentation à venir. |
| **Durée estimée** | 8h |
| **Dépendances** | S4401 ✅ (interface `BaseDriftDetector`) · S4402 ✅ (`base.py`) · `firmware/.../inc/ring_buffer.h` ✅ (miroir de fenêtre bornée pour cohérence board) |
| **Fichiers cibles** | `src/models/drift/adwin.py`, `src/models/drift/kswin.py`, `src/models/drift/ks_test.py`, `src/models/drift/mmd.py`, `src/models/drift/psi.py` |
| **Références** | Bifet & Gavaldà 2007 (ADWIN) · Raab 2020 (KSWIN) · Gretton 2012 (MMD) · PSI (credit scoring standard) · `src/evaluation/drift_detector.py` (baseline fenêtre) |

---

## Contexte

Les détecteurs non-supervisés surveillent la **distribution des features (ou d'un score)** sans jamais
voir de label — le cas réaliste d'une carte déployée seule sur une machine neuve (scénario S38). Ils sont
plus coûteux en mémoire (fenêtre O(W), parfois tri) → leur **viabilité MCU** est le point à trancher pour
S45. Cette tâche les implémente derrière l'interface commune, en gardant l'**état borné** en vue du
portage.

## Spec

Tous héritent de `BaseDriftDetector` (S4402), `requires_label = False`. Entrée = une **feature scalaire**
(ou score) par appel `update(x)`, ou une paire de fenêtres pour les tests deux-échantillons.

- **ADWIN** (`adwin.py`) : fenêtre adaptative ; maintient des **buckets d'histogrammes exponentiels**
  (moyenne + variance) ; coupe la fenêtre si deux sous-fenêtres diffèrent au-delà de la borne de
  Hoeffding `ε_cut` (paramètre `delta`). `DRIFT` au moment de la coupe. État O(log W)
  (`# MEM: ~M·bucket_bytes`). Prévoir une **borne de buckets** (config) pour un état majoré → argument MCU.
- **KSWIN** (`kswin.py`) : réservoir de taille `W` + fenêtre récente de taille `r` ; test
  Kolmogorov-Smirnov entre les deux ; `DRIFT` si `stat > seuil(α)`. État O(W) + tri (`# MEM: W·4 B`).
- **KS glissant** (`ks_test.py`) : `ks_2samp(ref_window, cur_window)` évalué tous les `stride`
  échantillons ; `ref_window` = enrôlement figé, `cur_window` = ring buffer courant. État O(W).
- **MMD** (`mmd.py`) : distance MMD² à noyau RBF entre `ref_window` et `cur_window` ; **variante linéaire**
  (estimateur non biaisé O(W)) privilégiée à la forme quadratique O(W²) pour la viabilité MCU ; seuil par
  permutation (PC) ou percentile d'enrôlement (portable). État O(W).
- **PSI / Jensen-Shannon** (`psi.py`) : histogrammes à **bacs fixes** calibrés sur l'enrôlement ; PSI =
  `Σ(p_cur−p_ref)·ln(p_cur/p_ref)`, JS = divergence de Jensen-Shannon ; `DRIFT` si `> seuil` (PSI>0.2
  standard). État **O(bins)** (indépendant de W) → **le plus MCU-friendly** des non-supervisés.

Multivarié : par défaut, appliquer par **feature** puis agréger (max ou fraction de features en drift,
paramètre config), sauf MMD (nativement multivarié). La feature/agrégation suivie ← config.

### Fenêtre bornée (cohérence board)

Les fenêtres utilisent une structure **à capacité fixe** (miroir de `ring_buffer.h`, S45) — pas de
`deque` non bornée — pour que l'empreinte soit **majorée et identique PC↔board**. `get_state_bytes()`
reflète la capacité réelle.

## Contraintes

- **État borné** partout (capacité de fenêtre/buckets/bins en config) — pas d'accumulation illimitée.
- **Annotations `# MEM:`** sur chaque structure.
- Calibration des seuils sur **segment d'enrôlement** (cohérent S38 : percentile) pour la portabilité —
  éviter les seuils par permutation coûteux côté board (les réserver au diagnostic PC).
- Ne pas dupliquer la baseline `SlidingWindowDriftDetector` — elle est catalToguée à part.

## Vérification

```bash
pytest tests/test_drift_detectors.py -k "adwin or kswin or ks_test or mmd or psi" -v
python -c "from src.models.drift.psi import PSI; d=PSI(bins=10); print(d.get_state_bytes())"
```
- Sur un flux à drift franc (feature moyenne 0 → 3 à t connu), chaque détecteur signale `DRIFT` après le
  changement ; PSI/JS montrent un état **indépendant de W** (O(bins)).
- ADWIN/KSWIN retrouvent le point de drift sur le **synthétique** S43 (points exacts) dans une tolérance
  documentée.
