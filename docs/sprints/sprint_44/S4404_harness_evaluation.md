# S4404 — Harnais de métriques d'évaluation du drift (+ RAM/latence)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 44 |
| **Priorité** | 🔴 Critique — sans métriques dédiées, « détecter le drift » n'est pas quantifiable ni justifiable. |
| **Statut** | ✅ Implémenté — `src/evaluation/drift_metrics.py` + `tests/test_drift_metrics.py` **12 PASS**. |
| **Durée estimée** | 6h |
| **Dépendances** | S4302 ✅ (ground-truth `drift_points`) · S4401 ✅ (interface détecteur) · `src/evaluation/memory_profiler.py` ✅ (RAM/latence proxies) · `src/evaluation/metrics.py` ✅ (pattern `save_metrics`) |
| **Fichiers cibles** | `src/evaluation/drift_metrics.py` |
| **Références** | `src/evaluation/anomaly_metrics.py` (gabarit `compute_*`/`save_*`) · MTFA/MTD (théorie des changements de régime) |

---

## Contexte

Un détecteur de drift ne s'évalue pas comme un classifieur : ce qui compte est **à quelle vitesse** il
signale un vrai drift (délai), **combien de fausses alarmes** il produit sur les segments stables, et
**combien de drifts** il manque. Ce sprint évalue aussi le **coût** (RAM/latence) car la finalité est le
portage MCU. Cette tâche fournit le harnais unique consommé par S4405/S4406.

## Spec

`src/evaluation/drift_metrics.py` — fonctions pures (miroir de `anomaly_metrics.py`) :

### 1. Métriques de détection (vs ground-truth `drift_points`)

`compute_drift_metrics(verdicts, drift_points, n_samples, tolerance) -> dict` :
- **Délai de détection moyen** (`mean_detection_delay`) : distance moyenne entre chaque vrai point de
  drift et la première alarme suivante (dans la fenêtre de tolérance).
- **Taux de fausses alarmes** (`false_alarm_rate`, FAR) : alarmes hors de toute fenêtre de tolérance /
  nombre d'échantillons stables.
- **Taux de manqués** (`missed_detection_rate`, MDR) : vrais points sans alarme dans la tolérance.
- **Précision / rappel / F1 des alarmes** vs points de drift (appariement dans la tolérance).
- **MTFA** (Mean Time between False Alarms) et **MTD** (Mean Time to Detection) — indicateurs classiques
  de compromis réactivité/robustesse.
- Gestion honnête : si `drift_points is None` (Electricity/NOAA), seuls FAR/MTFA sur segments réputés
  stables sont calculés ; délai/MDR → `null`.

### 2. Métriques de coût (proxies PC honnêtes)

`profile_drift_detector(detector, stream) -> dict` : réutilise `memory_profiler.py` pour
- `state_bytes` (via `get_state_bytes()` du détecteur — empreinte algorithmique, pas l'allocateur Python),
- `ram_peak_bytes` (tracemalloc — proxy PC, **étiqueté proxy**),
- `latency_us_per_update` (moyenne/écart-type sur N updates),
- `requires_label`.
Ces chiffres **préparent** S45 (mesure board réelle) — la distinction proxy-PC vs mesuré-board est
explicite (règle héritée).

### 3. Table comparative

`build_comparison_table(results_by_detector) -> dict` : une ligne par détecteur × dataset avec
{délai, FAR, MDR, F1, MTFA, MTD, state_bytes, latency, requires_label, viabilité_MCU}. Sérialisable JSON.

## Contraintes

- **Fonctions pures**, testables sans détecteur réel (verdicts synthétiques + points connus).
- Aucun chiffre en dur ; `null` pour non calculable/non mesuré (honnêteté).
- RAM/latence PC explicitement **proxies** (docstring + clé `_proxy: true`) — les vrais chiffres viennent
  de S45 (DWT/`.bss`).
- Réutiliser `memory_profiler.py` — ne pas réimplémenter le profilage.

## Vérification

```bash
pytest tests/test_drift_metrics.py -v
```
- Cas synthétique : un détecteur « oracle » (alarme exactement aux `drift_points`) → délai=0, FAR=0,
  MDR=0, F1=1.0.
- Cas « paresseux » (jamais d'alarme) → MDR=1.0, FAR=0, délai=`null`.
- Cas « paranoïaque » (alarme partout) → FAR élevé, délai≈0, précision faible.
- `state_bytes` cohérent avec l'annotation `# MEM:` du détecteur.
