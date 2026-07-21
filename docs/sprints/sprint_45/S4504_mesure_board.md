# S4504 — Mesure board réelle (RAM `.bss` + latence DWT) & agrégat

| Champ | Valeur |
|-------|--------|
| **Sprint** | 45 |
| **Priorité** | 🔴 Critique — c'est le résultat *système* du sprint : coût réel des détecteurs sur MCU. |
| **Statut** | ✅ Implémenté — agrégat + colonne `gas_sensor_drift` mesurée board réelle (3 détecteurs : 2 mesurés, PSI overflow honnête). |
| **Durée estimée** | 6h |
| **Dépendances** | S4503 ✅ (driver + parité) · `firmware/.../src/profiling.c` ✅ (DWT + `.bss`) · `make size` ✅ · `scripts/aggregate_sprint38.py` ✅ (gabarit agrégat) |
| **Fichiers cibles** | `scripts/aggregate_sprint45.py`, `experiments/exp_S45_board_{detector}_{dataset}/`, `experiments/exp_S45_summary.json` |
| **Références** | `docs/context/ram_measurement.md` (méthodo `.bss` + stack watermark) · `docs/triple_gap.md` (§ Gap 2/Gap 3) · Sprint 29 S2913 (grille board) |

---

## Contexte

Le S44 a donné des **proxies PC** (tracemalloc, latence Python). Ce sprint les remplace par les **chiffres
réels NUCLEO-F439ZI** : latence par update (DWT, `profiling_start/stop`), empreinte `.bss` (`make size`
sur le `.elf`), et vérifie les gaps (Gap 2 latence < 100 ms, Gap 3 RAM dans le budget). C'est la
contribution mesurée : *combien coûte réellement la détection de drift sur MCU, par méthode*.

## Spec

### 1. Collecte board (via S4503)

Pour chaque cellule `(détecteur, dataset)` streamée : latences DWT par échantillon → **P50/P99**, `.bss`
du binaire (`make size`), état effectif du détecteur (`get_state_bytes` C vs annotation `# MEM:`), 0 CRC.
Grille = détecteurs retenus (S4501) × datasets S43.

### 2. Agrégat — `scripts/aggregate_sprint45.py`

Lecture seule → `experiments/exp_S45_summary.json`, indexé `[detector][dataset][platform]` :
- `latency_us` P50/P99 (board réel) **vs** `latency_us_proxy` (PC, S44) — écart documenté.
- `bss_bytes` (board réel) **vs** `state_bytes` (algorithmique) — coût firmware total vs état pur.
- `verdict_parity` (rappel S4503), `gap2_ok` (< 100 ms), `gap3_ram_ok` (dans budget).
- **Table coût board** : delta `.bss` par méthode vs build par défaut (invariant) + latence.
- N/A honnête (`null` + `na_reason`) partout où non mesuré.

### 3. Vérifications de gap

- **Gap 2** : toutes les latences ≪ 100 ms (attendu : Page-Hinkley/DDM O(1) ~ quelques µs ; PSI O(bins)
  faible ; fenêtres O(W) plus élevé mais borné) — DWT réel.
- **Gap 3** : `.bss` de chaque méthode dans le budget 256 Ko, delta vs défaut documenté (Page-Hinkley/DDM
  quelques dizaines d'octets ; PSI ~ `2·bins·2 B` ; ADWIN majoré par borne de buckets).

## Contraintes

- **Aucun chiffre inventé** : tout depuis le board via `run_sprint45_board.py` ; `« à mesurer »` tant que
  non flashé.
- `.bss` build **par défaut invariant** (0 régression) — condition de recevabilité.
- Distinguer **mesuré-board** (ici) de **proxy-PC** (S44) dans chaque champ.
- Latence rapportée par **update de détecteur** (pas l'inférence complète) pour l'imputer correctement.

## Vérification

```bash
# après flash de chaque cellule (S4503)
python scripts/aggregate_sprint45.py            # → exp_S45_summary.json (lecture seule)
```
- `exp_S45_summary.json` : `gap2_ok = true` partout ; `gap3_ram_ok = true` ; `verdict_parity = 1.000`.
- Ordre attendu du coût `.bss` : Page-Hinkley/DDM (O(1)) < PSI (O(bins)) < fenêtres O(W) — cohérent avec
  les annotations `# MEM:` et les proxies S44 (mais **chiffres réels** ici).
- Écart latence proxy-PC ↔ DWT-board commenté (le board FPU Cortex-M4 peut être plus rapide que le proxy
  Python, cf. paradoxe latence Sprint 29).

---

## Résolution (implémentée)

**Fichiers** : `scripts/aggregate_sprint45.py` (lecture seule → `experiments/exp_S45_summary.json`,
indexé `[dataset][detector][platform]`, `platform ∈ {board, pc_proxy}`) ; colonne
`gas_sensor_drift` mesurée board réelle via `run_sprint45_board.py` + `board_pc_parity45.py`
(S4503, inchangés).

**Décisions de conception** :
- **Distinction mesuré-board / proxy-PC** stricte par plateforme : latence board = DWT (P50/P99),
  `.bss` = `make size` ; proxy PC = `cost.latency_us_per_update` + `cost.state_bytes` (S44,
  `is_proxy: true`). Champ `latency_board_vs_proxy_us` documente l'écart.
- **`.bss` : mesuré vs delta méthode non conflatés** — `bss_bytes` rapporté tel quel (inclut la
  tête EWC, variable en k) ; les deltas firmware par méthode (`+36 PH / +40 DDM / +132 PSI`,
  S4502) exposés en constantes documentées `bss_delta_by_method`, **pas** recalculés depuis
  `bss_bytes`. `BSS_DEFAULT = 105 036 B` (build défaut invariant, condition de recevabilité).
- **N/A honnête propagé** : la raison propre du board (ex. overflow SRAM) prime sur le générique.

**Colonne `gas_sensor_drift` mesurée — board réelle NUCLEO-F439ZI** (128 features, 13 910
échantillons, seed 42, 0 CRC) :

| Détecteur | parité verdict | lat DWT P50/P99 | `.bss` | F1 détection | Gap 2 | Gap 3 |
|-----------|----------------|-----------------|--------|--------------|-------|-------|
| Page-Hinkley (S4503) | **1.000** (0/13 910) | 270 / 270 µs | 166 352 B | 0.0 | ✅ | ✅ |
| DDM | **1.000** (0/13 910) | 270 / 271 µs | 166 356 B | 0.190 | ✅ | ✅ |
| PSI | **N/A** (build non flashable) | — | — | — | — | — |

**Constat clé PSI (limite matérielle mesurée, honnête)** : PSI est piloté à bord par le score
Mahalanobis (`signal ← maha_score`) dont la covariance est **O(k²)**. À k=128 features,
`sigma_inv` (128²×4 ≈ 64 Ko) fait **déborder la SRAM** au link (`.bss` overflow ~69 064 B) →
`na_reason` = overflow. C'est un résultat Gap 3 réel : **PSI n'est pas portable en haute
dimension** (goulot = sa source de signal, pas l'état O(bins) du détecteur lui-même). PSI reste
mesurable sur un dataset basse-dimension (ex. hydraulic 17 feat, `sigma_inv` ≈ 1 Ko) → runbook.

**Écart latence proxy-PC ↔ DWT-board** (paradoxe FPU, S29) : DDM proxy PC = 6.1 µs vs board DWT
270 µs — le proxy Python **n'est pas prédictif** de la latence board (le chemin d'inférence EWC
domine, surcoût fixe). **Seule la mesure board fait foi.**

**Honnêteté** : reste de la grille (`hydraulic`, `synthetic` × 3 détecteurs) non flashé →
`null`/« non flashé (runbook) » ; `electricity` → N/A (pas de vérité-terrain ponctuelle). Aucun
chiffre inventé.
