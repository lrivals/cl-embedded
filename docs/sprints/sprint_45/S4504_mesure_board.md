# S4504 — Mesure board réelle (RAM `.bss` + latence DWT) & agrégat

| Champ | Valeur |
|-------|--------|
| **Sprint** | 45 |
| **Priorité** | 🔴 Critique — c'est le résultat *système* du sprint : coût réel des détecteurs sur MCU. |
| **Statut** | 📝 Doc — spec ; implémentation à venir (nécessite la carte). |
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
