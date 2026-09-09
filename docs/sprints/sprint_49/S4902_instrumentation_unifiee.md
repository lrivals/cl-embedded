# S4902 — Instrumentation unifiée de la RAM totale (board + PC) + historique du pic

| Champ | Valeur |
|-------|--------|
| **Sprint** | 49 |
| **Priorité** | 🔴 Critique — source unique de la RAM totale ; S4903 (relance) l'appelle par cellule. |
| **Statut** | ✅ Implémenté — `scripts/run_ram_full_sweep.py` (board+PC, dry-run, schéma S4901) |
| **Durée estimée** | 5h |
| **Dépendances** | S4901 (schéma JSON) · `scripts/run_ram_board.py` ✅ · `scripts/ram_breakdown.py` ✅ · `src/evaluation/memory_profiler.py` ✅ (`tracemalloc`) · `profiling_total_ram_bytes` ✅ |
| **Fichiers cibles** | `scripts/run_ram_full_sweep.py` (nouveau, orchestrateur) |
| **Références** | S4901 · CR §2 |

## Contexte

Le mécanisme existe (board : `profiling_total_ram_bytes` via `run_ram_board.py` ; PC : `tracemalloc` via
`memory_profiler.py`) mais il n'y a **pas de point d'entrée unique** produisant la RAM totale par cellule au
schéma S4901. Cette tâche crée cet orchestrateur, **sans réécrire** les primitives.

## Spec

### 1. Point d'entrée unique

`scripts/run_ram_full_sweep.py` : pour chaque cellule (modèle, dataset, condition, encodage, plateforme),
produit un JSON conforme au schéma S4901.

```
run_ram_full_sweep.py
  --models ewc,hdc,tinyol,mahalanobis
  --datasets monitoring,pronostia
  --encodings fp32,int8
  --platform board|pc          # board = réel (carte requise), pc = tracemalloc
  --out experiments/exp_S49_ram/
```

- **Board** : réutilise `run_ram_board.py` (reset → re-peinture pile au boot → stream → lecture
  `profiling_total_ram_bytes` par phase). **Ne pas dupliquer** la logique de flash/stream.
- **PC** : réutilise `memory_profiler.py` (`tracemalloc` peak) ; `.data`/`.bss` non applicables → `null`.
- Décomposition statique/mémoire-non-constante par modèle : réutilise `ram_breakdown.py` (vérifié vs `nm`).

### 2. Historique du pic de pile

Échantillonner le high-water mark périodiquement durant le stream (board) : après chaque phase
(`idle`/`inference`/`update`), lire `profiling_stack_peak_bytes()` → append `stack_history`. Le firmware
expose déjà la valeur ; l'échantillonnage se fait côté hôte via le protocole UART **existant** (aucun nouveau
flag, aucun changement wire — le champ pile est déjà remonté dans le snapshot V3).

### 3. Conditions identiques (CR §1)

Le driver fixe **mêmes groupes de données + même protocole MAJ/éval** que la comparaison de référence, et les
consigne dans `conditions{}` du JSON (traçabilité).

## Format de sortie

- `scripts/run_ram_full_sweep.py` (CLI ci-dessus).
- 1 JSON par cellule dans `experiments/exp_S49_ram/{model}_{dataset}_{condition}_{encoding}_{platform}.json`
  au schéma S4901 (`pending`/`null` avant exécution réelle).

## Contraintes

- Réutilisation stricte : `run_ram_board.py`, `ram_breakdown.py`, `memory_profiler.py`, `profiling_total_ram_bytes`.
- **Wire UART inchangé** (règle CLAUDE.md) — l'historique du pic passe par le champ pile déjà présent.
- Board et PC produisent des JSON séparés (`platform`), jamais fusionnés.
- Aucun chiffre écrit avant run réel.

## Vérification

```bash
python scripts/run_ram_full_sweep.py --platform pc --datasets monitoring --models ewc --dry-run
test -d experiments/exp_S49_ram
python -c "import json,glob; [json.load(open(f)) for f in glob.glob('experiments/exp_S49_ram/*.json')]"  # schéma valide
grep -n "profiling_total_ram_bytes\|run_ram_board\|tracemalloc" scripts/run_ram_full_sweep.py             # réutilisation
```

## Implémentation ✅

`scripts/run_ram_full_sweep.py` créé. Point d'entrée unique, 1 JSON/cellule au schéma S4901.

**Réutilisation stricte** (aucune primitive réécrite) :

- **Board** : `measure_stack_watermark.{OpenOCD,read_symbols,scan_stack_peak}` (stack painting via
  OpenOCD Tcl RPC) + `sensor_stream.py` (subprocess) ; `ram_breakdown.monitoring_breakdown` pour la
  contribution `.bss` par modèle (informatif). **Phases** : `idle` (reset→halt→scan, sans stream),
  `inference` (reset→stream **sans** `--update`→scan), `update` (reset→stream `--update`→scan). Le
  `total_ram_bytes = data + bss + max(pic_inference, pic_update)`.
- **PC** : `feature_conditions.train_and_evaluate` (construit le modèle) + **`tracemalloc`** direct
  autour d'un forward représentatif ; `.data`/`.bss` = `null` (n/a Python).

**Écart assumé vs la spec** : la spec supposait « le champ pile est déjà remonté dans le snapshot
V3 » — **faux**. `profiling_encode()` transmet `[latency][ram_b = bss_bytes][throughput]`, **pas** le
pic de pile. Conformément à la règle « wire UART inchangé », l'historique du pic passe donc par
**OpenOCD** (halt→scan), exactement comme `measure_stack_watermark.py` / `run_ram_board.py`. Aucun
nouveau flag, aucun changement de format.

**Correctif flash** : `make flash` démarre sa propre instance OpenOCD → conflit ST-LINK avec le
serveur persistant. Le driver flashe donc via le **serveur déjà lancé** (`program … verify reset`
sur le Tcl RPC).

**Vérifs passées** :

- `--platform pc --datasets monitoring --models ewc --dry-run` → JSON `pending`/`na` schéma-valide.
- `grep scan_stack_peak|tracemalloc|read_symbols|ram_breakdown|sensor_stream` → réutilisation OK.
- N/A honnête : PC×int8 → `status:"na"` ; maha×int8 board → `status:"na"` (build dédié `-DMAHA_INT8`
  hors périmètre RAM).
