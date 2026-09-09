# S4901 — Cadrage RAM complète (`.data + .bss + pic de pile`) + instants de mesure

| Champ | Valeur |
|-------|--------|
| **Sprint** | 49 |
| **Priorité** | 🔴 Critique — fonde la méthode et le schéma JSON ; S4902 (instrumentation) et S4903 (relance) en découlent. |
| **Statut** | ✅ Implémenté — méthode figée dans `docs/context/ram_measurement.md` (§3bis/3ter/3quater) |
| **Durée estimée** | 3h |
| **Dépendances** | `firmware/stm32f4_blink/src/profiling.c` ✅ (`profiling_total_ram_bytes`) · `docs/presentation_ram_measurement.md` ✅ · `scripts/ram_breakdown.py` ✅ |
| **Fichiers cibles** | `docs/context/ram_measurement.md` (nouveau) · `docs/sprints/sprint_49/S4901_cadrage_ram_complete.md` |
| **Références** | CR 16 juillet 2026 §2 · `presentation_ram_measurement.md` (§3 « `.bss` oublie la pile ») |

## Contexte

Le CR corrige une **sous-estimation systématique** : les expériences rapportaient `.bss` seul, qui **exclut la
pile**. La formule officielle devient `RAM totale = .data + .bss + pic de pile`. Le mécanisme existe déjà
(`profiling.c`) ; cette tâche **fige la méthode, les instants de mesure et le schéma de données** que
consommeront S4902–S4904.

## Spec

### 1. Formule et composants

| Composant | Source firmware | Source PC |
|-----------|-----------------|-----------|
| `.data` | `&_edata − &_sdata` (linker) | n/a (statique) |
| `.bss` | `&_ebss − &_sbss` (linker) | n/a |
| pic de pile | `profiling_stack_peak_bytes()` (stack painting `[_ebss, _estack)`) | `tracemalloc` peak (`memory_profiler.py`) |
| **total** | `profiling_total_ram_bytes()` | `.data`+`.bss` analytiques + peak tracemalloc |

> **Board = référence** (mesure matérielle réelle). PC = proxy analytique + `tracemalloc` (dynamique).
> Les deux ne sont **pas directement comparables** (CR §1) — les reporter séparément, jamais fusionnés.

### 2. Instants de mesure alignés sur les phases réelles

Le pic de pile **dépend de la phase**. Instants à échantillonner (remplacent les étapes génériques « réseau de
neurones ») :

| Phase | Déclencheur | Attendu |
|-------|-------------|---------|
| `idle` | après boot, avant 1er échantillon | pile ≈ 0 (référence de peinture) |
| `inference` | forward seul (protocole gelé) | pic bas |
| `update` | inférence + mise à jour CL (SGD embarqué) | pic plus élevé (creuse la pile) |

L'**historique** = suite de `(phase, stack_peak_bytes)` échantillonnée durant le stream, pour tracer
l'évolution (CR §2 conclusion : « côté historique d'évolution du pic de pile durant exécution intéressant »).

### 3. Schéma JSON par cellule (consommé par S4903/S4904)

```json
{
  "model": "ewc", "dataset": "pronostia", "condition": "5feat", "encoding": "fp32",
  "platform": "board",
  "data_bytes": null, "bss_bytes": null,
  "stack_peak_inference_bytes": null, "stack_peak_update_bytes": null,
  "total_ram_bytes": null,
  "stack_history": [],
  "conditions": {"same_groups": true, "update_protocol": "...", "eval_protocol": "..."},
  "status": "pending"
}
```

`null`/`pending` tant qu'aucun run n'a tourné (règle projet : aucun chiffre inventé).

## Format de sortie

Document `docs/context/ram_measurement.md` :

```
# Mesure RAM complète — méthode CL-Embedded
## 1. Formule .data + .bss + pic de pile (+ pourquoi .bss seul sous-estime)
## 2. Stack painting (renvoi profiling.c, déjà implémenté)
## 3. Instants de mesure par phase (idle / inference / update)
## 4. Board (référence) vs PC (proxy) — non fusionnables (CR §1)
## 5. Schéma JSON par cellule + historique du pic
## 6. Renvois : presentation_ram_measurement.md, ram_breakdown.py, run_ram_board.py
```

## Contraintes

- Aucun chiffre de résultat (méthode pure).
- Ne pas réécrire le stack painting : **renvoyer** à `profiling.c` (déjà validé, tests Unity `test_stack_peak_*`).
- Board et PC restent des colonnes séparées, jamais additionnées.

## Vérification

```bash
test -f docs/context/ram_measurement.md
grep -c "pic de pile\|stack_peak\|\.bss" docs/context/ram_measurement.md   # > 0
grep -i "inference\|update\|idle" docs/context/ram_measurement.md          # phases présentes
grep -i "profiling.c\|stack painting" docs/context/ram_measurement.md      # renvoi infra existante
```

## Implémentation ✅

Le doc `docs/context/ram_measurement.md` **existait déjà** (créé S39 : contiguïté `.bss` + pic +
borne statique). S4901 l'a **enrichi** (pas recréé) de trois sections :

- **§3bis — Instants de mesure par phase** (`idle`/`inference`/`update`) : tableau déclencheur →
  méthode (reset entre passes ; frozen stream = pic inference ; `--update` stream = pic update) →
  attendu, + note « trame partagée `pipeline_run()` ».
- **§3ter — Board (référence) vs PC (proxy)** : tableau des colonnes non fusionnables (CR §1).
- **§3quater — Schéma JSON par cellule + historique du pic** : bloc JSON exact + renvoi à
  l'orchestrateur `scripts/run_ram_full_sweep.py` (S4902).

**Correctif de nommage** : la fonction firmware réelle est **`profiling_ram_peak_bytes`** (= `.data
+ .bss + pic`), pas `profiling_total_ram_bytes` (nom cité dans la dépendance, inexistant). Le doc et
le code utilisent le vrai nom.

**Nuance mesurée (board réelle, S4903)** : `idle` **≠ 0** (≈ 4 236 B) — le compilateur réserve dès
le boot la trame `pipeline_run()`. La hypothèse « idle ≈ 0 » de la spec est donc corrigée par la
mesure ; seul EWC voit `update > inference` (SGD backward), les autres modèles ont `inference ==
update` (trame partagée).
