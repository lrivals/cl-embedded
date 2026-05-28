# S2113 — Protocole expérimental board : cold/warm run, répétitions × 3, vérification état carte

| Champ | Valeur |
|-------|--------|
| **Sprint** | 21 |
| **Priorité** | 🟡 Moyenne |
| **Statut** | ✅ Terminé (2026-05-28) |
| **Durée estimée** | 2 h |
| **Dépendances** | S1906 (protocol_v3) · S1907 (experiment_recorder) |
| **Fichiers cibles** | `scripts/board_experiment_recorder.py` |
| **Référence** | Applicable rétrospectivement aux expériences sprints 16–21 |

---

## Contexte

Les expériences sur carte (sprints 16–21) sont lancées sans protocole formalisé. Cela introduit trois biais potentiels :

1. **Cache I/D ARM Cortex-M4 chaud** — un warm run donne une latence artificiellement basse car le branch predictor et les caches L1 instruction/data sont déjà chargés. Sans cold run préalable, les premières mesures ne sont pas reproductibles.
2. **État résiduel des poids et variables globales** — si on enchaîne deux expériences sans reset matériel, les tableaux globaux C (poids EWC, centroïde Mahalanobis, buffer TinyOL) contiennent des valeurs résiduelles du run précédent. Le startup code remet les `.bss` à zéro uniquement après un reset hardware.
3. **DWT non réinitialisé** — le compteur DWT (`DWT->CYCCNT`) peut dépasser `UINT32_MAX` et wrap si l'expérience précédente ne l'a pas arrêté proprement, faussant la mesure de latence.

Sans protocole, les métriques publiées (notamment `inference_latency_ms`) ne sont pas comparables entre expériences.

---

## Objectif

Définir un protocole expérimental standard, reproductible, applicable à toutes les expériences board passées (16–21) et futures. Implémenter le support dans `board_experiment_recorder.py` pour automatiser les répétitions et journaliser les conditions de run.

---

## Protocole en 4 points

### a) Cold run vs Warm run

| Type | Définition | Quand utiliser |
|------|------------|----------------|
| **Cold run** | Reset hardware (NRST) immédiatement avant le run → caches ARM vidés, pipeline flushed, tous les globaux `.bss` réinitialisés par le startup code | Toujours en premier, pour vérifier l'absence de régression au démarrage |
| **Warm run** | Exécution immédiate après le cold run **sans reset** → conditions de régime établi | Source des métriques de performance publiées |

**Règle** : les métriques reportées dans `results.json` et le manuscrit proviennent du **warm run**. Le cold run est enregistré séparément à titre de référence.

Commande reset via OpenOCD :
```bash
openocd -f interface/stlink.cfg -f target/stm32f4x.cfg \
    -c "init; reset halt; resume; exit"
```

Ou reset physique : bouton NRST de la NUCLEO-F439ZI (équivalent, préféré si ST-LINK non connecté en permanence).

### b) Répétitions × 3

Chaque expérience est lancée **3 fois** de façon indépendante (3 cycles cold + warm) :

- Reporter : **moyenne ± écart-type** pour `inference_latency_ms`, `acc_final`, `avg_forgetting`, `ram_peak_bytes`
- **Seuil de cohérence** : si σ(latency) > 5 % de la moyenne → investiguer (DWT mal configuré, interruptions UART parasites pendant mesure, ou température MCU variable)
- La répétition est le cycle complet (reset + stream + collecte résultats), pas un simple re-run du streamer

### c) Vérification état de la carte avant chaque run

Checklist à appliquer **avant chaque répétition** :

1. **Reset NRST hard** — via bouton physique ou `openocd reset halt` (pas de soft reset logiciel qui ne vide pas tous les caches)
2. **Vérification LED firmware** — LED verte LD2 = board prête et firmware démarré (défini dans `main.c` après init PLL + UART)
3. **Attente 2 s après reset** — stabilisation horloge PLL HSE (risque de timeout UART si UART init avant PLL locked)
4. **Re-flash si changement de modèle** — ne jamais réutiliser le flash d'une expérience précédente d'un autre modèle ; re-flasher le binaire correspondant :
   ```bash
   make -C firmware/stm32f4_blink/ flash MODEL=ewc
   ```
5. **Vérifier absence de processus zombie** — s'assurer qu'aucun `sensor_stream.py` précédent n'est encore connecté au port série (`lsof /dev/ttyACM0`)

### d) Journalisation des conditions de run

Chaque `results.json` doit inclure un champ `run_conditions` :

```json
"run_conditions": {
  "run_type": "warm",        // "cold" | "warm"
  "repetition": 2,           // index 1-based dans la série (1, 2 ou 3)
  "reset_method": "nrst",    // "nrst" | "openocd"
  "flash_fresh": true,       // vrai si le binaire a été re-flashé avant ce run
  "board_temp_c": null       // optionnel — si thermomètre externe disponible
}
```

Et un champ `run_statistics` en fin de série :

```json
"run_statistics": {
  "n_repetitions": 3,
  "inference_latency_ms_mean": 5.49,
  "inference_latency_ms_std": 0.03,
  "acc_final_mean": 0.782,
  "acc_final_std": 0.004,
  "avg_forgetting_mean": 0.053,
  "avg_forgetting_std": 0.001
}
```

---

## Ce qu'il faut implémenter dans `board_experiment_recorder.py`

### Nouveaux flags CLI

```
--repetitions N       Nombre de répétitions (défaut : 3)
--run-type TYPE       "cold" ou "warm" (défaut : "warm") — injecté dans run_conditions
--reset-method METHOD "nrst" ou "openocd" (défaut : "nrst") — pour journalisation
--flash-fresh         Flag booléen : indique que le binaire a été re-flashé avant ce run
```

### Boucle de répétitions

```python
results_per_rep = []
for rep in range(1, args.repetitions + 1):
    # Attente confirmation reset de l'utilisateur (ou automatique via openocd si --reset-method openocd)
    result = run_single_experiment(...)
    result["run_conditions"] = {
        "run_type": args.run_type,
        "repetition": rep,
        "reset_method": args.reset_method,
        "flash_fresh": args.flash_fresh,
        "board_temp_c": None,
    }
    results_per_rep.append(result)

# Calcul mean ± std après la boucle
aggregate_statistics(results_per_rep)  # écrit run_statistics dans le JSON final
```

### Affichage en fin de run

```
=== Résultats (3 répétitions) ===
inference_latency_ms : 5.49 ± 0.03 ms  (σ/μ = 0.5 % ✅)
acc_final            : 0.782 ± 0.004   (σ/μ = 0.5 % ✅)
avg_forgetting       : 0.053 ± 0.001
```

---

## Vérification

```bash
# Dry-run avec 3 répétitions — vérifie que la boucle tourne
python scripts/board_experiment_recorder.py \
    --config configs/board_ewc.yaml \
    --model ewc \
    --exp-id test_protocol \
    --repetitions 3 \
    --run-type warm \
    --dry-run \
    --output /tmp/test_protocol

# Vérifier le JSON produit
python -c "
import json, pathlib
r = json.loads(pathlib.Path('/tmp/test_protocol/results.json').read_text())
assert 'run_conditions' in r, 'run_conditions manquant'
assert 'run_statistics' in r, 'run_statistics manquant'
assert r['run_statistics']['n_repetitions'] == 3
print('Protocol test OK:', r['run_statistics'])
"

# Application réelle : utiliser S2104 (exp_pronostia_board) comme premier run avec protocole
python scripts/board_experiment_recorder.py \
    --config configs/board_pronostia.yaml \
    --model ewc \
    --exp-id ewc_pronostia_l400_protocol \
    --repetitions 3 \
    --run-type warm \
    --flash-fresh \
    --output experiments/exp_S21_04_protocol
```

---

## Première application — Sprint 21 (2026-05-28)

Protocole appliqué à toutes les expériences board sprint 21. Résultats avec **3 répétitions warm run**, reset OpenOCD automatique entre répétitions, firmware re-flashé avant E21-01.

| Expérience | Modèle | Dataset | acc moy ± σ | AF moy ± σ | lat ms ± σ | RAM B | σ/μ lat | Gap 2 |
|------------|--------|---------|:-----------:|:----------:|:----------:|:-----:|:-------:|:-----:|
| E21-01 | Mahalanobis | Monitoring | 0.107 ± 0.012 | 0.011 ± 0.008 | 0.004 ± 0.000 | 200 | 0.0 % ✅ | ✅ |
| E21-02 | TinyOL | Monitoring | 0.114 ± 0.010 | 0.000 ± 0.000 | 0.004 ± 0.000 | 5 800 | 0.0 % ✅ | ✅ |
| E21-03 | Mahalanobis | Pronostia | 0.094 ± 0.007 | 0.000 ± 0.000 | 0.004 ± 0.000 | 200 | 0.0 % ✅ | ✅ |
| E21-04 | EWC λ=400 | Pronostia | 0.886 ± 0.023 | 0.146 ± 0.025 | 0.251 ± 0.000 | 9 728 | 0.2 % ✅ | ✅ |
| E21-04b | EWC λ=0 | Pronostia | 0.852 ± 0.011 | 0.204 ± 0.017 | 0.250 ± 0.001 | 9 728 | 0.2 % ✅ | ✅ |
| E19-02b | EWC λ=400 | Monitoring | 0.896 ± 0.003 | 0.010 ± 0.012 | 0.249 ± 0.001 | 9 728 | 0.2 % ✅ | ✅ |

**Observations** :

- Seuil σ/μ latence ≤ 5 % respecté pour tous les modèles → protocole validé.
- EWC board : acc très stable (σ/μ ≤ 2.6 %) ; AF plus variable (σ/μ ~17 %) → normal, dépend de l'ordre des samples stream.
- Mahalanobis et TinyOL : acc ~10 % = cold start sans poids pré-chargés (voir FIXME gap1 dans S2103).
- Propriété EWC vérifiée sur board : AF(λ=400)=0.146 < AF(λ=0)=0.204.

---

## Questions ouvertes

- `TODO(arnaud)` : Faut-il reporter mean ± std dans les tableaux du manuscrit, ou seulement la moyenne avec σ en note de bas de tableau ?
- `TODO(arnaud)` : Les expériences sprints 16–20 doivent-elles être re-courues avec ce protocole, ou les valeurs existantes (dry-run) sont-elles suffisantes pour le chapitre comparatif ?
