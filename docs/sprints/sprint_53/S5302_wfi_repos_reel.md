# S5302 — `__WFI()` sur l'attente UART : créer un vrai repos

| Champ | Valeur |
|-------|--------|
| **Sprint** | 53 |
| **Priorité** | 🔴 Critique — verrou n°3 du sprint |
| **Statut** | ✅ Implémenté et **mesuré sur carte réelle** (NUCLEO-F439ZI + X-NUCLEO-LPM01A, 2026-08-06) |
| **Durée estimée** | 3 h (2 h firmware + 1 h banc) |
| **Dépendances** | S5301 (verdict sur l'anomalie) · carte + sonde |
| **Fichiers cibles** | `firmware/stm32f4_blink/src/pipeline.c` · `firmware/stm32f4_blink/src/hw_info.c` (armement IT) · `configs/hw_profile_f439zi.yaml` (`veille_uA`) |
| **Références** | `run_s50_board_current.py:23-25` (le levier y est déjà identifié, hors périmètre S50) |

## Contexte

Le firmware attend chaque trame en **scrutation active** :

```c
/* firmware/stm32f4_blink/src/pipeline.c:240-243 */
static uint8_t uart_getbyte(void)
{
    while (!(USART3->SR & (1U << 5))) {}   /* attendre RXNE */
    return (uint8_t)USART3->DR;
}
```

Le cœur tourne donc à 180 MHz, 100 % du temps, y compris « au repos ». `PHASE_IDLE`
(`profiling.h:157`) n'est qu'un marqueur GPIO, pas un état de veille. Conséquence directe
et déjà écrite dans le pilote S50 : **il n'existe aucune référence de repos exploitable**,
donc l'énergie marginale par inférence du protocole delta ressort négative.

`run_s50_board_current.py:23-25` identifie explicitement le levier — « mise en sommeil
`WFI` de l'attente UART, pour disposer d'un vrai repos » — et le place hors périmètre.
**Ce sprint le prend en charge.**

## Spec

### 1. Modification, gardée par `-DUART_WFI_IDLE`

```c
static uint8_t uart_getbyte(void)
{
    while (!(USART3->SR & (1U << 5))) {
#ifdef UART_WFI_IDLE
        __asm volatile ("wfi");
#endif
    }
    return (uint8_t)USART3->DR;
}
```

**Piège central à traiter, sinon le firmware se fige** : `WFI` ne sort que sur une
interruption (ou un événement de débogage). Sans source de réveil armée, la carte ne
répondra plus jamais à une trame. Deux voies, à trancher **sur banc** :

| Voie | Mécanique | Coût |
|------|-----------|------|
| **A — IT RXNE** | `USART3->CR1 \|= USART_CR1_RXNEIE` + `NVIC_ISER` pour USART3, avec un `USART3_IRQHandler` **vide** (il ne consomme pas `DR`). Le `while` reste la source de vérité, l'IT ne sert qu'à sortir du `WFI`. | un vecteur d'IT + ~20 B |
| **B — `WFE` + `SEVONPEND`** | `SCB->SCR \|= SEVONPEND` puis `__asm volatile ("wfe")` : une IT *pending* réveille sans être servie, donc **aucun handler requis**. | pas de vecteur, mais sémantique plus subtile |

Recommandation : commencer par **A** (comportement le plus explicite et le plus facile à
déboguer) ; basculer sur B si le vecteur pose problème dans `startup_stm32f439xx.s`.

L'armement de l'IT doit lui-même être gardé par `#ifdef UART_WFI_IDLE` dans
`hw_uart_init` (`hw_info.c:156+`), sinon le build par défaut change.

### 2. Non-régression obligatoire (avant toute mesure d'énergie)

- `make test` → **145 tests, 0 échec** (baseline vérifiée ; les 2 échecs TinyOL
  historiques ont été résolus au Sprint 52 — aucun échec n'est désormais toléré).
- Build par défaut : **`.bss` invariant 105 036 B**, aucun warning.
- Build `-DUART_WFI_IDLE` : flux de **300 échantillons à 200 Hz, 0 erreur CRC**, parité
  board↔PC inchangée. C'est le point qui valide que le réveil fonctionne à cadence élevée
  — un WFI mal armé se manifeste par des trames perdues, pas par un plantage franc.

### 3. Mesures de banc

1. **Courant de repos WFI** — carte flashée `-DUART_WFI_IDLE`, aucun flux, port fermé.
   C'est le **plancher** qui manque à `configs/hw_profile_f439zi.yaml:34-36`
   (`actif_mA` et `veille_uA`, tous deux `null` aujourd'hui).
2. **Reprise du protocole delta** — relancer `scripts/run_s50_energy_delta.py` (écrit,
   13 tests, jamais exécuté sur carte) avec cette nouvelle référence.
   `energy_capture.energy_uj_per_inference_delta()` (`energy_capture.py:174`) renvoie déjà
   `None` si `delta_a <= 0` : si le delta redevient positif, les µJ/inférence sont
   calculés ; sinon le N/A persiste avec une raison actualisée.
3. **Reprise des 8 cellules S50** avec la référence WFI, pour comparaison directe avec
   `exp_S50_energy/` à protocole identique.
4. **Autonomie** — `autonomy.average_current_ma_from_delta()`
   (`src/evaluation/autonomy.py:92`), écrite et jamais exercée, devient la voie principale.

### 4. Sortie

`experiments/exp_S53_wfi/` :

- `idle_reference.json` — courant de repos avec et sans WFI, même session, contre-balancé
  (réutiliser le protocole S5301).
- `{model}_{encoding}.json` — 8 cellules reprises, schéma identique à `exp_S50_energy/`
  plus un champ `firmware_build: "UART_WFI_IDLE"`.
- `delta_recovery.json` — sortie de `run_s50_energy_delta.py`, avec `method: "delta vs
  repos WFI"`.

## Critères d'acceptation

- [x] Build par défaut strictement inchangé : `.bss` **105 036 B**, `make test` **145 / 0
      échec** (+ `make test-batch` 148 / 0).
- [x] Build WFI : **300 échantillons reçus, 0 erreur CRC**, latence DWT 50 µs inchangée
      (`experiments/exp_S53_wfi/wake_validation.json`).
- [x] Courant de repos WFI mesuré et écrit dans `hw_profile_f439zi.yaml` :
      `veille_uA: 27370.0`, `actif_mA: 32.891`, avec `mesure_source` — **aucune valeur
      estimée**.
- [x] `energy_uj_per_inference` par delta : **8/8 cellules chiffrées**, `method: "delta vs
      repos UART_WFI_IDLE"`, reporté jusque dans chaque fichier de cellule.
- [x] Gain énergétique du WFI chiffré : **−22,40 mA, soit −45,0 %** au repos.

## Extension — boucle par lot (`-DINFER_BATCH_N`)

**Ajoutée le 2026-08-05 sur mesure de banc**, dans cette tâche parce qu'elle touche le même
fichier, le même patron de garde et le même reflash que le WFI.

### Pourquoi elle est nécessaire, et pas confortable

Un balayage de cadence a été piloté sur deux cellules (2 répétitions × 5 cadences, ordre
entrelacé). Résultats :

| cellule | latence | pente → µJ/inférence | R² |
|---|---|---|---|
| HDC INT8 | 1958 µs | 233,9 ± 23,7 | 0,924 |
| Mahalanobis (témoin) | 5 µs | **60,6 ± 4,6** | 0,956 |

Le témoin devait sortir indiscernable de zéro (~0,4 µJ de calcul attendu). Il sort à
60,6 µJ : **la pente mesure le coût d'une trame UART, pas le calcul**. La double différence
donne le coût de calcul réel, `233,9 − 60,6 = 173 ± 24 µJ` pour HDC INT8 — ce qui valide la
méthode de S5304, mais montre aussi sa limite.

Cette limite est structurelle : **le plafond de cadence est l'UART, pas le modèle.** Une
transaction fait 32 B (requête) + 23 B (réponse V3) = 55 B, soit 4,77 ms à 115200 bauds →
**~209 Hz au maximum**. À cette cadence, le taux d'occupation vaut :

| modèle | latence | occupation à 209 Hz |
|---|---|---|
| HDC INT8 | 1958 µs | 41 % — mesurable |
| TinyOL | 85 µs | 1,8 % |
| EWC | 50 µs | 1,0 % |
| Maha | 5 µs | 0,1 % — noyé |

Les modèles rapides — ceux que `S5300` §4 déclare « sous le bruit », et dont dépend la
question INT8 du Gap 3 — **restent inaccessibles quel que soit le balayage de cadence**.

### Spec

Exécuter `N` inférences par trame reçue, sous garde de compilation, `N` connu de l'hôte :

```c
#ifndef INFER_BATCH_N
#define INFER_BATCH_N 1            /* build par défaut strictement inchangé */
#endif
for (int i = 0; i < INFER_BATCH_N; i++) { /* … chemin d'inférence existant … */ }
```

Le taux d'occupation devient réglable **indépendamment de l'UART** : à 100 Hz avec
`N = 200`, EWC passe de 1 % à ~100 % d'occupation. La régression se fait alors sur `N` à
cadence fixe — `I_moy(N) = I_base + (E_calcul / V) · f · N` — où `I_base` absorbe le coût
de trame que le témoin Maha vient de chiffrer. C'est la seule voie mesurée vers un
µJ/inférence pour EWC, TinyOL et Mahalanobis.

**Contraintes** : seule la dernière inférence alimente la réponse UART (format V3 inchangé,
`sensor_stream.py` intact) ; le lot ne doit pas déclencher `N` mises à jour CL quand
`--update` est actif (borner la MAJ à la dernière itération, sinon la sémantique CL change) ;
`N` est reporté dans la cellule JSON, jamais supposé.

**Critères d'acceptation** : `INFER_BATCH_N=1` ⇒ `.bss` 105 036 B et `make test` 145/0
inchangés ; à `N > 1`, 0 erreur CRC et prédiction identique à `N = 1` sur le même
échantillon ; pente vs `N` linéaire (R² ≥ 0,95) pour au moins une cellule rapide.

## État d'implémentation (2026-08-05) — code

### Ce qui est en place et vérifié

| Élément | Où | Vérification |
|---|---|---|
| `__WFI` sur l'attente UART (voie **A**, IT RXNE) | `pipeline.c:239-266` | `make check-wfi` : compile, **`.bss` 105 036 B identique** au build par défaut (le WFI n'ajoute aucun état ; Flash +72 B) |
| Armement `NVIC_ISER1` + `USART3_IRQHandler` qui ne consomme pas `DR` | `hw_info.c:234-257` | gardés `#ifdef UART_WFI_IDLE` — build par défaut sans aucune interruption |
| Vecteur `USART3_IRQHandler` → `Idle_Wake_Handler` (retour immédiat) | `startup_stm32f439xx.s:88-166` | — |
| Boucle par lot `-DINFER_BATCH_N` | `pipeline.c:826-960` (`INFER_BATCH_N` déplacé dans `pipeline.h`) | `make test` **145 / 0 échec**, `make test-batch` **148 / 0** |
| Pilote de banc (`idle` / `cells` / `batch`, delta, régression `I(N)`) | `scripts/run_s53_wfi.py` | 10 tests dans `tests/test_s53_bench.py` |
| Écriture du repos mesuré dans le profil matériel | `run_s53_wfi.py --write-hw-profile` | substitution ligne à ligne (commentaires préservés) ; **refus d'écrire sans mesure** |

### Défaut trouvé et corrigé : le lot exécutait le mauvais modèle

`infer_extra()` dispatchait par **masque de sous-ensemble** alors que la boucle de lot
s'exécute **avant** les sorties anticipées des modes composés. Conséquence, à `N > 1` :

| trame | branche empruntée par le lot (avant correctif) |
|---|---|
| `0xF0` Maha Q15, `0xE0` TRIPLE, `0x70` DUAL | HDC INT8 (`& 0x60 == 0x60`) |
| `0xD0` TRIPLE | TinyOL INT8 (`& 0xC0 == 0xC0`) |
| `0x30` multiclasse, `0x50` RUL | `ewc_forward` au lieu de `ewc_mc_forward` / `ewc_reg_predict` |

La prédiction restait juste — la passe finale est la vraie chaîne — mais **les µJ auraient
été imputés au mauvais noyau**. C'est un défaut d'**attribution**, invisible à toute
assertion sur l'état des modèles : il fallait donc l'instrumenter. `infer_extra` teste
désormais le nibble en **égalité**, sort sans aucune passe pour les modes composés (hors
périmètre de la campagne énergie, qui porte sur les 8 cellules de `STREAM_MODEL`), et
route multiclasse/RUL vers leur propre tête. Le contrat est verrouillé par
`test_pipeline_batch_composite_non_batche`, qui échoue effectivement (`Expected 0 Was 3`)
lorsqu'on réinjecte le défaut.

## Campagne de banc (2026-08-06) — carte réelle NUCLEO-F439ZI + X-NUCLEO-LPM01A

Session unique, sonde alimentant `VDD_MCU` (`JP5` retiré) : **le domaine mesuré est le
STM32**, hors ST-LINK. Toute exécution hôte (flash, flux) passe par
`lpm01a_probe.py --hold-run` — hors acquisition la sortie de la sonde ne tient pas la
carte. **Point de banc découvert et consigné** : sur un binaire WFI, le cœur dort quand
OpenOCD se présente et `make flash` échoue (`init mode failed` / `Unable to reset
target`) ; il faut connecter sous reset :
`-c 'reset_config srst_only srst_nogate connect_assert_srst'`.

### 1. Le réveil tient la cadence

`wake_validation.json` : **300/300 trames reçues, 0 erreur CRC**, latence DWT P50 = P99 =
**50 µs** (identique au FP32 sans WFI) — le sommeil ne coûte rien au chemin d'inférence.
Cadence atteinte 166,8 Hz, sous le plafond UART de ~209 Hz, sans perte de trame.

### 2. Gain du WFI — le résultat principal

| Repos | Courant établi | Dispersion |
|---|---|---|
| scrutation active (build par défaut) | **49,767 mA** | 10 acquisitions, même session |
| `-DUART_WFI_IDLE` | **27,370 mA** | 10 acquisitions, même session |
| **gain** | **−22,397 mA, −45,0 %** | `idle_reference.json` → `wfi_gain` |

Les deux références sont prises dans la **même session**, chacune après deux acquisitions
de préchauffage écartées, régime établi isolé par `counterbalance.established_regime`
(protocole S5301). La mesure du 2026-08-05 est conservée dans
`idle_reference_2026-08-05.json`.

### 3. Le protocole delta est débloqué — 8/8 cellules chiffrées

Le repos WFI passe **sous** toutes les cellules : le delta redevient positif, ce qui lève
le N/A du Sprint 50 (`delta_recovery.json`, 100 Hz, fenêtre 10 s, N = 1000).

| cellule | ΔI vs repos | µJ / inférence |
|---|---|---|
| maha_fp32 | +4,120 mA | 134,5 |
| tinyol_int8 | +4,323 mA | 141,1 |
| tinyol_fp32 | +4,420 mA | 144,3 |
| ewc_int8 | +4,443 mA | 145,0 |
| maha_int8 | +4,473 mA | 146,0 |
| ewc_fp32 | +4,497 mA | 146,8 |
| hdc_fp32 | +6,427 mA | 209,8 |
| hdc_int8 | +11,467 mA | 374,3 |

**Lecture honnête** : ces µJ sont ceux d'une **transaction complète** (trame UART +
inférence), pas du calcul seul. Le témoin le prouve — Mahalanobis, dont l'inférence dure
5 µs, sort à 134 µJ. Le coût de trame est le plancher commun ; c'est exactement ce que le
balayage par lot sépare ci-dessous.

`maha_int8` a exigé son propre flash (`-DMAHA_INT8 -DUART_WFI_IDLE`, `.bss` 105 096 B =
+60 B conforme S29) et se mesure par `--mode cells --only maha_int8` : sans cette
restriction, elle serait écrite depuis le build FP32 — le bug du Sprint 52.

### 4. Le lot `-DINFER_BATCH_N` isole le calcul

Régression `I(N)` à 100 Hz, `N ∈ {1, 50, 100, 200}` (un reflash par point), 3 répétitions.

| cellule | pente | r² | **µJ / inférence (calcul seul)** |
|---|---|---|---|
| ewc_fp32 | 0,15558 mA/N | **0,9998** | **5,08** |
| maha_fp32 | 0,01340 mA/N | **0,9998** | **0,437** |

Les deux dépassent le critère r² ≥ 0,95. Cohérence croisée : à 50 µs et 5 µs de latence
DWT sous ~79 mW de surcoût pleine charge, on attend ≈ 4 µJ et ≈ 0,4 µJ — mesuré 5,08 et
0,437. Le témoin Maha, que le protocole delta plaçait à 134 µJ, tombe à **0,437 µJ** une
fois la trame retirée : **le coût d'une inférence rapide est ~300× plus petit que le coût
de la transaction qui la transporte.**

**Règle de saturation, mesurée et non supposée** : le point `N = 200` d'EWC a été
**écarté** de la régression parce que la cadence réellement atteinte tombe à 66,2 Hz
(< 95 % de 100 Hz) — 200 × 50 µs = 10 ms occupe toute la période de trame et
`sensor_stream` sature **en silence** (0 CRC, 0 trame perdue, le flux tourne simplement
moins vite). Le retenir aurait rabattu la pente et **sous-estimé** l'énergie. La règle vit
dans le pilote (`SATURATION_RATE_TOLERANCE`), le point reste consigné avec
`saturated: true`.

**Parité du lot vérifiée à bord** : à chaque `N > 1`, les prédictions sont **identiques à
celles de `N = 1`** (parité 1.000 sur 490–560 échantillons communs par point, appariés
par vecteur de features — deux flux successifs ne tirent pas la même séquence, un
appariement par indice inventerait des désaccords). Latence DWT × N : 50 → 2369 → 4735 →
9467 µs, linéaire comme attendu.

### 5. Autonomie duty-cyclée — enfin calculable

`autonomy_delta.json` (`autonomy.average_current_ma_from_delta`, jamais exercée jusqu'ici) :
avec le repos WFI comme plancher, une période d'inférence de 1 s donne **≈ 73 h sur
2000 mAh** pour toutes les cellules, contre **43 h** au Sprint 50 (flux continu, carte
jamais endormie). À 100 Hz (T = 0,01 s) le modèle redonne exactement le courant mesuré de
chaque cellule (ex. EWC 31,868 mA reconstruit vs 31,867 mA mesuré) : le modèle delta se
referme sur lui-même. La période est un **paramètre de scénario déclaré**, pas une mesure.

### 6. Corrections d'outillage apportées par la campagne

- `--only` en mode `cells` (patron S50) : sans lui, la cellule à firmware dédié était
  inatteignable ; et le delta est désormais reconstruit depuis **toutes** les cellules du
  build présentes sur disque, sinon un passage `--only` écrasait le delta des 7 autres.
- `apply_fit` extrait en fonction pure : les clés du verdict opposé (`na_reason` ↔ valeur)
  sont retirées à chaque passage. Le défaut était réel — après la re-mesure concluante, le
  fichier portait **à la fois** 5,08 µJ et la raison de son absence.
- `propagate_delta_into_cells` : la cellule elle-même porte le résultat du delta. Sans
  cela, `ewc_fp32.json` disait « à mesurer » avec la raison du Sprint 50 pendant que
  `delta_recovery.json` du même répertoire donnait 146,8 µJ.
- `iter_cell_files` : source unique de « qu'est-ce qu'un fichier de cellule », partagée par
  le profil matériel, le delta et les tests (trois listes de noms réservés divergeaient).

`tests/test_s53_bench.py` : **35 PASS** (+10 : saturation, clés périmées, parité par
features, propagation, schémas mesurés). `make test` **145 / 0**, `make test-batch`
**148 / 0**, `.bss` par défaut **105 036 B**. La carte est **rendue au build par défaut**.

## Portée et limite honnête

Le WFI change le **firmware de banc**, pas le firmware de référence des sprints
précédents : toutes les latences DWT et `.bss` publiés restent valides (le build par
défaut est inchangé). Le gain étant avéré (−45 % au repos), il faut dire dans le mémoire
que **les autonomies antérieures ont été mesurées sans veille, donc pessimistes** — et que
l'autonomie WFI (≈ 73 h à 1 Hz) suppose ce build, qui n'est pas celui des mesures de
latence publiées.

Deux limites subsistent, mesurées et non contournées : les µJ du protocole delta portent
la **trame UART** en plus du calcul (seul le lot les sépare), et le lot n'a été balayé que
sur deux cellules — les six autres gardent leur µJ « transaction » de la section 3.
