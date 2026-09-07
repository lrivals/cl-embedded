# S5301 — Protocole contre-balancé repos/charge : l'anomalie −8 mA est-elle un artefact d'ordre ?

| Champ | Valeur |
|-------|--------|
| **Sprint** | 53 |
| **Priorité** | 🔴 Critique — **premier geste de la journée de banc** |
| **Statut** | ✅ Implémenté et mesuré sur carte réelle (sessions 2026-08-05 et 2026-08-06) |
| **Durée estimée** | 1 h (dont 30 min de banc) |
| **Dépendances** | Sonde posée · `scripts/run_s50_board_current.py` ✅ · `scripts/lpm01a_probe.py` ✅ |
| **Fichiers cibles** | `scripts/run_s50_board_current.py` (option `--interleave-idle`) · `experiments/exp_S53_counterbalance/` |
| **Références** | `docs/sprints/sprint_50/S5008_handoff_mesures.md` · `run_s50_board_current.py:13-26` (constat) |

## Contexte

Le Sprint 50 a constaté, et documenté honnêtement, que **la référence au repos
(54,81 ± 0,11 mA) est plus haute que TOUS les régimes de flux mesurés (46,46 à
51,26 mA)** — y compris Mahalanobis, dont l'inférence n'occupe que ~0,05 % du temps. Un
écart de −8 mA que le taux d'occupation ne peut pas expliquer. Le pilote conclut, avec
raison : « on ne l'explique pas, on la constate », et laisse `energy_uj_per_inference` à
`"à mesurer"` avec une `na_reason` mesurée.

**Ce que le Sprint 50 n'a pas testé** : dans `run_s50_board_current.py:263-285`, la
séquence est `warmup → repos ×3 → cellule₁ ×3 → cellule₂ ×3 → …`. La référence au repos
est donc systématiquement **en tête de session**, les cellules **après**.
**L'ordre est confondu avec la condition.**

Or le préchauffage lui-même montre une dérive d'établissement qui n'est pas terminée en
une acquisition : 62,74 → 54,94 → 54,86 → 54,82 mA. Le pilote jette la première
acquisition (`lp.warmup`), ce qui traite le saut initial, mais rien ne prouve que la
décroissance soit achevée au moment où les trois acquisitions de repos sont prises.

**Hypothèse à tester** : l'écart −8 mA n'est pas un effet de charge, c'est la queue de la
dérive thermique/d'établissement de la session. Si elle est vraie, le protocole delta est
réhabilité tel quel et `scripts/run_s50_energy_delta.py` (247 lignes, 13 tests, **jamais
exécuté sur carte**) devient exploitable sans une ligne de code neuve.

## Spec

### 1. Protocole contre-balancé

Séquence dans une **session unique**, après `lp.warmup()` :

```
repos → EWC → repos → HDC → repos → EWC → repos → HDC → repos
```

- **3 répétitions** de la séquence complète.
- Deux modèles aux extrémités du spectre de latence (EWC 50 µs, HDC 2095 µs) : si l'effet
  était un effet de charge, l'écart EWC↔HDC devrait rester stable alors que le repos
  dérive.
- Cadence, fenêtre et tension identiques à la campagne S50 (100 Hz, 10 s, 3,3 V) pour que
  la comparaison avec `exp_S50_energy/` soit directe.
- Chaque acquisition de repos conserve **son rang dans la session** (`session_index`) —
  c'est la variable explicative testée.

### 2. Critère de décision

| Observation | Conclusion | Conséquence |
|-------------|-----------|-------------|
| Les repos successifs **décroissent** vers ~46 mA et le dernier repos ≈ les cellules | **Artefact d'ordre confirmé** | Le protocole delta est réhabilité ; relancer `run_s50_energy_delta.py` tel quel ; corriger la `na_reason` des 8 cellules S50 |
| Les repos restent **stables à ~54,8 mA** quel que soit le rang | **Effet réel**, cause non établie | S5302 (WFI) tranche : si le repos WFI s'effondre, la scrutation active était bien la cause |
| Les repos **dérivent sans converger** vers les cellules | Dérive thermique partielle | Rapporter la pente de dérive ; imposer un préchauffage plus long (`--warmup-repeats`) à toutes les campagnes suivantes |

**Aucun de ces trois résultats n'est un échec.** Les trois sont des conclusions
publiables sur la méthodologie de banc.

### 3. Implémentation

Ajouter à `scripts/run_s50_board_current.py` :

- `--interleave-idle` : intercale une mesure de repos avant **et** après chaque cellule au
  lieu du bloc de repos unique en tête.
- `--warmup-repeats N` (défaut 1) : nombre d'acquisitions de préchauffage jetées.
- Chaque mesure de repos est enregistrée avec son `session_index` dans
  `current_measurement.i_idle_runs_a` (liste d'objets `{session_index, i_a}`).

**Contrainte de compatibilité** : le champ `i_idle_a` existant **garde sa sémantique**
(moyenne des repos de la session) — les 8 cellules `exp_S50_energy/` déjà écrites en
dépendent, ainsi que `delta_vs_idle_a`. Ne pas le redéfinir ; ajouter à côté.

Réutiliser `measure_current`, `stream_command`, `build_cell`, `lp.warmup` — **ne rien
réécrire**.

### 4. Sortie

`experiments/exp_S53_counterbalance/counterbalance.json` :

```json
{
  "protocol": "repos et cellules alternés, 3 répétitions, session unique",
  "sequence": [{"session_index": 0, "condition": "idle|ewc|hdc", "i_a": 0.0}],
  "idle_by_rank": [{"session_index": 0, "i_a": 0.0}],
  "drift_slope_ma_per_acquisition": 0.0,
  "verdict": "artefact_ordre | effet_reel | derive_partielle",
  "verdict_rationale": "…critère chiffré appliqué…",
  "rate_hz": 100.0, "window_s": 10.0, "n_repeats": 3,
  "warmup_discarded_a": [0.0]
}
```

`verdict` est **calculé**, pas saisi : règle explicite dans le code (par exemple, artefact
d'ordre si `|dernier_repos − moyenne_cellules| < 3 × dispersion_banc`).

## Critères d'acceptation

- [x] La séquence contre-balancée tourne en une session, 0 erreur CRC sur les flux
      (150/150 trames par cellule, vérifiées séparément — le pilote redirige la sortie des
      flux vers `DEVNULL` pendant l'acquisition).
- [x] `idle_by_rank` contient au moins 9 points ordonnés (**9**, session du 2026-08-06).
- [x] `verdict` est dérivé d'une règle codée et testée, jamais écrit à la main
      (`src/evaluation/counterbalance.py`, couvert par `tests/test_s53_bench.py`).
- [x] `i_idle_a` et `delta_vs_idle_a` conservent leur sémantique S50 (non-régression des
      8 cellules existantes — la session écrit dans `exp_S53_counterbalance/`, jamais dans
      `exp_S50_energy/`).
- [x] La `na_reason` de `energy_uj_per_inference` est **mise à jour** dans les cellules S50
      pour refléter le verdict (repropagée par `scripts/refresh_s50_na_reason.py`, jamais à
      la main ; le verdict étant inchangé en 2ᵉ session, rien à repropager).

## Ce que cette tâche ne fait pas

- Elle ne modifie **pas** le firmware (c'est S5302).
- Elle ne produit **pas** de µJ/inférence — elle décide seulement si la voie delta est
  praticable.

---

## Éléments mesurés apportés le 2026-08-05 (avant exécution de la tâche)

**Le biais de première acquisition est établi et reproductible** — cinq sessions
indépendantes, carte au repos, conditions strictement identiques :

| session | capture 1 | captures suivantes |
|---|---|---|
| témoin S5008 | 62,74 | 54,94 / 54,86 / 54,82 |
| campagne 1 | 62,78 | 54,81 ± 0,11 |
| campagne 2 | 62,92 | 54,78 ± 0,13 |
| diagnostic A | 63,60 | 54,54 / 54,43 / 54,40 |
| diagnostic B | 63,87 | 54,52 / 54,32 / 54,29 |

Biais : **+9,1 à +9,5 mA**, toujours sur la première acquisition, jamais ensuite. La
dispersion du régime établi est de ±0,06 mA — le banc est donc largement assez fin pour
séparer les modèles une fois le rebut écarté.

**Hypothèse réfutée — la calibration de la sonde n'y est pour rien.** Les offsets d'étage
analogique annoncés « prev: 0 mV » par l'autotest suggéraient une calibration non appliquée
à la première acquisition. Deux sessions comparées, l'une précédée d'un `calib`, l'autre
non : biais **+9,14 mA sans `calib`** contre **+9,49 mA avec**. La commande `calib`
(`lpm01a_probe.py:640`, jamais utilisée jusque-là) est donc **sans effet** sur ce biais.
Inutile de la réessayer.

**Piste discriminante restante, peu coûteuse** : faire varier la **durée** de la première
capture (5 / 10 / 30 s). Un transitoire de constante de temps τ voit son excès moyen
diminuer quand la fenêtre s'allonge ; un offset fixe appliqué à la première acquisition ne
bouge pas. Ce seul test sépare les deux familles de causes.

**Conséquence pour cette tâche** : le contre-balancement reste indispensable, mais le
verdict à produire n'est plus « le biais existe-t-il ? » (il est établi) — c'est « l'écart
repos↔charge de −8 mA en est-il une continuation, ou un effet de charge réel ? ». Le
protocole d'entrelacement spécifié ici y répond directement, à condition que le repos soit
mesuré **entre** chaque cellule et non une seule fois en tête de session.

---

## Compte rendu S5301b — « de quel repos parle-t-on ? » (2026-08-05)

**Verrou levé.** La reprise du protocole delta était bloquée par un écart non expliqué
entre deux mesures de repos prises à quelques minutes d'intervalle sur le même banc et le
même firmware (carte jamais streamée d'un côté, repos intercalé après un flux interrompu de
l'autre). Tant qu'il subsistait, on ne savait pas quelle référence le contre-balancement
avait comparée aux cellules.

**Hypothèse de travail, réfutée.** `measure_current` interrompt le flux par
`proc.terminate()` : le port se ferme, DTR/RTS retombent, et la carte aurait pu rester tenue
à l'arrêt. Mesuré (`scripts/diag_s53_idle_states.py`, 4 états × cycles répétés, rang de
chaque acquisition consigné) : le repos après flux interrompu est **indiscernable** du repos
port fermé (écart ≤ 0,1 mA, sous 0,1 × le surcoût de charge), et la carte répond **20/20
trames** immédiatement après, à chaque cycle. L'ouverture seule du port ne déplace rien non
plus. → verdict calculé `repos_coherent`
(`experiments/exp_S53_counterbalance/idle_states_diagnostic.json`).

**Cause réelle, identifiée.** En coupant l'alimentation cible par la sonde puis en la
rétablissant, la carte se retrouve à un **second niveau de repos, stable et ~10 mA plus
haut** : 49,74–49,90 mA tant qu'elle n'a reçu aucune trame — y compris port ouvert et après
une impulsion DTR — puis **39,74–40,08 mA dès le premier flux reçu**, définitivement. Pas
en 10 mA (`step_first_stream_a`), séries plates avant et après, reproductible sur trois
sessions (`experiments/exp_S53_counterbalance/idle_states_power_cycle.json`). Ce n'est donc
ni un établissement thermique, ni la sonde, ni l'état des lignes de l'hôte : c'est le
**traitement de vraies trames** qui fait basculer la carte. Le mécanisme côté carte reste
non identifié — question distincte, non instruite ici.

**Ce que cela change.**

1. Le verdict S5301 `artefact_ordre` est **confirmé et expliqué** : la « dérive
   d'établissement » du repos (54,41 mA en tête de session → 43,96 mA ensuite, cf.
   `counterbalance.json`) est en réalité ce pas unique, franchi au premier flux — donc à la
   première cellule mesurée. La référence du Sprint 50 était prise sur le niveau haut, les
   cellules sur le niveau bas : d'où des µJ/inférence négatifs.
2. `energy_na_reason` des 8 cellules `exp_S50_energy/` est repropagée par
   `scripts/refresh_s50_na_reason.py` avec le **mécanisme nommé**, à la place de la seule
   « queue d'établissement ».
3. Règle de protocole ajoutée à `docs/context/lpm01a_setup.md` § 4bis.3 : référence au repos
   **après au moins un flux**, et **aucune comparaison de courants absolus entre sessions**
   (les niveaux relevés diffèrent d'une session à l'autre) — contrainte à porter par S5302,
   dont la comparaison veille/scrutation exige deux firmwares, donc deux sessions.

**Correctifs de règle** (`src/evaluation/counterbalance.py`, tous couverts par
`tests/test_s53_bench.py`) : `idle_state_verdict` refuse de conclure quand aucun état n'est
répété (`dispersion_non_estimee` — sinon la dispersion de repli rendait n'importe quel écart
significatif à des centaines de σ), et un écart significatif mais inférieur à
`MARGINAL_FRACTION` du surcoût de charge **mesuré dans la même session** est reporté comme
résidu négligeable au lieu d'être promu en verdict.

**Livrables** : `scripts/diag_s53_idle_states.py`, `counterbalance.idle_state_verdict` /
`pooled_dispersion_a`, 2 JSON de diagnostic, `tests/test_s53_bench.py` (17 PASS),
§ 4bis.3 du guide de banc. Aucune mesure antérieure modifiée.

---

## Compte rendu S5301c — seconde session, 9 repos ordonnés (2026-08-06)

**Pourquoi une seconde session.** La session du 2026-08-05 concluait déjà `artefact_ordre`,
mais avec `--repeats 3` × 2 cellules elle ne produisait que **7** repos ordonnés, là où les
critères d'acceptation en exigent **9**. `--repeats 4` les fournit, et la reprise vaut
**réplication indépendante** : le verdict est intra-session, donc reproductible malgré
l'interdiction de comparer des courants absolus entre sessions (§ 4bis.3, règle 2).

**Conditions.** Firmware **reconstruit et reflashé par défaut** avant la session
(`make clean && make all && make flash`, **aucun `EXTRA_CFLAGS`**) — la carte portait un
build S5302/S5303/S5305 postérieur à la première session. `.bss` = **105 036 B**, invariant
du projet : le firmware mesuré est bien celui de la campagne S50. Cadence, fenêtre et
tension inchangées (100 Hz, 10 s, 3,263 V relus sur la sonde), 17 acquisitions, préchauffage
jeté à 60,170 mA.

| Rang | Condition | I mesuré |
|---|---|---|
| #0 | repos, **avant tout flux** | **50,130 mA** |
| #2 … #16 | repos, après le premier flux (8 points) | 39,87 – 40,21 mA, **moyenne 40,073 mA** (dispersion 0,112 mA) |
| — | `ewc_fp32` (4 rép.) | 42,100 ± 0,055 mA |
| — | `hdc_fp32` (4 rép.) | 43,428 ± 0,106 mA |

**Verdict calculé : `artefact_ordre`** — écart final repos↔cellules **−2,894 mA = −25,9 σ**,
dérive du repos sur la session **−5,396 mA = 48,2 σ** (pente −0,3372 mA/acquisition sur
9 points, dispersion du régime établi 0,112 mA sur 8 points, 1 écarté comme queue
d'établissement).

**Ce que la session ajoute au S5301b.** Le pas est retrouvé tel quel : le repos chute de
**−10,06 mA** entre l'acquisition #0 (carte jamais streamée depuis que le câblage de la
sonde a coupé `VDD_MCU`) et le régime établi — contre −10,44 mA le 05/08 et −9,8 mA au
diagnostic par coupure d'alimentation. Le mécanisme « bascule au premier flux reçu » est
donc reproduit une troisième fois, sur un firmware refait à neuf.

**Réplication des écarts intra-session, malgré des niveaux absolus différents** — c'est la
règle 2 du § 4bis.3 vérifiée par la mesure :

| Grandeur | 2026-08-05 | 2026-08-06 |
|---|---|---|
| repos établi (absolu, **non comparable**) | 43,970 mA | 40,073 mA |
| surcoût `ewc_fp32` vs repos établi | +2,167 mA | **+2,027 mA** |
| surcoût `hdc_fp32` vs repos établi | +3,457 mA | **+3,355 mA** |
| pas repos #0 → régime établi | +10,440 mA | **+10,057 mA** |

Les niveaux absolus se déplacent de ~4 mA d'une session à l'autre ; les **différences**
tiennent à ~0,1 mA. Les surcoûts restent **positifs** et **ordonnés selon la latence**
(EWC 50 µs < HDC 585 µs, latences DWT relevées pendant la vérification CRC) : sur une
référence de repos correctement établie, il n'y a plus aucune anomalie de signe.

**Intégrité du flux** : 150/150 trames et **0 erreur CRC** sur chacune des deux cellules,
vérifiées par une passe séparée (le pilote redirige la sortie des flux vers `DEVNULL`
pendant l'acquisition, donc les CRC n'y sont pas observables). Cette vérification s'exécute
sous `lpm01a_probe.py --hold-run` : `JP5` étant retiré, la cible n'est alimentée que pendant
une acquisition maintenue.

**Livrables** : `experiments/exp_S53_counterbalance/counterbalance.json` (session du 06/08,
9 repos) ; la session du 05/08 est **conservée** sous `counterbalance_2026-08-05.json`.
Verdict inchangé ⇒ aucune repropagation de `energy_na_reason` nécessaire. Aucun code
modifié : `--interleave-idle`, `--warmup-repeats` et la règle de verdict étaient déjà en
place et ont été réutilisés tels quels.

**Observation hors périmètre** (aucune action ici) : `sensor_stream.py::_compute_stats`
lève `ValueError: Target is multiclass but average='binary'` en fin de flux sur certains
sous-échantillons, avant l'écriture de `--output`. Sans effet sur cette tâche — la campagne
interrompt les flux avant cette étape — mais à traiter quand un pilote aura besoin du JSON
de statistiques.
