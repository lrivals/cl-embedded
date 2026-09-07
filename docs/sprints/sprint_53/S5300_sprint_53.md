# Sprint 53 — Campagne énergie élargie : lever les verrous de banc, mesurer les axes système

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 53 |
| **Semaine** | À confirmer — **fenêtre matérielle : tant que le X-NUCLEO-LPM01A est posé sur la carte** |
| **Statut** | 🟡 En cours — S5301 ✅ · S5302 ✅ · **S5304 ✅** (7 cellules, 0 CRC, Gap 3 énergie tranché : l'INT8 ne change pas mesurablement l'énergie/inférence) · **S5303 ✅** (3 fréquences ; **−26,2 % d'énergie** — 214,6 → 170,1 µJ, valeurs recalculées sous la règle d'ajustement unifiée A3, cellule 90 MHz **N/A** en attente de B1 — et −54 % de courant de base de 180 → 45 MHz, Gap 2 tenu 12,8× ; `acqmode dyn` débloqué à 45 MHz) · **S5305 ⚠️ N/A honnête** (voie A insuffisante, chiffres à l'appui) · **S5306 ⚠️ séance 1/3** (N/A honnête, effet dépendant du binaire isolé) ; S5307–S5310 non exécutés. **7 défauts d'outillage corrigés en séance, dont 3 produisaient des résultats crédibles et faux** · **A1–A7 ✅ (2026-09-07, hors banc)** : pilote de sonde sous tests, garde-fou de plausibilité, ajustement unifié, règle A4 arrêtée, contrats requalifiés, provenance de la cadence atteinte, lot versionné. **Reste B1–B7 (banc).** |
| **Priorité globale** | 🔴 Critique — la sonde est disponible *maintenant* ; le Sprint 50 n'a pu produire que le courant moyen de 8 cellules, tout le reste (µJ/inférence, profil par phase, `by_component`, µJ/mise-à-jour) reste `"à mesurer"` |
| **Durée estimée totale** | ~26 h (banc ~9 h · firmware ~5 h · pilotes/analyse ~7 h · figures+notebook ~3 h · tests+docs ~2 h) |
| **Dépendances** | X-NUCLEO-LPM01A **posé** · NUCLEO-F439ZI · `scripts/lpm01a_probe.py` ✅ (validé 100 000/100 000 éch.) · `scripts/run_s50_board_current.py` ✅ · `scripts/energy_capture.py` ✅ · `src/evaluation/autonomy.py` ✅ · pilotes board S38/S48 ✅ |
| **Références** | `docs/sprints/sprint_50/S5008_handoff_mesures.md` (contraintes de banc mesurées) · `docs/context/lpm01a_setup.md` · `docs/context/int8_cost_benefit.md` |

## Contexte et motivation

Le Sprint 50 a posé la sonde et produit un résultat réel mais mince : le **courant moyen**
de 8 cellules (4 modèles × 2 encodages) à 100 Hz, et l'autonomie qui en découle. Tout le
reste reste `"à mesurer"`, pour trois raisons **constatées sur banc**, non supposées
(`S5008_handoff_mesures.md:34-45`) :

1. **Pas de voie de synchronisation** — le LPM01A ne capture aucune entrée numérique. Les
   marqueurs PA8 (`-DENERGY_MARKERS`, `derive_phase_windows`, `segment_by_phase`) sont
   codés et testés depuis le Sprint 33 mais **n'ont jamais pu être exercés**.
2. **Mode dynamique inaccessible** — `acqmode dyn` refuse au-delà de 59 mA, la carte en
   tire 63–75 mA → une moyenne par acquisition, jamais de profil temporel.
3. **Pas de vrai repos** — `pipeline.c:241` attend la trame en scrutation active
   (`while (!(USART3->SR & (1U << 5))) {}`) à 180 MHz, 100 % du temps. Le protocole delta
   ressort négatif et le champ reste N/A avec sa raison.

Deux limites supplémentaires ressortent des données du Sprint 50 mais n'y sont pas
formulées :

4. **Le signal utile est sous le bruit du banc.** À 100 Hz, EWC INT8 vs FP32 vaut
   −0,003 mA pour une dispersion de ±0,03 mA. La question centrale du Gap 3 (« l'INT8
   consomme-t-il moins ? ») est donc **non tranchée** — pas tranchée par la négative : le
   taux d'occupation (~0,5 % à 100 Hz pour EWC) noie l'effet cherché.
5. **L'anomalie −8 mA est peut-être un artefact d'ordre.** Dans
   `run_s50_board_current.py:263-285`, la référence au repos est mesurée *juste après* le
   préchauffage, les cellules *ensuite*. Or le préchauffage relevé décroît
   (62,74 → 54,94 → 54,86 → 54,82 mA) : rien ne prouve que l'établissement soit terminé au
   moment du repos. **L'ordre est confondu avec la condition** — hypothèse jamais testée.

**Objectif du sprint** : lever ces cinq verrous et convertir la disponibilité de la sonde
en résultats système que le mémoire n'a pas encore — le coût énergétique du gate autonome
(Sprint 38), celui du bit-packing (Sprint 48), et l'arbitrage fréquence/autonomie.

## Décisions de cadrage (utilisateur)

- **Modification firmware autorisée** : `__WFI()` sur l'attente UART **et** balayage de
  fréquence SYSCLK — les deux **sous garde `#define`**, build par défaut strictement
  inchangé (précédents `-DENERGY_MARKERS`, `-DINT8_SEGMENT_PROFILE`, `-DEWC_AUTO_UPDATE`).
- **Quatre axes retenus** : politiques de mise à jour P0–P3, sub-INT8 packé/non-packé,
  balayage de cadence (duty-cycle), diagnostic repos/charge + isolation par composant.
- **Budget banc** : une journée ou plus, plusieurs reflashes acceptés.
- **Langue** : français. **Aucun chiffre inventé** (règle CLAUDE.md maintenue).

## Nœud honnête

Ce sprint peut légitimement se terminer avec des champs encore `"à mesurer"` : si S5301
montre que l'anomalie est réelle et que le WFI ne suffit pas, le profil par phase reste
inaccessible. Ce qui n'est **pas** acceptable, c'est de combler par une estimation. Chaque
N/A sort avec une `na_reason` **mesurée**.

Symétriquement, un résultat négatif est un **résultat**, pas un échec : « le bit-packing
coûte plus d'énergie qu'il n'en économise », « l'INT8 ne réduit pas la consommation », «
ralentir le MCU ne gagne rien » sont exactement les conclusions que ce sprint doit être
capable de produire.

Trois états explicites dans les JSON, comme au Sprint 50 : `"à mesurer"` (non mesurable
sur ce banc, avec sa raison), valeur mesurée, `"na"` (non applicable).

## Tâches

### Bloc A — Lever les verrous de banc

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S5301 | **Protocole contre-balancé repos/charge** : l'ordre est aujourd'hui confondu avec la condition ; mesure de repos intercalée entre chaque cellule, 3 répétitions, même session → l'anomalie −8 mA est-elle un effet de charge ou une dérive d'établissement ? **Tranché sur carte réelle** : verdict calculé **`artefact_ordre`**, répliqué sur 2 sessions (05/08 avec 7 repos, 06/08 avec **9 repos ordonnés**). Cause identifiée en S5301b et reproduite une 3ᵉ fois : la carte bascule d'un niveau de repos à l'autre (**−10,06 mA**) **au premier flux reçu** — la référence S50 était sur le niveau haut, ses cellules sur le bas, d'où des µJ/inférence négatifs. Sur une référence correctement établie, les surcoûts redeviennent **positifs et ordonnés selon la latence** (EWC +2,03 · HDC +3,36 mA), et se répliquent à ~0,1 mA près entre sessions malgré ~4 mA d'écart absolu | 🔴 | `docs/sprints/sprint_53/S5301_cadrage_verrous.md`, `scripts/run_s50_board_current.py` (`--interleave-idle`), `src/evaluation/counterbalance.py`, `experiments/exp_S53_counterbalance/` | ✅ |
| S5302 | **`__WFI()` sur l'attente UART** (firmware, gardé `-DUART_WFI_IDLE`) → crée un vrai repos, débloque le protocole delta et le plancher `veille_uA`. **+ extension `-DINFER_BATCH_N`** (boucle par lot) : découple le taux d'occupation de l'UART — **seule voie mesurée** vers un µJ/inférence pour EWC/TinyOL/Maha (cf. pilote S5304 du 2026-08-05). **Mesuré sur carte réelle 2026-08-06** : réveil validé (300 trames @200 Hz, 0 CRC, latence DWT 50 µs inchangée) ; **repos WFI 27,370 mA vs scrutation 49,767 mA = −45,0 %**, les deux dans la même session contre-balancée ; le repos passe SOUS toutes les cellules ⇒ **delta débloqué, 8/8 cellules chiffrées** (134–374 µJ par transaction UART+inférence) ; `veille_uA`/`actif_mA` écrits dans le profil matériel ; **lot `-DINFER_BATCH_N` : EWC 5,08 µJ et Maha 0,437 µJ par inférence (calcul seul), r² = 0,9998**, parité de prédiction 1.000 vs `N=1`, point `N=200` d'EWC écarté par la règle de saturation mesurée (cadence tombée à 66 Hz) ; autonomie duty-cyclée ≈ 73 h à 1 Hz / 2000 mAh (vs 43 h S50, flux continu) | 🔴 | `firmware/stm32f4_blink/src/pipeline.c`, `scripts/run_s53_wfi.py`, `experiments/exp_S53_wfi/` | ✅ |
| S5303 | **Balayage SYSCLK 180/90/45 MHz** (firmware, gardé `-DSYSCLK_MHZ`) → énergie vs fréquence, marge Gap 2 à 45 MHz, et passage éventuel sous le plafond 59 mA du mode dynamique | 🟠 | `firmware/stm32f4_blink/src/hw_info.c`, `scripts/run_s53_freq_sweep.py`, `tests/test_s53_freq.py` | 🟡 Code prêt (3 builds compilés, `.bss` invariant) — banc en attente de carte |

### Bloc B — Méthodes de mesure

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S5304 | **Balayage de cadence + µJ/inférence par régression** : `I_moy = I_base + pente · rate` sur rate ∈ {0…**200**} Hz (**plafond UART ~209 Hz mesuré**, l'ancien 400 Hz était infaisable) ; `I_base` absorbe repos et scrutation active → méthode robuste **même si S5301/S5302 échouent**. **Pilote exécuté 2026-08-05** : méthode validée, HDC INT8 = **173 ± 24 µJ/inférence** par double différence contre le témoin Maha (coût de trame 60,6 µJ). Tranche HDC seul — les cellules rapides exigent `-DINFER_BATCH_N` (S5302). **Code livré et testé hors banc 2026-08-05** (règle de calcul isolée dans `src/evaluation/rate_regression.py`, saturation détectée par cadence ATTEINTE, ordre randomisé, `--refit` sans matériel ; 27 tests PASS dont la reproduction exacte des 20 points du pilote) — **reste l'acquisition** | 🔴 | `scripts/run_s53_rate_sweep.py`, `src/evaluation/rate_regression.py`, `tests/test_s53_rate.py`, `experiments/exp_S53_rate_sweep/` | 🟢 Code prêt, banc en attente |
| S5305 | **Profil temporel par phase** : si S5303 fait passer sous 59 mA, `acqmode dyn` à 100 kSPS ; segmentation sur le courant lui-même (voie A, sans câblage) ou PA8→D7 `--trigsrc` (voie B). **Code livré et testé hors banc 2026-08-05** : voie A ajoutée à côté de `derive_phase_windows` (inchangée), seuil médiane + k·MAD **déduit de la trace**, refus honnête si les créneaux détectés s'écartent de > 5 % de `cadence × durée` ; les deux voies publient la **même** grandeur (surcoût marginal, comparable au delta et à la régression) ; 14 tests PASS, `lpm01a_probe.py` non modifié — **reste l'acquisition** | 🟠 | `scripts/energy_capture.py` (fonctions **ajoutées**), `scripts/run_s53_phase_profile.py`, `tests/test_s53_phase.py`, `experiments/exp_S53_phase_profile/` | 🟢 Code prêt, banc en attente |

### Bloc C — Axes scientifiques

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S5306 | **Énergie des politiques de mise à jour P0–P3** : le gate autonome économise ~97 % des MAJ (S3808) — combien de mA, combien d'heures d'autonomie ? Seule voie vers `energy_uj_per_update`, obtenu par la différence **intra-séance** P1−P0 (même binaire, même trame UART : le coût de trame s'annule exactement). **Code livré et testé hors banc 2026-08-06** (règle de calcul isolée dans `src/evaluation/policy_energy.py` ; deux phases `--prepare`/`--measure` car le flash sous alimentation sonde est instable ; garde anti-mislabel par manifeste de flash ; `update_rate` MESURÉ en séance via les slots V3 réinterprétés exposés à `--dump-samples` ; témoin de séance contre la dérive inter-builds ; 33 tests PASS) — **reste l'acquisition** : **6 flashes** (3 binaires × 2 datasets, corrigé : les poids et `EWC_IN=k` sont propres au dataset) | 🔴 | `scripts/run_s53_policy_energy.py`, `src/evaluation/policy_energy.py`, `tests/test_s53_policy_energy.py`, `experiments/exp_S53_policy_energy/` | 🟢 Code prêt, banc en attente |
| S5307 | **Énergie sub-INT8 packé vs non-packé** : le dépacking (+55 µs, S4804) coûte-t-il plus que la RAM ÷8 n'économise ? | 🟠 | `scripts/run_s53_depth_energy.py`, `experiments/exp_S53_depth_energy/` | 📝 Spec |
| S5308 | **Isolation par composant + autonomie à duty-cycle réaliste** : A/B strict `-DETH_PHY_POWERDOWN`, LED PA5, UART ; puis autonomie `I_repos_WFI + pente · f_usage` | 🟠 | `experiments/exp_S53_components/`, MAJ `configs/energy_campaign_s50.yaml`, `configs/hw_profile_f439zi.yaml` | 📝 Spec |

### Bloc D — Consolidation

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S5309 | **Agrégation + figures + notebook** : agrégateur lecture seule, catalogue `energy_real` étendu (E1/E5 cessent d'être grises si S5304 aboutit), figures d'axe | 🟠 | `scripts/aggregate_s53_energy.py`, `src/figures/catalogs/energy_real.py`, `notebooks/cl_eval/energy_cost/comparison.ipynb` | 📝 Spec |
| S5310 | **Tests + docs + resynchronisation** : tests de la régression, resync des docs en retard sur les données, clôture des `TODO(dorra)` énergie si la calibration est établie | 🟠 | `tests/test_s53_energy.py`, `docs/context/int8_cost_benefit.md`, `docs/triple_gap.md`, roadmap, CLAUDE.md | 📝 Spec |

## Ordre d'exécution recommandé

```
S5301 (contre-balancement repos/charge)      ← 30 min, aucun code neuf, PREMIER GESTE
   │
   ├──► S5304 (balayage de cadence)          ← indépendant du résultat de S5301
   │        │
   │        ▼
   ├──► S5302 (WFI) ──► reprise repos/delta ──► S5308 (autonomie duty-cycle)
   │
   └──► S5303 (SYSCLK) ──► S5305 (acqmode dyn, si < 59 mA)
                │
                ▼
        S5306 (politiques P0–P3) ──► S5307 (sub-INT8 packé/non-packé)
                │
                ▼
        S5309 (agrégation + figures) ──► S5310 (tests + docs + graphify)
```

**S5304 est délibérément placé tôt et sans dépendance** : c'est la seule tâche qui produit
des µJ/inférence *quoi qu'il arrive* aux verrous firmware.

## Ordre de banc conseillé (une journée)

| # | Tâche | Reflash ? | Durée |
|---|-------|:---------:|-------|
| 1 | S5301 contre-balancement repos/charge | non | 30 min |
| 2 | S5304 balayage de cadence, 8 cellules | non | 2 h |
| 3 | S5302 build WFI + reprise repos/delta | oui (1) | 1 h |
| 4 | S5303 balayage SYSCLK 180/90/45 | oui (3) | 1 h 30 |
| 5 | S5305 `acqmode dyn` si < 59 mA | non | 1 h |
| 6 | S5306 politiques P0–P3 | oui (6) | 2 h |
| 7 | S5307 sub-INT8 packé/non-packé | oui (6) | 1 h 30 |
| 8 | S5308 isolation par composant | oui (1) | 30 min |

## Rappels de banc — contraintes mesurées, à ne pas recombattre

Source : `docs/sprints/sprint_50/S5008_handoff_mesures.md:34-45`, `docs/context/lpm01a_setup.md`.

- La sonde **n'alimente la carte que pendant une acquisition** → sinon reboot en boucle ;
  utiliser `--hold-run`.
- Le **flash sous alimentation sonde est instable** → flasher avec **JP5 en place**, puis
  basculer (manipulation manuelle de cavalier à chaque reflash).
- **Jeter la première acquisition** de chaque session (biais ~+8 mA) → `lp.warmup()`.
- Port de la sonde via `/dev/serial/by-id/*PowerShield*`, **jamais `/dev/ttyACM0`**
  (occupé par l'ST-LINK).
- Le courant mesuré est celui du **rail VDD_MCU entier** (MCU + PHY + périphériques), pas
  le cœur seul.
- `acqmode dyn` échoue au-delà de 59 mA — ce n'est **pas** un réglage à trouver, c'est une
  limite de la sonde. S5303 est la seule voie de contournement.

## Règles d'honnêteté (non négociables)

- Grandeur non mesurée → chaîne littérale `"à mesurer"` **+ `na_reason` mesurée**. Jamais
  0, jamais une estimation.
- `energy_uj_per_inference` obtenu par **régression** (S5304) porte un `method` distinct de
  celui obtenu par **delta** (S5302). **Ne jamais les fusionner dans un même champ.**
- `run_s50_energy.py:load_measured_cell` protège déjà les cellules réelles contre un
  écrasement par placeholder — le conserver.
- Toute mesure d'énergie est précédée d'une **non-régression du flux** : 300 échantillons,
  0 erreur CRC, parité board↔PC inchangée.

## Sources de données (Sprint 53)

| Source | Rôle |
|--------|------|
| `scripts/lpm01a_probe.py` | Pilotage headless de la sonde (ASCII/binaire, `warmup`, `hold_run`) |
| `experiments/exp_S50_energy/` | Courant moyen de référence (8 cellules, 100 Hz) — point de comparaison |
| `experiments/exp_S50_int8_latency/ewc.json` | Cycles DWT par segment — contrôle de cohérence croisée |
| `experiments/exp_S38_summary.json` | `economy_table` du gate (MAJ économisées) — dénominateur de S5306 |
| `experiments/exp_S48_summary.json` | `.bss` packé/non-packé et latence de dépacking — dénominateur de S5307 |
| `configs/hw_profile_f439zi.yaml` | Capacités batterie, calibration sonde, `actif_mA`/`veille_uA` (à remplir) |

## Livrables

1. `docs/sprints/sprint_53/S5301`–`S5310` — specs et comptes rendus de mesure.
2. Firmware : `-DUART_WFI_IDLE` et `-DSYSCLK_MHZ` gardés, **`.bss` défaut invariant
   105 036 B**, `make test` **145 / 0 échec**.
3. `experiments/exp_S53_rate_sweep/` — µJ/inférence par régression, pente vs latence DWT.
4. `experiments/exp_S53_phase_profile/` — profil par phase (ou N/A avec raison mesurée).
5. `experiments/exp_S53_policy_energy/` — énergie des 4 politiques + `energy_uj_per_update`.
6. `experiments/exp_S53_depth_energy/` — énergie packé vs non-packé.
7. `experiments/exp_S53_components/` — `by_component` renseigné + autonomie duty-cycle.
8. `exp_S53_summary.json` + figures `energy_real` étendues + notebook.
9. `tests/test_s53_energy.py` + MAJ `roadmap_phase2.md`, `triple_gap.md`, `CLAUDE.md`.

---

## Reste à faire — issu de la séance de banc du 2026-09-07

### A. Corrections de code (hors banc, sans matériel)

| # | Correction | Fichier | Pourquoi |
|---|---|---|---|
| A1 ✅ | **Fonctions série du pilote de sonde sous tests** : `command`, `take_control`, `voltage_v`, `power_on`, `hold_run`, `capture` | `tests/test_lpm01a_probe.py` (+21 tests, `FakeSerial`) | Les 4 défauts de la séance étaient là, aucun test ne les exerçait. Le cas MESURÉ « `volt get` → `handled` » est désormais un test, comme la reprise unique de `htc` et l'arrêt d'acquisition quand la commande hôte échoue. Suite ramenée de 68 s à 1,4 s (temporisations neutralisées). |
| A2 ✅ | **Garde-fou de plausibilité du courant** (`MIN_PLAUSIBLE_CURRENT_A = 1 mA`, échappatoire explicite `min_current_a=None`) | `scripts/lpm01a_probe.py::capture` | Placé au **point d'étranglement unique** — toutes les acquisitions du dépôt passent par `capture()` — et non dans chaque pilote. Seuil calé sur la mesure : 18,2 mA est le courant le plus bas jamais relevé (repos 45 MHz), ×18 au-dessus du seuil. Les appelants qui consignent déjà un échec écrivent leur `na_reason` sans changement. |
| A3 ✅ | **Ajustement unifié** : `_linear_fit` supprimé, cellule dérivée par `rate_regression.fit_cell` ; `--refit` ajouté | `scripts/run_s53_freq_sweep.py` | Le pilote transmettait des couples `(x, y)` **sans les écarts-types** : il fittait non pondéré là où S5304 pondère, et deux chiffres coexistaient pour la même mesure. Après recalcul (sans remesure) : 45 MHz 167,6 → **170,1 µJ**, 180 MHz 199,7 → **214,6 µJ**, 90 MHz → **N/A** (r² 0,919 non pondéré → 0,783 pondéré). Collision de clé corrigée au passage (`dwt_latency_us_p50` scalaire vs dictionnaire par modèle). |
| A4 ✅ | **Règle arrêtée** : une DIFFÉRENCE de pentes exige la linéarité de chaque cellule (`r² ≥ 0,9`), **plus** leur significativité individuelle | `src/evaluation/policy_energy.py` (`_lineaire` vs `_publiable`) | Le bruit commun aux deux cellules (binaire, trame UART, séance, ordonnée à l'origine) s'annule dans l'écart : une différence peut être séparable du bruit là où aucune des deux pentes ne l'est seule — le régime exact du coût d'une mise à jour CL. Le test `Δ > 2 σ` reste seul juge et écarte toujours les différences négatives. **Décidée avant la reprise des séances 2 et 3.** |
| A5 ✅ | **Lot S53 versionné** : pilote de sonde, 5 pilotes S53, diagnostic, 6 fichiers de tests, doc de banc, mesures | dépôt | Les correctifs de la séance vivaient dans un fichier jamais suivi par git, et les pilotes qui l'importent non plus. |
| A6 ✅ | **Contrats explicités** : `f1_binary_contract`/`n_classes`/`f1_na_reason` ; trois modes d'échec distingués et `stderr` remonté | `scripts/sensor_stream.py`, `scripts/run_s50_board_current.py::stream_in_acquisition` | Sur des labels non binaires, `compute_fault_f1` ne rendait pas un nombre discutable : il **levait**, faisant tomber tout le flux. Les champs F1 sortent désormais à `null` avec leur raison, accuracy et latences restant mesurées. Côté flux, « rien produit » / « produit puis échoué » / « illisible » ne demandent pas le même geste au banc et étaient indiscernables (`stderr` dans `DEVNULL`) ; le garde-fou est mutualisé entre les deux pilotes. |
| A7 ✅ | **Provenance de la cadence atteinte** : `achieved_rate_source` ∈ {mesuré, inféré du plafond mesuré, non mesuré} ; option de banc `--achieved-at-each-rate` | `src/evaluation/rate_regression.py::normalize_achieved_source`, les deux balayages | Les pilotes recopiaient la **consigne** dans un champ de mesure. La règle est rétroappliquée aux cellules déjà écrites : **sans aucun effet sur les ajustements** (la saturation ignore déjà les points sans cadence atteinte), seule la traçabilité change. La mesure point par point reste à faire au banc (avec B1/B2). |

### B. Mesures à refaire ou à compléter (avec carte + sonde)

| # | Mesure | Coût | Ce qu'elle débloque |
|---|---|---|---|
| B1 | **Rejouer la cellule 90 MHz** avec le code final (propagation du plafond, erreur-type) | 1 flash, ~15 min | Sous la règle d'ajustement unifiée (A3), cette cellule est **N/A** : r²=0,783 pondéré, pente 41,6 ± 12,6 µA/Hz — non séparable du bruit. C'est la seule des trois produite par une version antérieure du pilote, et elle porte le point milieu de la tendance S5303. |
| B2 | **Rejouer `hdc_fp32`** de S5304, sortie en N/A (r² 0,885) | ~10 min, sans flash | Complète la grille S5304 à 7/7 cellules chiffrées. |
| B3 | **Isoler proprement l'effet « build S38 »** : comparer `EWC_IN=4` vs `EWC_IN=5` **à en-têtes de poids identiques** | 2 flashes, ~30 min | L'isolation actuelle fait varier dimensions **et** poids (`.text` 46 680 vs 51 976 B). C'est le verrou de S5306. |
| B4 | **Reprendre S5306** une fois B3 comprise : séances 2 et 3 (gated), puis Pronostia | 5 flashes, ~2 h | `energy_uj_per_update`, le champ N/A depuis le Sprint 50, plus le verdict du gate. |
| B5 | **S5305 voie B** — soudure PA8 → Arduino D7, `--trigsrc d7` | 1 soudure | Seule voie restante vers le 3ᵉ estimateur : la voie A est mesurément insuffisante (pas de palier de détection). **Décision matérielle utilisateur.** |
| B6 | **Caractériser le seuil réel du mode dynamique** (68,6 mA passe, 69,6 mA échoue) | ~30 min | Permet de dire ce qui déclenche l'arrêt — pic, moyenne glissante, ou durée au-dessus d'un seuil. |
| B7 | **S5307 / S5308** non entamés | 6 + 2 flashes | Énergie sub-INT8 packé/non-packé ; isolation par composant et autonomie duty-cycle. |

### C. Ordre conseillé pour la prochaine séance

A1 → A2 → A3 (hors banc, ~2 h) · puis B3 → B1 → B2 (banc, ~1 h) · puis B4 si B3 conclut.
