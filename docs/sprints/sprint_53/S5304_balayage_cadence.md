# S5304 — Balayage de cadence : µJ/inférence par régression

| Champ | Valeur |
|-------|--------|
| **Sprint** | 53 |
| **Priorité** | 🔴 Critique — **la tâche qui tranche le Gap 3 énergie**, et sans firmware |
| **Statut** | ✅ **Mesuré sur carte réelle (2026-09-07)** — 7 cellules, 0 erreur CRC, verdict Gap 3 énergie prononcé |
| **Durée estimée** | 4 h (2 h pilote ✅ + 2 h banc ⏳) |
| **Dépendances** | Sonde posée ⏳ · `scripts/run_s50_board_current.py` ✅ · `scripts/sensor_stream.py --rate-hz` ✅ |
| **Fichiers cibles** | `scripts/run_s53_rate_sweep.py` ✅ · `src/evaluation/rate_regression.py` ✅ · `tests/test_s53_rate.py` ✅ · `experiments/exp_S53_rate_sweep/` ⏳ |
| **Références** | `run_s50_board_current.py:113-131` · `sensor_stream.py:633-702` (balayage de saturation existant) |

## Contexte

Deux problèmes bloquent aujourd'hui les µJ/inférence, et cette tâche les contourne tous
les deux **sans toucher au firmware ni au câblage** :

- **Pas de référence de repos fiable** (anomalie −8 mA, S5301) → le protocole delta est
  inexploitable.
- **Le signal est sous le bruit** : à 100 Hz, EWC INT8 vs FP32 vaut −0,003 mA pour une
  dispersion de ±0,03 mA. Le taux d'occupation d'EWC à 100 Hz est de ~0,5 % : 99,5 % de la
  mesure est du bruit de fond commun.

L'idée : au lieu de soustraire un repos douteux, **faire varier la cadence et régresser**.

## Spec

### 1. Méthode

Pour chaque cellule (modèle × encodage), mesurer `I_moy` à
`rate ∈ {0, 10, 25, 50, 100, 150, 200}` Hz, puis ajuster

```
I_moy(rate) = I_base + pente · rate
```

d'où `E_par_inférence = pente · V` (en µJ, avec `pente` en A·s et `V` = 3,3 V).

**Pourquoi c'est robuste** : `I_base` absorbe *intégralement* le repos, la scrutation
active, le trafic USB de l'hôte et l'anomalie de S5301. Seule la **pente** porte le coût
marginal d'une inférence. La méthode reste valide même si S5301 conclut à un effet réel et
même si S5302 échoue.

### 2. Les deux raffinements qui font la valeur scientifique

**a. Séparer le coût UART du coût de calcul.**
La pente contient le coût de la trame UART (identique pour tous les modèles) *plus* le
coût de calcul. Donc :

- `pente(HDC) − pente(Maha)` isole le calcul pur (2095 µs vs 5 µs de latence connue).
- Mieux : régresser **`pente` contre la latence DWT** sur les ~8 cellules. La droite donne
  le **coût énergétique par µs de calcul** (sa pente) et le **coût énergétique d'une trame
  UART** (son ordonnée à l'origine). Deux grandeurs que le projet n'a jamais mesurées, sans
  aucune modification matérielle.

**b. Amplifier l'effet INT8 — et pourquoi la cadence seule n'y suffit pas.**

> **Corrigé le 2026-08-05 par un pilote de banc.** La borne haute de 400 Hz de la version
> initiale de cette spec est **infaisable**, et le plafond n'est pas celui du modèle.

Le plafond est **l'UART, pas le calcul** : une transaction fait 32 B (requête) + 23 B
(réponse V3) = 55 B, soit 4,77 ms à 115200 bauds → **~209 Hz au maximum** (débit maximal
relevé : 140–218 inf/s selon les cellules). Au-delà, le flux tourne moins vite que la
consigne, l'axe des cadences **sature en silence** et la pente est sous-estimée — d'où la
borne ramenée à 200 Hz ci-dessus. La lecture « HDC sature vers ~475 Hz » de la version
initiale confondait le plafond de calcul (1/2095 µs) avec le plafond réel de transport.

Taux d'occupation atteignables à 209 Hz :

| modèle | latence | occupation |
|---|---|---|
| HDC INT8 | 1958 µs | 41 % — mesurable |
| TinyOL | 85 µs | 1,8 % |
| EWC | 50 µs | 1,0 % |
| Maha | 5 µs | 0,1 % |

**Conclusion opérationnelle** : le balayage de cadence tranche le Gap 3 pour HDC seulement.
Pour EWC, TinyOL et Mahalanobis — précisément les cellules que `S5300` §4 déclare « sous le
bruit » — il faut découpler l'occupation de l'UART, c'est-à-dire la **boucle par lot
`-DINFER_BATCH_N`** spécifiée en extension de [`S5302`](S5302_wfi_repos_reel.md). Cette
tâche reste néanmoins exécutable et utile **sans** elle : elle produit HDC, et elle produit
le coût de trame qui sert de référence à tout le reste.

### 2bis. Pilote déjà exécuté (2026-08-05) — méthode validée, chiffres de départ

Deux cellules, 2 répétitions × 5 cadences (25→125 Hz), ordre **entrelacé** (sens inversé à
chaque répétition, pour qu'une dérive de session ne corrèle pas avec la cadence) :

| cellule | latence | pente → µJ/inférence | R² | I_base |
|---|---|---|---|---|
| HDC INT8 | 1958 µs | 233,9 ± 23,7 | 0,924 | 43,85 mA |
| Mahalanobis (témoin) | 5 µs | 60,6 ± 4,6 | 0,956 | 44,19 mA |

Le témoin, dont le calcul vaut ~0,4 µJ, sort à 60,6 µJ : **c'est le coût d'une trame UART**,
et il valide le raffinement (a) — la double différence isole le calcul :

> **HDC INT8 : 233,9 − 60,6 = 173 ± 24 µJ/inférence de calcul pur.**

Contrôle de plausibilité : 173 µJ sur 1958 µs = 88 mW de surcroît, soit ~27 mA de plus
pendant le calcul à 3,26 V — cohérent pour un Cortex-M4 parcourant des hypervecteurs en
SRAM. **Premier µJ/inférence du projet obtenu sans référence de repos valide**, ce qui était
l'objectif de la méthode.

Réserve : l'écart entre répétitions à 25 Hz (47,11 vs 44,09 mA) trahit une dérive lente de
session que l'entrelacement atténue sans l'annuler — c'est elle qui domine le ±24. Passer à
3 répétitions et encadrer par un repos (S5301) doit la réduire.

### 3. Implémentation

`scripts/run_s53_rate_sweep.py`, réutilisant **sans les réécrire** :
`measure_current`, `stream_command`, `lp.warmup`, `_load_calibration` de
`run_s50_board_current.py`.

- `--rates 0,10,25,50,100,200,400` (liste modifiable).
- `--repeats 3` par point → écart-type, donc barres d'erreur sur la régression.
- **Ordre des points randomisé** (`--shuffle-order`, défaut activé) : c'est la leçon de
  S5301 — ne jamais laisser l'ordre corrélé à la condition.
- Le point `rate = 0` est le repos ; il alimente aussi S5308 (UART inactif).
- Régression par moindres carrés pondérés (poids = 1/σ²), avec `r²` rapporté.

### 4. Sortie

`experiments/exp_S53_rate_sweep/{model}_{encoding}.json` :

```json
{
  "model": "ewc", "encoding": "int8",
  "method": "régression I(rate)",
  "points": [{"rate_hz": 0, "i_mean_a": 0.0, "i_std_a": 0.0, "n_repeats": 3,
              "session_index": 0}],
  "slope_ua_per_hz": 0.0,
  "slope_std_ua_per_hz": 0.0,
  "intercept_ma": 0.0,
  "r2": 0.0,
  "energy_uj_per_inference": 0.0,
  "duty_cycle_at_max_rate": 0.0,
  "saturation_rate_hz": null,
  "tension_v": 3.3,
  "dwt_latency_us_p50": 0.0
}
```

Plus `experiments/exp_S53_rate_sweep/slope_vs_latency.json` : la régression de second
niveau (pente vs latence DWT) avec `uj_per_us_compute`, `uj_per_uart_frame`, `r2`.

**Règle d'écriture** : `energy_uj_per_inference` porte ici `method: "régression I(rate)"`.
Il ne doit **jamais** être fusionné avec le champ homonyme obtenu par delta (S5302) —
ce sont deux estimateurs différents, dont la comparaison est elle-même un résultat.

### 5. N/A honnête

Si `r² < 0.9` ou si la pente n'est pas significativement positive (pente < 2 × son
écart-type), écrire `energy_uj_per_inference: "à mesurer"` avec
`na_reason: "régression non significative : r²=…, pente=… ± …"`. **Ne jamais rapporter
une pente négative comme une énergie.**

## Critères d'acceptation

- [ ] 7 points de cadence × 3 répétitions × 8 cellules, ordre randomisé, une session.
- [ ] 0 erreur CRC à toutes les cadences (vérifier notamment la borne haute : au-delà de la
      saturation, le flux perd des trames — c'est ce qui définit `saturation_rate_hz`).
- [ ] `r²` rapporté pour chaque cellule ; N/A honnête si la régression n'est pas
      significative.
- [ ] `slope_vs_latency.json` produit, avec les deux grandeurs dérivées
      (µJ/µs de calcul, µJ/trame UART).
- [ ] **Verdict Gap 3 énergie** explicite : à la cadence de saturation, l'écart INT8↔FP32
      sur EWC est-il significatif, et dans quel sens ?
- [ ] Contrôle de cohérence croisée : `uj_per_us_compute` × (latence INT8 − latence FP32)
      doit être du même ordre que l'écart de pente mesuré entre les deux encodages.

## Implémentation livrée (2026-08-05, sans carte ni sonde)

Le code est écrit, exécutable et testé **hors banc** ; seule l'acquisition reste à faire.

| Fichier | Rôle |
|---------|------|
| `src/evaluation/rate_regression.py` | Règle de calcul ET de décision, hors du pilote (précédent `counterbalance.py`, S5301) : moindres carrés pondérés `1/σ²` avec erreur-type de la pente, N/A honnête, détection de saturation, régression de second niveau, verdict Gap 3, cohérence croisée |
| `scripts/run_s53_rate_sweep.py` | Pilote de banc — n'acquiert que des courants, n'arbitre rien |
| `tests/test_s53_rate.py` | 27 tests PASS + 2 `skip` (cellules non encore mesurées) |
| `scripts/sensor_stream.py` | `duration_s` et `achieved_rate_hz` **additifs** dans `_compute_stats` (protocole UART inchangé) |
| `scripts/run_s53_freq_sweep.py` | `_linear_fit` délègue au module commun — une seule régression dans le dépôt |

**Ce que le code fait, au-delà de la lettre de la spec :**

- **Saturation constatée, jamais supposée.** Rien ne mesurait la cadence réellement
  atteinte : au-delà du plafond de transport, `sensor_stream.py` ne perd aucune trame et ne
  lève aucun CRC, il émet simplement moins vite. `achieved_rate_hz` est dérivée des
  horodatages d'émission déjà portés par chaque échantillon. Les points saturés sont
  **écartés de l'ajustement mais restent écrits** : une mesure écartée se montre.
- **Ordre randomisé par défaut** (`--seed 42`, `--no-shuffle-order` pour s'en passer), avec
  le rang `session_index` de chaque acquisition — la leçon de S5301.
- **Latence DWT mesurée en session**, par un flux de contrôle sous `hold_run`, jamais
  recopiée d'un sprint antérieur : c'est le second axe de la régression de niveau 2.
- **`--refit`** recalcule pentes, énergies et agrégats depuis les points écrits, sans
  matériel. Le pilote du 2026-08-05 n'est pas touché (il ne porte pas de `points`).
- `maha_int8` reste hors du balayage par défaut (`BUILD_SPECIFIC`, build `-DMAHA_INT8`) —
  protection contre le bug de drapeau du Sprint 52.

**Validation disponible sans banc** : les 20 points mesurés du pilote repassent dans la
chaîne de calcul et redonnent **exactement** ses chiffres (pente, ordonnée, r²,
erreur-type, µJ et incertitude, à `rel=1e-9`) — `test_reproduit_le_pilote_mesure`. C'est la
seule validation possible sur des données de banc réelles aujourd'hui, et le garde-fou du
jour où quelqu'un « améliorera » la régression.

**Nuance honnête relevée à l'implémentation** : pour une régression simple,
`r² = t²/(t² + n − 2)` avec `t = pente/σ_pente`. Le critère `r² ≥ 0,9` impose donc déjà
`t ≥ 3√(n−2) > 2` : le seuil `2σ` de la spec §5 **n'ajoute aucune sévérité** au critère de
linéarité. Ce qu'il attrape réellement, c'est le **signe** — une droite décroissante
parfaitement linéaire passe le test du r² et doit malgré tout être refusée (c'est
exactement le cas où le protocole delta du Sprint 50 rendait des µJ négatifs). Les deux
critères sont conservés, avec ce rôle explicité dans le code.

**Reste à faire, et uniquement au banc** : les 7 × 3 × 8 acquisitions, le contrôle
d'intégrité à toutes les cadences, et le verdict Gap 3 — qui sortira des chiffres, pas de
ce document.

## Ce que cette tâche règle définitivement

C'est la seule tâche du sprint qui produit des µJ/inférence **quelles que soient** les
conclusions de S5301, S5302 et S5303. Elle est donc placée tôt dans l'ordre de banc, juste
après le diagnostic, et avant tout reflash.

---

## Résultats de banc — 2026-09-07 (NUCLEO-F439ZI + X-NUCLEO-LPM01A)

Grille complète : 7 cadences (0/10/25/50/100/150/200 Hz) × 3 répétitions × 7 cellules,
ordre randomisé (seed 42), fenêtre 10 s, `maha_int8` exclue (build `-DMAHA_INT8` dédié).
**Contrôles d'intégrité 300/300 sur les 7 cellules, 0 erreur CRC.** Préchauffage écarté
(60,34 mA, non retenu).

| Cellule | Latence DWT P50 | µJ/inférence | r² | Plafond atteint à 200 Hz |
|---|---|---|---|---|
| `ewc_fp32` | 50 µs | **67,7 ± 0,7** | 1,000 | 168,5 Hz |
| `ewc_int8` | 53 µs | **67,5 ± 2,2** | 0,996 | 167,8 Hz |
| `hdc_fp32` | 585 µs | *à mesurer* | 0,885 | 164,0 Hz |
| `hdc_int8` | 1958 µs | **191,6 ± 16,0** | 0,973 | 126,3 Hz |
| `tinyol_fp32` | 85 µs | **70,5 ± 0,7** | 1,000 | 166,6 Hz |
| `tinyol_int8` | 65 µs | **71,2 ± 1,3** | 0,999 | 167,2 Hz |
| `maha_fp32` | 5 µs | **57,4 ± 7,5** | 0,935 | 169,3 Hz |

Régression de second niveau (`slope_vs_latency.json`) :
**0,0656 ± 0,0022 µJ/µs de calcul** et **63,41 µJ par trame UART**, r² = 0,996.

### Verdict Gap 3 énergie — `non_significatif`

EWC INT8 67,5 µJ contre FP32 67,7 µJ : **Δ = −0,2 µJ pour ±2,3 µJ d'incertitude combinée**
(< 2 σ). L'INT8 ne change pas mesurablement l'énergie par inférence ; **son gain reste la
RAM**. Ce constat rejoint celui du Sprint 50 par une méthode plus propre : rien n'est
soustrait d'une référence de repos, seule la pente porte le coût marginal.

### Réserves d'interprétation

- **Le coût de trame UART domine.** Pour EWC, 63,4 µJ sur 67,7 (94 %) sont la transaction
  série, pas le modèle. Seule la part de calcul (latence × 0,0656 µJ/µs) est imputable au
  modèle — c'est la raison d'être de la régression de second niveau. Artefact de banc
  assumé : le capteur est simulé par UART.
- `hdc_fp32` sort en **N/A honnête** (r² = 0,885 < 0,9) : le courant ne suit pas une droite
  en cadence sur ce balayage.
- Le `coherence_check` sort à `false` (prédit +0,2 µJ, mesuré −0,2 µJ) : il oppose deux
  quantités toutes deux très en dessous de l'incertitude. Il ne contredit pas le verdict.
- Cohérence interne : 63,41 + 1958 × 0,0656 = 191,8 µJ prédits pour `hdc_int8`, contre
  191,6 mesurés.
