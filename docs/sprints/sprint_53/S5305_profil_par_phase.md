# S5305 — Profil temporel par phase (`acqmode dyn` ou `--trigsrc d7`)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 53 |
| **Priorité** | 🟠 Important — **conditionnel au succès de S5303** |
| **Statut** | ⚠️ **Mesuré sur carte réelle (2026-09-07) — N/A honnête** : `acqmode dyn` débloqué à 45 MHz, mais la voie A ne segmente pas ce profil |
| **Durée estimée** | 2 h 30 (1 h 30 code + 1 h banc) |
| **Dépendances** | **S5303** (passage sous 59 mA) ou une soudure PA8→D7 |
| **Fichiers cibles** | `scripts/energy_capture.py` (fonction **ajoutée**) · `experiments/exp_S53_phase_profile/` |
| **Références** | `energy_capture.py:262` (`derive_phase_windows`) · `docs/context/lpm01a_setup.md:270-283` (§7, absence de voie numérique) |

## Contexte

Toute la chaîne de segmentation par phase existe depuis le Sprint 33 —
`derive_phase_windows`, `segment_by_phase`, `integrate_energy_uj`, les marqueurs GPIO PA8
(`-DENERGY_MARKERS`), le manifeste `configs/energy_campaign_s50.yaml` — et **n'a jamais pu
être exercée sur des données réelles**, pour une raison unique et documentée : le LPM01A
**ne capture aucune voie numérique** (`lpm01a_setup.md:270-283`). La colonne `sync` que
`_load_csv` sait lire ne peut pas venir de la sonde.

Deux voies restent ouvertes. Cette tâche les traite dans l'ordre de coût croissant.

## Spec

### Voie A (préférée) — segmenter sur le courant lui-même, sans câblage

Si S5303 fait descendre la carte sous 59 mA, `acqmode dyn` fournit une trace de courant à
100 kSPS. À cadence **basse et connue**, les bursts d'inférence y sont des créneaux nets :
il n'y a nul besoin d'une voie de synchro, le signal *est* la synchro.

- Cadence de validation : **10 Hz**. Modèle : **HDC (2095 µs)** → chaque burst couvre
  ~200 échantillons à 100 kSPS, largement segmentable. EWC (50 µs → 5 échantillons) est à
  la limite : le rapporter comme tel, ou allonger artificiellement le travail par un
  paquet de trames consécutives.
- Écrire `derive_phase_windows_from_current(current_a, dt_s, threshold)` **à côté** de
  `derive_phase_windows` (`energy_capture.py:262`) — **ne pas modifier cette dernière**,
  elle reste la voie PA8 et ses 16 tests.
- Seuil : déterminé par la médiane + k·MAD de la trace, pas saisi à la main. Le nombre de
  créneaux détectés doit être **vérifié contre `rate_hz × durée`** — si l'écart dépasse
  5 %, la segmentation est refusée et le résultat sort en N/A.

### Voie B (repli) — une soudure, PA8 → Arduino D7

`--trigsrc d7` de la sonde. Exerce enfin `-DENERGY_MARKERS`, `segment_by_phase` et
`configs/energy_campaign_s50.yaml` **tels quels, sans une ligne de code neuve**. Coût :
une soudure sur la carte, décision utilisateur.

### Limite à conserver telle quelle

Le marqueur PA8 est **1 bit** : `startup`, `acquisition` et `inference` partagent le niveau
haut (`profiling.c:148-170`). La reddition honnête existante (haut → `inference`, bas →
`idle`) doit être conservée. **Ne pas prétendre à un profil 4 phases** — une granularité
réelle exigerait un encodage multi-bit (PA8 + PA9) qui n'est amorcé nulle part dans le
firmware, et qui sort du périmètre de ce sprint.

La voie A a la même limite, formulée autrement : elle sépare *actif* de *inactif*, pas les
trois sous-phases actives.

### Sortie

`experiments/exp_S53_phase_profile/{model}_{sysclk}.json` :

```json
{
  "model": "hdc", "sysclk_mhz": 45, "acqmode": "dyn", "fs_hz": 100000,
  "segmentation": "courant (voie A) | pa8_d7 (voie B)",
  "n_bursts_detected": 100, "n_bursts_expected": 100, "detection_error_pct": 0.0,
  "phases_uj": {"startup": "à mesurer", "acquisition": "à mesurer",
                "inference": 0.0, "idle": 0.0},
  "phases_na_reason": "marqueur 1 bit : startup/acquisition/inference partagent le niveau actif",
  "total_uj": 0.0,
  "energy_uj_per_inference": 0.0,
  "method": "intégration du profil temporel"
}
```

Troisième estimateur de `energy_uj_per_inference`, à nouveau avec son propre `method` —
**la comparaison des trois voies (delta S5302, régression S5304, intégration S5305) est
elle-même un résultat de méthodologie**, et le meilleur contrôle de validité de la
campagne.

## Critères d'acceptation

- [ ] `derive_phase_windows` et ses tests existants **inchangés**.
- [ ] La nouvelle fonction refuse de segmenter (N/A honnête) si le nombre de créneaux
      détectés s'écarte de plus de 5 % de l'attendu.
- [ ] Au moins une cellule profilée, ou un N/A avec la raison **mesurée** (courant relevé
      à l'échec de `acqmode dyn`).
- [ ] Les trois estimateurs de µJ/inférence sont comparés dans le JSON de synthèse S5309,
      jamais fusionnés dans un même champ.

## Implémentation (2026-08-05, hors banc — carte et sonde absentes)

Code complet livré et **testé sur traces synthétiques**. Aucun JSON de mesure n'est écrit :
`experiments/exp_S53_phase_profile/` ne contient qu'un `summary.json` vide de cellules.
**Reste l'acquisition**, conditionnée au build S5303 sous 59 mA.

### `scripts/energy_capture.py` — fonctions **ajoutées**

`derive_phase_windows` (l. 262) et ses 16 tests : **inchangés** (garde de non-régression
explicite dans les tests, signature figée). Ajouts à côté :

| Ajout | Rôle |
|-------|------|
| `MAD_K`, `MAD_TO_SIGMA`, `BURST_TOLERANCE`, `METHOD_INTEGRATION`, `NA_PHASES_1BIT` | Constantes documentées — la tolérance de 5 % et le libellé de méthode ne sont pas saisis dans un JSON |
| `robust_current_threshold` | Seuil `médiane + k·MAD·1,4826` **déduit de la trace**. Cas dégénéré MAD = 0 (repos plat au pas de quantification) → milieu repos↔maximum, toujours déduit de la trace ; binarisation **stricte** pour qu'une trace plate ne rende aucun créneau |
| `derive_phase_windows_from_current(current_a, dt_s, threshold)` | Voie A — mêmes conventions de sortie que la voie PA8 |
| `count_bursts` / `validate_burst_count` | Contrôle du **dénominateur** : écart > 5 % avec `cadence × durée` ⇒ refus + raison chiffrée |
| `marginal_uj_per_burst` | Surcoût d'un créneau, repos de **la même trace** déduit — grandeur commune aux deux voies |
| `profile_from_current` | Enchaînement complet → bloc JSON, ou `"à mesurer"` + raison |

Deux énergies distinctes, jamais confondues : `energy_uj_per_inference` (**marginale**,
comparable au delta S5302 et à la régression S5304) et `energy_uj_per_burst_gross`
(**brute**, repos compris — utile à l'autonomie, pas à la comparaison).

### `scripts/run_s53_phase_profile.py` — pilote de banc (neuf)

- Contrôle la fréquence **avant** toute acquisition en réutilisant
  `run_s53_freq_sweep.check_frequency` (leçon du drapeau TinyOL S52) ; `lp.warmup()` jeté
  d'abord (biais mesuré ~+8 mA).
- Voie A : flux `sensor_stream.py --rate-hz 10` lancé en `Popen` avec temps
  d'établissement, puis `lp.capture(freq="100k", acqmode="dyn")` — motif déjà éprouvé de
  `run_s50_board_current.measure_current`. **`lpm01a_probe.py` n'est pas modifié**
  (`hold_run` fige `freq 1`/ASCII et ne convenait pas ; aucun hook n'a été nécessaire).
- Refus du mode dynamique ⇒ cellule **N/A honnête** portant le message relevé.
- Voie B : `--trigsrc d7` ; une trace portant réellement une colonne `sync` est traitée par
  `derive_phase_windows` **existante** — zéro ligne neuve, et la soudure reste une décision
  utilisateur non engagée par le pilote.
- `--csv` rejoue une trace hors banc ; `--save-csv` conserve la trace brute ; `--summary`
  recalcule l'agrégat sans matériel.
- `samples_per_burst_median` expose la limite : HDC ≈ 210 échantillons par créneau à
  100 kSPS, EWC ≈ 5 (à la limite du segmentable, rapporté comme tel).

### `tests/test_s53_phase.py` — 14 PASS + 1 skip

Non-régression de la voie PA8 · bornes exactes des créneaux · seuil déduit, jamais saisi ·
repos bruité ⇒ 0 créneau · refus hors tolérance ⇒ `"à mesurer"` + raison, **jamais 0** ·
`startup`/`acquisition` toujours `"à mesurer"` · **voies A et B publient la même grandeur**
(garde contre la comparaison de grandeurs différentes en S5309) ·
`METHOD_INTEGRATION ≠ rate_regression.METHOD` · schéma de cellule `skip` tant que le banc
n'a pas tourné.

Non-régression globale : `pytest tests/test_s53_phase.py tests/test_energy_capture.py
tests/test_s53_rate.py tests/test_s53_freq.py tests/test_lpm01a_probe.py
tests/test_s50_energy.py tests/test_energy_delta.py tests/test_s53_bench.py` →
**145 PASS / 8 skip**. Firmware non touché.

### À faire au retour de la carte

```bash
# après flash du build S5303 (-DSYSCLK_MHZ=45), sonde posée
python scripts/run_s53_phase_profile.py --model hdc --sysclk-mhz 45 \
    --board-port /dev/serial/by-id/…STLink… --save-csv captures/hdc_45mhz.csv
```

Si `acqmode dyn` reste refusé, la cellule sort en N/A avec le message de la sonde, et la
voie B (soudure) est à proposer explicitement.

## Si S5303 échoue

Si la carte ne descend pas sous 59 mA même à 45 MHz, cette tâche sort en **N/A honnête**
avec le courant relevé, et la voie B (soudure) est proposée à l'utilisateur comme décision
explicite — elle n'est pas engagée sans son accord.

---

## Résultats de banc — 2026-09-07 (NUCLEO-F439ZI + X-NUCLEO-LPM01A, 45 MHz)

### Le verrou d'entrée est levé

`acqmode dyn` **réussit à 45 MHz** : 1 000 000 d'échantillons sur 1 000 000 attendus à
100 kSPS, sur 9,99999 s (`exp_S53_freq_sweep/45.json`). C'est ce que S5303 devait établir, et
la chaîne de segmentation du Sprint 33 a donc pu être **exercée pour la première fois sur des
données réelles**. Cellule : `experiments/exp_S53_phase_profile/hdc_45.json`.

### La voie A ne segmente pas ce profil — N/A honnête

| Grandeur | Valeur |
|---|---|
| Seuil auto (médiane + k·MAD) | **22,1 mA** (plausible, dans la plage réelle) |
| Créneaux détectés / attendus | **3792 / 100** (+3692 %, tolérance ±5 %) |
| Médiane d'échantillons par créneau | 3 (30 µs) — une inférence en dure 2338 µs, soit ~234 |
| `energy_uj_per_inference` | `"à mesurer"` + `energy_na_reason` chiffré |

**Cause mesurée** : sur la trace, l'écart-type robuste (MAD × 1,4826) vaut **0,362 mA**,
comparable à la marche de courant d'une inférence, pour un taux d'occupation de seulement
2,3 % (2338 µs à 10 Hz) et une distribution à queues lourdes (p99 = 33 mA, max = 72,8 mA).
Échantillon par échantillon à 100 kSPS, **le burst n'est pas séparable du bruit**.

**Le lissage ne sauve pas la méthode.** Moyenne glissante appliquée hors ligne sur la trace
sauvegardée (`captures/hdc_45mhz.csv`), seuil recalculé à chaque fenêtre :

| Fenêtre | 96 | 112 | 128 | 144 | 160 | 176 | 192 | 208 | 224 |
|---|---|---|---|---|---|---|---|---|---|
| Créneaux (attendu 100) | 151 | 126 | **105** | 110 | 91 | 82 | 72 | 71 | 88 |

**Il n'y a pas de palier** : le 105 obtenu à 128 échantillons est une coïncidence, pas un
régime stable. Choisir la fenêtre qui tombe sur 100 reviendrait à ajuster la méthode sur la
réponse attendue — c'est refusé.

### Conclusion à porter au sprint

Résultat **négatif et publiable** : le mode dynamique est accessible à 45 MHz, mais la
segmentation par le courant seul (voie A) est insuffisante pour un profil à faible taux
d'occupation. Le **troisième estimateur** de µJ/inférence n'existe donc pas encore ; le delta
(S5302) et la régression (S5304) restent les deux seuls, et ils ne doivent pas être fusionnés.

### Trois défauts d'outillage corrigés pendant cette tâche

1. `PowerShield.take_control()` échouait sur un prompt résiduel (`htc → Command unknown`) :
   la première exécution de la campagne tombait entièrement. Drain jusqu'au silence + une
   reprise (`_drain`).
2. `voltage_v()` décodait **le dernier jeton** de la réponse à `volt get` ; une ligne de
   service intercalée (jeton lu : `handled`) faisait tomber la cellule. Recherche du dernier
   jeton **au format** mantisse/exposant.
3. `check_frequency` s'exécutait **avant** `probe.take_control()`, donc sur carte hors
   tension (`JP5` retiré ⇒ alimentée seulement pendant une acquisition). Déplacé dans la
   session, après le préchauffage.
