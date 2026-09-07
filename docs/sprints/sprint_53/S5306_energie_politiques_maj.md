# S5306 — Énergie des politiques de mise à jour P0–P3 (gate autonome)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 53 |
| **Priorité** | 🔴 Critique — **l'axe le plus fort du sprint** |
| **Statut** | ⚠️ **Séance 1 mesurée (2026-09-07) — N/A honnête** : régressions non publiables sur le build S38 ; cause partiellement isolée (effet dépendant du binaire) |
| **Durée estimée** | 3 h (1 h pilote ✅ + 1 h banc ⏳ + 1 h analyse ⏳) |
| **Dépendances** | S5304 (méthode retenue) ✅ · `scripts/run_sprint38_board.py` ✅ · `experiments/exp_S38_summary.json` ✅ |
| **Fichiers cibles** | `scripts/run_s53_policy_energy.py` ✅ · `src/evaluation/policy_energy.py` ✅ · `tests/test_s53_policy_energy.py` ✅ · `experiments/exp_S53_policy_energy/` ⏳ |
| **Références** | Sprint 38 (S3805–S3809) · `exp_S38_summary.json:economy_table` |

## Contexte

Le Sprint 38 a mesuré sur carte réelle que le gate de nouveauté embarqué économise
**~97 % des mises à jour** (update_rate 0,025 contre 1,0 pour la politique `always`) et
**159–169 µs par échantillon**, au prix de **+300 B** de RAM, à **F1 préservé** en mode
`pretrained` (Δ ≤ 0,02). C'est le résultat le plus « système » du projet.

Mais il n'existe **qu'en latence et en RAM**. Or l'argument de déploiement d'un gate
autonome est fondamentalement énergétique : *à quoi bon éviter 97 % des mises à jour si on
ne sait pas ce qu'elles coûtent ?* La sonde étant posée, cette conversion coûte une heure
de banc.

Second bénéfice : c'est la **seule voie** vers `energy_uj_per_update`, aujourd'hui N/A dans
les 8 cellules du Sprint 50 avec la raison « campagne d'inférence seule ».

## Spec

### 1. Cellules

4 politiques × 2 datasets = **8 cellules**, reprenant strictement le cadrage S38 :

| Politique | Build | Flux |
|-----------|-------|------|
| **P0 `frozen`** | défaut | sans `--update` |
| **P1 `always`** | défaut | avec `--update` |
| **P2 `gated_truelabel`** | `-DEWC_AUTO_UPDATE` | label transmis (SGD à bord) |
| **P3 `gated_pseudolabel`** | `-DEWC_AUTO_UPDATE -DGATE_PSEUDO_LABEL` | 100 % autonome |

Datasets : **Monitoring** (drift inter-équipements) et **Pronostia** (temporel), comme S38.
Mode `pretrained` (celui où le F1 est préservé — c'est la configuration défendable).

**3 flashes** suffisent (défaut, gate-P2, gate-P3) : P0 et P1 partagent le build par
défaut et ne diffèrent que par le drapeau UART `PROTO_FLAG_UPDATE`.

> **Corrigé à l'implémentation (2026-08-06).** Ces 3 flashes le sont **par dataset**, soit
> **6 au total**. `run_sprint38_board.build_and_flash*` compile avec `EWC_IN=k` et exporte
> les poids de la tête EWC du dataset : Monitoring (k=4) et Pronostia (k=5) n'ont ni la même
> dimension ni les mêmes poids. Aucun binaire ne peut servir les deux.

### 2. Mesure

Appliquer la méthode retenue à l'issue de S5301/S5302/S5304 — par défaut le **balayage de
cadence** (S5304), qui donne directement la pente, donc l'énergie par échantillon traité,
politique par politique.

Grandeurs dérivées, **calculées et non saisies** :

- `energy_uj_per_sample` par politique (pente × V).
- `energy_uj_per_update` = `(pente(P1) − pente(P0)) × V` — la différence entre « toujours
  mettre à jour » et « jamais » **isole exactement le coût d'une mise à jour CL**. C'est le
  chemin le plus propre vers ce champ, et il n'exige aucune instrumentation nouvelle.
- `energy_saved_vs_always_pct` par politique gated.
- **Autonomie par politique**, via `autonomy.autonomy_hours` et les capacités de
  `hw_profile_f439zi.yaml:116-122` — la traduction en heures est ce qui parle à un
  industriel.

### 3. Croisement avec les données S38 existantes

Le pilote lit `experiments/exp_S38_summary.json` (`economy_table`) en **lecture seule** et
produit le tableau de synthèse à quatre colonnes que le mémoire n'a pas :

| Politique | MAJ économisées (S38) | µs économisées (S38) | **mA économisés (S53)** | **Autonomie (S53)** | F1 (S38) |
|-----------|:---------------------:|:--------------------:|:-----------------------:|:-------------------:|:--------:|

**Ne jamais recopier** les valeurs S38 dans les JSON S53 : les charger.

### 4. Sortie

`experiments/exp_S53_policy_energy/{policy}_{dataset}.json` — schéma de cellule commun
(`method`, `points[]`, `slope_ua_per_hz`, `r2`, `energy_uj_per_sample`, `tension_v`,
`firmware_build`, `update_rate` rechargé depuis S38), plus
`experiments/exp_S53_policy_energy/economy_energy.json` portant les grandeurs dérivées et
le tableau croisé.

## Critères d'acceptation

- [ ] 8 cellules mesurées, 0 erreur CRC, `update_rate` cohérent avec S38
      (frozen 0 < gated ≈ 0,025 < always 1).
- [ ] `energy_uj_per_update` calculé par différence P1−P0, avec sa barre d'erreur propagée.
      Si la différence n'est pas significative, N/A honnête avec la valeur relevée.
- [ ] Autonomie par politique, sur les 5 capacités de batterie du profil HW.
- [ ] Aucune valeur S38 recopiée en dur — toutes chargées depuis `exp_S38_summary.json`.
- [ ] Le surcoût énergétique du **gate lui-même** est isolé : `pente(P2) − pente(P0)` moins
      la part imputable aux 2,5 % de mises à jour effectuées. Le gate coûte ~27 µs
      d'overhead par échantillon (S38) sur *tous* les échantillons — il faut vérifier qu'il
      ne mange pas l'économie qu'il procure.

## Le résultat que cette tâche peut produire

Trois issues, toutes publiables :

1. **Le gate économise significativement** → argument de déploiement complet
   (RAM +300 B, latence −165 µs, énergie −X mA, autonomie +Y h, F1 préservé).
2. **Le gate est neutre** → son overhead permanent compense les mises à jour évitées ; la
   justification du gate redevient la latence pire-cas et l'autonomie de décision, pas
   l'énergie.
3. **Le gate coûte plus qu'il ne rapporte** à faible taux de dérive → **résultat honnête et
   utile**, qui borne le domaine de validité de la contribution du Sprint 38.

## Implémentation livrée (2026-08-06, sans carte ni sonde)

Le code est écrit, exécutable et testé **hors banc** ; seule l'acquisition reste à faire.
`experiments/exp_S53_policy_energy/` est créé **vide** : aucun placeholder chiffré n'y est
écrit tant que la sonde n'a pas tourné.

| Fichier | Rôle |
|---------|------|
| `src/evaluation/policy_energy.py` | Règle de calcul ET de décision, hors du pilote (précédents `counterbalance.py` S5301, `rate_regression.py` S5304) : différence de pentes, incertitudes propagées, N/A honnête, surcoût du gate, verdict des trois issues, autonomie, témoin de séance, croisement S38 |
| `scripts/run_s53_policy_energy.py` | Pilote de banc — n'acquiert que des courants, n'arbitre rien |
| `tests/test_s53_policy_energy.py` | 33 tests PASS + 2 `skip` (cellules non encore mesurées) |
| `scripts/sensor_stream.py` | `auroc`/`forgetting` bruts exposés dans `--dump-samples` (**additif**, protocole UART inchangé) |

**Ce que le code fait, au-delà de la lettre de la spec :**

- **Deux phases séparées, `--prepare` et `--measure`.** Le flash sous alimentation sonde est
  instable (contrainte de banc mesurée, S5008) : le pilote ne flashe donc **jamais** pendant
  qu'il mesure. `--prepare` construit et flashe (sonde débranchée, JP5 en place) en
  réutilisant **sans les réécrire** `run_sprint38_board.build_and_flash` et
  `build_and_flash_gated` — les checkpoints PC du Sprint 38 sont repris tels quels, rien
  n'est ré-entraîné (ré-entraîner produirait un autre modèle, donc une autre énergie).
- **Garde anti-mislabel.** `--prepare` écrit un manifeste `firmware_state.json` (binaire,
  dataset, `k`, `.bss`) que `--measure` confronte au `--build` déclaré : binaire, dataset ou
  politique incompatibles ⇒ **refus d'acquérir**. Sans cette garde, une politique gated
  mesurée sur le binaire par défaut serait écrite sous un nom qui ment sur son contenu —
  exactement le bug du drapeau TinyOL du Sprint 52, et le motif du `hw_check` de S5303.
- **`update_rate` MESURÉ en séance, pas seulement rechargé.** Le compteur du gate ne
  franchissait pas la frontière du sous-processus : `_stream_uart` le voit, `--dump-samples`
  ne l'exposait pas. Les slots V3 réinterprétés (`auroc` = verdict, `forgetting` = compteur
  **cumulé** — lu sur le dernier échantillon, jamais sommé) sont désormais dumpés tels
  quels. Le taux constaté est ensuite **confronté** à celui du Sprint 38, au lieu d'être
  simplement recopié.
- **Témoin de séance.** P0/P1 (binaire par défaut) et P2/P3 (binaires gate) sont
  nécessairement mesurés dans des séances séparées par un reflash : une dérive de banc s'y
  confondrait avec l'effet du gate. Une cellule témoin (Mahalanobis, chemin d'exécution
  inchangé par `-DEWC_AUTO_UPDATE`) est rebalayée à chaque séance ; si sa pente bouge, le
  JSON déclare les séances **non comparables** et le dit dans son `rationale`. La grandeur
  maîtresse — `energy_uj_per_update` = P1 − P0 — reste, elle, **intra-séance**.
- **Ordre randomisé** (`--seed 42`, `--no-shuffle-order` pour s'en passer) avec le rang
  `session_index` de chaque acquisition : la leçon de S5301, appliquée à l'axe des
  politiques. P0 et P1 sont ainsi entrelacées, ce qui rend leur différence robuste à une
  dérive lente.
- **`--refit`** recalcule pentes, différences, verdicts et autonomies depuis les points
  écrits, **sans matériel**.

**Nuance honnête relevée à l'implémentation** : le pourcentage `energy_saved_vs_always_pct`
est rapporté à l'énergie **marginale** de `always` (`pente(always) − pente(frozen)`), et le
dénominateur est écrit dans le JSON. Le rapporter à la consommation totale de la carte le
diluerait dans le coût de trame UART (60,6 µJ au pilote S5304) et le PHY — que la politique
ne change pas — et ferait paraître dérisoire une économie qui peut être majoritaire sur la
part qu'elle concerne réellement.

**Validation disponible sans banc** : la chaîne complète (8 cellules synthétiques →
régressions → différence P1−P0 → surcoût du gate → verdict → autonomie sur les 5 capacités
→ tableau croisé) a été exercée bout en bout, et les colonnes du Sprint 38 en ressortent
**chargées** depuis `exp_S38_summary.json` (le test les lit dans un fichier factice aux
valeurs reconnaissables : un chiffre en dur le ferait échouer).

**Reste à faire, et uniquement au banc** : les 6 flashes (3 binaires × 2 datasets), les
7 × 3 × 8 acquisitions, le contrôle d'intégrité à toutes les cadences, et le verdict — qui
sortira des chiffres, pas de ce document.

### Séquence de banc

Pour chaque dataset ∈ {monitoring, pronostia} :

```bash
python scripts/run_s53_policy_energy.py --prepare --policy frozen --dataset <ds>
# basculer JP5 sur l'alimentation sonde
python scripts/run_s53_policy_energy.py --measure --build default \
       --policies frozen,always --dataset <ds> --board-port /dev/ttyACM0
python scripts/run_s53_policy_energy.py --prepare --policy gated_truelabel --dataset <ds>
python scripts/run_s53_policy_energy.py --measure --build gate \
       --policies gated_truelabel --dataset <ds> --board-port /dev/ttyACM0
python scripts/run_s53_policy_energy.py --prepare --policy gated_pseudolabel --dataset <ds>
python scripts/run_s53_policy_energy.py --measure --build gate_pseudo \
       --policies gated_pseudolabel --dataset <ds> --board-port /dev/ttyACM0
```

Firmware **inchangé** par cette tâche (les binaires gate existent depuis S3804) :
`make test` **145 / 0 échec**, `.bss` défaut **invariant 105 036 B**.

---

## Résultats de banc — 2026-09-07 (NUCLEO-F439ZI + X-NUCLEO-LPM01A)

Périmètre exécuté : **dataset `monitoring` seul** (k=4), séance 1 sur 3 (build `default`,
politiques P0 `frozen` et P1 `always`). Pronostia et les séances gated non exécutées.

### Ce qui a fonctionné

- `--prepare` : export, compilation et flash — **k=4, `.bss` = 100 152 B**, manifeste
  `firmware_state.json` écrit, garde anti-mislabel opérationnelle.
- Flux de contrôle **300/300, 0 CRC** sur les deux politiques, avec la différence de charge
  attendue : **`frozen` P50 = 48 µs**, **`always` P50 = 238 µs** (inférence + SGD).
  La différence de travail entre P0 et P1 est donc **mesurée et réelle en latence**.
- Plafonds de transport relevés : `frozen` 184,2 Hz, `always` 170,3 Hz.

### Ce qui bloque — les deux régressions sont non publiables

| Cellule | Pente | r² | Énergie |
|---|---|---|---|
| `frozen_monitoring` | **−0,94 ± 0,62 µA/Hz** | 0,366 | `"à mesurer"` |
| `always_monitoring` | **7,82 ± 2,58 µA/Hz** | 0,696 | `"à mesurer"` |

`energy_uj_per_update` sort donc en N/A, et tout le reste cascade (`gate_overhead`,
`energy_saved_vs_always`, `gate_verdict`), avec des raisons traçables. `update_rate` mesuré
est `None` (les slots du gate ne sont pas peuplés sur le binaire par défaut) ; le taux du
Sprint 38 est utilisé en repli et **affiché comme tel**, jamais recopié en dur.

> **Observation soumise, non publiée** *(au moment de la séance)*. La *différence* de pentes
> vaut 8,76 ± 2,65 µA/Hz, soit **28,6 ± 8,7 µJ par mise à jour** — à plus de 3 σ de zéro, et
> compatible à 1,8 σ avec les ~12,5 µJ prédits par le modèle de second niveau de S5304
> (190 µs × 0,0656 µJ/µs). La règle de publication exigeait alors que **chaque** régression
> soit publiable individuellement, ce qui est plus strict que nécessaire pour une différence
> où le bruit commun s'annule. **La règle n'a pas été assouplie en séance** : la relâcher
> après avoir vu le résultat aurait été un ajustement sur la réponse.

#### Règle tranchée hors banc (2026-09-07, A4)

Ce qu'exige désormais une DIFFÉRENCE de pentes (`policy_energy._lineaire`) :

- **la linéarité de chaque cellule** (`r² ≥ 0,9`) — sans elle une pente ne décrit aucun coût
  marginal, et la différence de deux non-coûts n'en décrit pas davantage ;
- **la significativité de la différence elle-même** (`Δ > 2 σ` propagée), qui reste le seul
  juge de la publication et écarte toujours, par construction, les différences négatives.

Ce qui n'est **plus** exigé : que chaque pente soit séparable de zéro *prise isolément*. Les
deux cellules d'une paire partagent le binaire, la trame UART, la séance et l'ordonnée à
l'origine ; le bruit commun s'annule dans l'écart. C'est précisément le régime où vit le coût
d'une mise à jour CL.

La règle est arrêtée **avant** la reprise des séances 2 et 3, et **la séance 1 ne devient pas
publiable pour autant** : ses deux cellules ont des r² de 0,366 et 0,696, très en dessous du
seuil de linéarité, qui lui n'a pas bougé. Le déblocage attendu reste **B3** (isoler l'effet
« build S38 » à en-têtes de poids identiques), pas un assouplissement de règle.

### Cause partiellement isolée — l'effet dépend du binaire

L'anomalie « repos plus haut que la charge » du Sprint 50 est réapparue sur le build S38
(repos 41,95 mA contre 41,71 mA à 200 Hz). Mesure d'isolation faite dans la foulée, **même
carte, même sonde, même modèle EWC, même grille, à quelques minutes d'intervalle** :

| Build | Pente | r² | µJ/inférence |
|---|---|---|---|
| Dimensions par défaut (`make all`) | **+17,96 µA/Hz** | 0,979 | 58,7 |
| Stock, S5304 (plus tôt le même jour) | +20,73 ± 0,21 µA/Hz | 0,9996 | 67,7 |
| **Build S38 `EWC_IN=4 MAHA_DIM=4 TINYOL_IN=4 HDC_N_FEATURES=4`** | **−0,94 ± 0,62 µA/Hz** | 0,366 | N/A |

Points du build par défaut : 41,91 / 42,59 / 42,62 / 43,65 / 44,70 mA (0→150 Hz), répétitions
serrées à 0,02 mA sur les cadences hautes.

**La dépendance à la cadence est présente sur le build aux dimensions par défaut et absente
sur le build S38.** L'anomalie du Sprint 50 n'est donc **pas une fatalité du banc** : elle
dépend du binaire flashé. La mécanique exacte **n'est pas établie** et n'est pas supposée ici.

> Réserve sur l'isolation : `--prepare` ayant régénéré les en-têtes de poids, le binaire
> « dimensions par défaut » n'est pas identique à celui de S5304 (`.text` 46 680 B contre
> 51 976 B). La comparaison oppose « build S38 complet » à « dimensions par défaut », pas une
> variable unique.
