# Inventaire de référence des détecteurs de drift (Sprint 44, S4401)

> **Source de vérité textuelle** de la détection de drift pour les slides et le manuscrit — miroir de
> `docs/context/quantization_strategies.md` (S4202). Chaque détecteur y est catalogué : famille, signal
> d'entrée, principe, **état mémoire** (`# MEM:`), hyperparamètres, **viabilité MCU** argumentée, et
> référence. La colonne viabilité MCU décide la **sélection du Sprint 45** (portage board).

Implémentation : `src/models/drift/` (interface commune `BaseDriftDetector` — S4401). Hyperparamètres :
`configs/sprint44_drift_detection.yaml` (aucune valeur en dur dans le code). Datasets de validation :
registre `src/data.DRIFT_LOADERS` (Sprint 43), avec le synthétique à **vérité-terrain exacte**
`[1500, 3000, 4500]` pour calibrer les métriques.

## Interface commune (`src/models/drift/base.py`)

- `update(value) -> DriftVerdict` — un échantillon par appel (erreur `0/1` **ou** feature/score).
- `set_params_from_reference(reference_values)` — calibration sur le segment d'enrôlement (no-op pour
  les auto-calibrants DDM/EDDM/PH).
- `reset()` · `get_state_bytes() -> int` (empreinte **majorée**, constante dans le temps) ·
  propriété `requires_label -> bool`.
- Helper `error_stream(model, X, y) -> np.ndarray` : `e_t = 1[ŷ_t ≠ y_t]`, réutilise l'inférence d'un
  modèle de faute existant (EWC tête binaire **ou** Mahalanobis seuillé — ne fige pas `TODO(arnaud)` S4400).

**Verdict à 3 niveaux** `DriftVerdict = {NORMAL, WARNING, DRIFT}` : `WARNING` (zone d'alerte) est requis
par DDM/EDDM. Distinct :
- du baseline `SlidingWindowDriftDetector` (`"NORMAL"/"FAULT"/"DRIFT"`, laissé tel quel) ;
- de l'enum firmware `DriftVerdict` (`DRIFT_NORMAL/FAULT/DRIFT`, `inc/drift_detector.h`) — le mapping
  vers le binaire board est **déféré au Sprint 45**.

---

## Familles

### Supervisés — flux d'erreur `0/1` (S4402) · `requires_label = True`

Surveillent le **taux d'erreur** d'un modèle de faute : sa hausse signale un changement de distribution.
État strictement **O(1)** (aucune fenêtre) → **très portables MCU**. Coût : exigent un retour de
vérité-terrain (label) — sur carte déployée seule = scénario « active learning » (P2 du Sprint 38).

#### DDM — Drift Detection Method
- **Signal** : flux d'erreur `0/1`.
- **Principe** : moyenne courante `p_t`, écart-type binomial `s_t = √(p_t(1−p_t)/t)` ; minimum
  `(p_min, s_min)` de `p+s`. `WARNING` si `p+s ≥ p_min + 2·s_min` ; `DRIFT` si `≥ p_min + 3·s_min` ;
  reset au drift.
- **État** : `{n, p, s, p_min, s_min}` — 5 scalaires. `# MEM: 20 B @ FP32`.
- **Hyperparamètres** : `warning_level=2.0`, `drift_level=3.0`, `min_instances=30`.
- **Viabilité MCU** : ✅ **excellente** — O(1), 5 flottants, aucune allocation/tri.
- **Réf.** : Gama et al. 2004.

#### EDDM — Early Drift Detection Method
- **Signal** : flux d'erreur `0/1`.
- **Principe** : distance moyenne **entre erreurs** `p'` + écart-type `s'` (Welford) ; maximum
  `(p'_max, s'_max)` de `p'+2s'`. `WARNING` si `(p'+2s')/(p'_max+2s'_max) < α` ; `DRIFT` si `< β`.
  Meilleur sur drift **graduel**.
- **État** : 8 scalaires. `# MEM: 32 B @ FP32`.
- **Hyperparamètres** : `alpha=0.95`, `beta=0.90`, `min_errors=30`.
- **Viabilité MCU** : ✅ O(1). ⚠️ plus sujet aux **faux positifs** sur flux bruités (comportement connu).
- **Réf.** : Baena-García et al. 2006.

#### Page-Hinkley — test séquentiel (CUSUM)
- **Signal** : flux d'erreur **ou** feature scalaire (frontière avec S4403).
- **Principe** : moyenne courante `x̄_t`, cumul `m_T = Σ(x_t − x̄_t − δ)`, minimum `min_T` ;
  `DRIFT` si `m_T − min_T > λ`. Pas de `WARNING` (test binaire).
- **État** : `{n, mean, cumulative, min}` — 4 scalaires. `# MEM: 16 B @ FP32`.
- **Hyperparamètres** : `delta=0.005` (tolérance), `lambda_=50.0` (seuil), `min_instances=30`.
- **Viabilité MCU** : ✅ **excellente** — O(1), détection rapide sur drift soudain (souvent ≤ DDM).
- **Réf.** : Page 1954.

### Non-supervisés — features / score (S4403) · `requires_label = False`

Surveillent la **distribution des features** sans label — cas réaliste d'une carte déployée seule sur une
machine neuve (S38). Plus coûteux (fenêtre O(W), parfois tri) → **viabilité MCU = point à trancher S45**.
Fenêtres à **capacité fixe** (miroir de `firmware/.../inc/ring_buffer.h`) : empreinte majorée, identique
PC↔board. Multivarié : par défaut **par feature puis agrégation** (`max`/`fraction`, config) via
`MultiFeatureDriftDetector`, sauf MMD (nativement multivarié).

#### PSI / Jensen-Shannon
- **Signal** : feature scalaire (comptée dans des bacs fixes).
- **Principe** : histogramme de référence figé à l'enrôlement ; par bloc de `block_size` échantillons,
  `PSI = Σ(p_cur−p_ref)·ln(p_cur/p_ref)` (ou divergence JS). `DRIFT` si `> seuil` (PSI > 0.2 standard).
  Comptage **incrémental** dans les bacs → aucune valeur brute stockée.
- **État** : `edges (bins+1) + ref_probs (bins) + cur_counts (bins)`. `# MEM: (3·bins+1)·4 B @ FP32`.
  **O(bins), indépendant de W**.
- **Hyperparamètres** : `bins=10`, `block_size=200`, `metric∈{psi,js}`, `psi_threshold=0.2`, `js_threshold=0.1`.
- **Viabilité MCU** : ✅ **le plus MCU-friendly** des non-supervisés — histogramme borné, pas de tri.
- **Réf.** : PSI (standard credit-scoring) · Lin 1991 (JS).

#### KS glissant deux-échantillons
- **Signal** : feature scalaire.
- **Principe** : `ks_2samp(ref_window, cur_window)` tous les `stride` échantillons ; `ref` figé à
  l'enrôlement, `cur` = fenêtre bornée courante. `DRIFT` si p-valeur `< α`.
- **État** : `ref (≤ ref_size) + fenêtre (window_size)`. `# MEM: (ref_size+window_size)·4 B @ FP32`. O(W).
- **Hyperparamètres** : `window_size=100`, `stride=50`, `alpha=0.01`, `ref_size=200`.
- **Viabilité MCU** : ⚠️ O(W) + **tri** dans le test KS (`scipy.stats.ks_2samp`). Portable à W borné, mais
  tri à réimplémenter en C.
- **Réf.** : test de Kolmogorov-Smirnov.

#### KSWIN — Kolmogorov-Smirnov Windowing
- **Signal** : feature scalaire.
- **Principe** : fenêtre bornée W ; compare les `r` échantillons récents à un tirage aléatoire de `r`
  du réservoir (KS). `DRIFT` si p-valeur `< α` → purge de la partie ancienne (adaptation).
- **État** : fenêtre (W). `# MEM: W·4 B @ FP32`. O(W) + tri.
- **Hyperparamètres** : `window_size=100`, `stat_size=30`, `alpha=0.005`, `seed=42`.
- **Viabilité MCU** : 🟡 O(W) + tri + tirage aléatoire ; auto-adaptatif mais **plus de faux positifs**.
- **Réf.** : Raab et al. 2020.

#### MMD — Maximum Mean Discrepancy (noyau RBF)
- **Signal** : feature **ou vecteur** (nativement multivarié).
- **Principe** : MMD² RBF entre `ref` figée et fenêtre courante ; **estimateur linéaire O(W)** (Gretton
  2012 §6) privilégié à la forme quadratique O(W²). Seuil = **percentile d'enrôlement** (portable) plutôt
  que permutation (diagnostic PC). `γ` par heuristique de la médiane.
- **État** : `ref (n_ref·d) + fenêtre (W·d)`. `# MEM: (n_ref+W)·d·4 B @ FP32`. O(W).
- **Hyperparamètres** : `window_size=100`, `stride=50`, `gamma=null`, `estimator=linear`,
  `calib_percentile=99`, `calib_blocks=20`, `seed=42`.
- **Viabilité MCU** : ⚠️ estimateur **linéaire requis** (quadratique irréaliste) ; exponentielles RBF sur
  FPU. Portable à W borné, coût > histogramme.
- **Réf.** : Gretton et al. 2012.

#### ADWIN — ADaptive WINdowing
- **Signal** : feature scalaire (ou score).
- **Principe** : fenêtre **adaptative** via histogramme exponentiel de buckets (moyenne + variance) ;
  coupe la fenêtre — `DRIFT` — dès que deux sous-fenêtres diffèrent au-delà de la borne de Hoeffding
  `ε_cut(delta)`.
- **État** : O(log W) buckets ; **borné par `max_rows`** (config).
  `# MEM: max_rows·(max_buckets+1)·8 B @ FP32`.
- **Hyperparamètres** : `delta=0.002`, `max_buckets=5`, `min_window_length=5`, `min_clock=32`, `max_rows=40`.
- **Viabilité MCU** : 🟡 état majoré (buckets bornés) mais logique de fusion/coupe complexe à porter en C.
- **Réf.** : Bifet & Gavaldà 2007.

### Baseline projet (cataloguée, réutilisée — pas réimplémentée)

#### SlidingWindowDriftDetector
- **Signal** : score d'anomalie (Mahalanobis, reconstruction…).
- **Principe** : double seuil sur fenêtre glissante — `FAULT` si `score > fault_threshold` (instantané),
  `DRIFT` si fraction de la fenêtre `> drift_threshold` dépasse `drift_ratio`. Verdict
  `"NORMAL"/"FAULT"/"DRIFT"`.
- **État** : `deque(maxlen=W)`. `# MEM: 200 B @ FP32 / 50 B @ INT8` (W=50, d=4). O(W).
- **Hyperparamètres** : `window_size=50`, `fault_multiplier=2.5`, `drift_multiplier=1.3`, `drift_ratio=0.6`.
- **Viabilité MCU** : ✅ **déjà porté C** (`firmware/.../src/drift_detector.c`, Sprint 38, ring buffer).
- **Réf.** : Sprint 9/38 · `src/evaluation/drift_detector.py`.

---

## Tableau comparatif

| Détecteur | Famille | Signal | État | `get_state_bytes` (défaut) | Label | MCU |
|-----------|---------|--------|------|----------------------------|:-----:|:---:|
| DDM | supervisé | erreur | O(1) | 20 B | ✅ | ✅ |
| EDDM | supervisé | erreur | O(1) | 32 B | ✅ | ✅ |
| Page-Hinkley | supervisé | erreur/feature | O(1) | 16 B | ✅ | ✅ |
| PSI / JS | non-sup. | feature | **O(bins)** | 124 B (bins=10) | ❌ | ✅ |
| KS glissant | non-sup. | feature | O(W) | 1200 B | ❌ | ⚠️ |
| KSWIN | non-sup. | feature | O(W) | 400 B | ❌ | 🟡 |
| MMD | non-sup. | feature/vecteur | O(W) | (n_ref+W)·d·4 | ❌ | ⚠️ |
| ADWIN | non-sup. | feature | O(log W) borné | max_rows·(M+1)·8 | ❌ | 🟡 |
| SlidingWindow (baseline) | baseline | score | O(W) | 200 B | ❌ | ✅ (porté C) |

**Lecture Sprint 45** : les 3 supervisés + PSI sont les candidats board les plus solides (état majoré,
pas de tri) ; KS/KSWIN/MMD demandent un tri/kernel à porter ; ADWIN une logique de buckets complexe.

---

## Recommandation de portage MCU (S4406 — livrable pour le Sprint 45)

> **Traçabilité** : ce classement s'appuie sur la grille PC **mesurée** (S4405,
> `experiments/exp_S44_PC_{detector}_{dataset}/results.json`) et le tableau de synthèse du notebook
> `notebooks/cl_eval/drift_detection/comparison.ipynb`. Les `state_bytes` sont l'empreinte
> **algorithmique** (`get_state_bytes()`) ; la latence est un **proxy PC** (mesure DWT / `.bss` = S45).
> La colonne `viabilite_mcu` des `results.json` est dérivée de l'état **mesuré** (seuils nommés dans
> `scripts/run_sprint44_pc.py` : ≤ 1 024 B = *haute*, ≤ 16 384 B = *moyenne*, au-delà = *pc_only*).

Constat clé mesuré : l'empreinte des non-supervisés **dépend de la dimensionnalité** du dataset (un
détecteur scalaire par feature via `MultiFeatureDriftDetector`), tandis que les supervisés restent
**O(1) invariant au dataset** (flux d'erreur `0/1`, une dimension).

**Candidats primaires** (retenus pour `S4501`) :

1. **Page-Hinkley (16 B), DDM (20 B), EDDM (32 B)** — état O(1) *invariant*, viabilité **haute** sur les
   4 datasets, délai fini sur le synthétique à vérité-terrain exacte. **Coût** : supervisés → exigent un
   label (retour de vérité-terrain = *active learning* P2 du Sprint 38). Portages les plus sûrs.
2. **PSI (O(bins))** — **non-supervisé** (autonome, aucun label), état O(bins) *indépendant de la
   fenêtre*, viabilité **haute**/*moyenne* selon la dimensionnalité. Meilleur compromis autonomie/coût
   côté non-supervisé.

**Référence déjà portée** : **`SlidingWindowDriftDetector`** (200 B, `drift_detector.c`, Sprint 38).

**Secondaires** (à valider sous budget board) : **KSWIN / KS-Test / ADWIN** — état O(W) borné mais qui
**croît avec la dimensionnalité** (mesuré : *moyenne* sur les datasets basse-dim, *pc_only* sur Gas
128 features) ; tri KS / fusion de buckets ADWIN à réimplémenter en C.

**PC-only** : **MMD** — stocke la **référence complète** `n_ref·d` (mesuré *pc_only* dès Hydraulique/Gas).
Bon diagnostic PC, non prioritaire embarqué sans réduction de référence.

**Décision d'axe** : sur carte déployée **sans retour de label**, seuls les **non-supervisés** (PSI en
tête, baseline en référence) sont pleinement autonomes ; les supervisés offrent une meilleure
réactivité au prix d'un label. Le Sprint 45 tranche sur mesures board réelles.
