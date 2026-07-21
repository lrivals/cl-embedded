# Moments de quantification — taxonomie CL-Embedded (S4601)

> **Résumé** : ce document est la source de vérité textuelle des trois **moments** où la
> quantification peut intervenir dans le cycle de vie d'un modèle — `before` (pendant
> l'entraînement, QAT), `after` (sur le modèle figé, PTQ) et `both` (QAT puis export PTQ,
> le vrai chemin de déploiement firmware). Il fige le vocabulaire, le mapping par modèle
> et la métrique de référence, pour que le harnais [`run_s46_quant_moment.py`](../../scripts/run_s46_quant_moment.py)
> (S4602) et les figures (S4606) parlent le même langage. **Aucun chiffre de résultat
> ici** — ce document est une taxonomie ; les valeurs sortent de S4603–S4605.

**Axe orthogonal**. Ce document traite le **moment** (quand quantifier). Le *format*
(comment quantifier : INT8 per-tensor, per-canal, Q15, int16-AM…) est traité par
[`quantization_strategies.md`](quantization_strategies.md) (S4202). Les deux axes sont
indépendants et se citent mutuellement : un même format (INT8 calibré) peut s'appliquer
`after` ou `both` ; un même moment (`after`) peut employer plusieurs formats (`legacy_c`,
`per_tensor_calib`, `per_channel_int8`, `q15`).

---

## 1. Les trois moments (`before` / `after` / `both`)

| Moment | Nom | Définition opérationnelle | Implémentation existante |
|--------|-----|---------------------------|--------------------------|
| **`before`** | QAT — *quantization-aware training* | Un nœud `quant → déquant` (fake-quant) est inséré dans le forward **pendant** l'entraînement ; les gradients traversent l'arrondi par *straight-through estimator* (STE, gradient ≈ identité). Le modèle **apprend en voyant le bruit de quantification** et adapte ses poids. L'accumulateur reste flottant : c'est une *simulation* de l'INT8, pas un chemin entier. | [`src/models/ewc/ewc_mlp_int8.py`](../../src/models/ewc/ewc_mlp_int8.py) `EWCMlpInt8Classifier` · [`src/models/tinyol/tinyol_int8.py`](../../src/models/tinyol/tinyol_int8.py) |
| **`after`** | PTQ — *post-training quantization* | On entraîne en FP32, puis on quantifie les poids **du modèle figé**. Aucun ré-apprentissage. La qualité dépend entièrement du *format/calibration* choisi (`after_scheme`). | [`src/utils/int8_c_emulation.py`](../../src/utils/int8_c_emulation.py) `forward_quant` (legacy figé, per-tensor calibré, per-canal, Q15) |
| **`both`** | QAT → export PTQ | On entraîne avec fake-quant (`before`), on **extrait les poids appris**, puis on les passe dans le **noyau PTQ** entier (`after`). **C'est le chemin réel de déploiement** : la board n'exécute jamais de fake-quant, elle exécute un noyau entier alimenté par des poids appris sous contrainte de quantification. | **câblé par S4602** — `EWCHeadWeights.from_state_dict` consomme les `fc1/fc2/fc3` de `EWCMlpInt8Classifier` (mêmes noms que la tête FP32) |

**Schéma cycle de vie.**

```
                 données ─► entraînement ─► modèle figé ─► export C ─► noyau entier (board)
                              │                  │              │            │
   before (QAT) ..............●                  │              │            │   fake-quant DANS la boucle (float, STE)
   after  (PTQ) .................................●─────────────►│            │   quantif du modèle FP32 figé
   both   (QAT→PTQ) ...........●·················· (poids QAT) ─►●───────────►●   appris quantif-conscient PUIS noyau entier
```

> **Point d'honnêteté** (repris de la roadmap S46) :
> - `before` est une **borne haute** — la carte n'exécute jamais de fake-quant, donc la
>   métrique `before` (fake-quant à l'inférence) surestime ce qui sera réellement déployé.
> - `after` isole l'effet de la PTQ **sans** le bénéfice de l'entraînement conscient.
> - `both` est la **seule colonne fidèle au déploiement** (noyau entier + poids appris
>   sous contrainte). C'est la variante à privilégier pour conclure sur le firmware.

---

## 2. Mapping par modèle (quels moments s'appliquent)

| Modèle | `before` (QAT) | `after` (PTQ) | `both` | Justification |
|--------|:---:|:---:|:---:|---------------|
| **EWC** | ✅ | ✅ | ✅ | Vraie boucle d'entraînement à fake-quant (`EWCMlpInt8Classifier`) **et** noyau PTQ bit-exact (`int8_c_emulation`). Axe 3-way complet — modèle **prioritaire** du sprint. |
| **TinyOL** | ✅ | ✅ | ✅ | Fake-quant dans la boucle online (`OtOHeadInt8` / `tinyol_int8.py`) **et** calibration PTQ des activations. Axe 3-way exerçable (S4604). |
| **HDC** | ✖ structurel | ✖ structurel | ✖ | **Nativement entier** : hypervecteurs ±1 (int8), mémoire associative int16. La quantification n'est pas une *conversion* mais la *structure* même du modèle → la métrique INT8 ≡ FP32 **par construction** ([`quantization_strategies.md` §6](quantization_strategies.md), `int16_am`). Il n'existe pas d'axe avant/après : rien n'est « converti ». |
| **Mahalanobis** | ✖ pas d'entraînement | ✅ (INT8 / Q15) | ✖ | Détecteur **sans entraînement par gradient** (fit statistique μ/Σ⁻¹) : ni `before` ni `both` n'ont de sens (pas de boucle à rendre quantif-consciente). Son axe pertinent n'est pas le *moment* mais le **format** : INT8 casse `sigma_inv_` (grande dynamique), Q15 récupère (Sprint 34, [`quantization_strategies.md` §5](quantization_strategies.md)). |

**Conséquence de cadrage.** La grille 3-way ne concerne que **EWC + TinyOL × {Monitoring,
Pronostia}**. HDC et Mahalanobis sont documentés en **contexte** (S4605), avec des cellules
explicitement **N/A** (structurel / hors-axe) — **jamais remplies par un chiffre
artificiel**. Une case N/A porte toujours sa justification (structurel pour HDC, hors-axe
pour Mahalanobis), elle n'est jamais laissée vide.

---

## 3. Réconciliation voie / métrique de référence par modèle

Deux voies historiques divergent (à documenter, pas à masquer) :

| Voie | Modèle-classe | Tête | Métrique | Loader |
|------|---------------|------|----------|--------|
| QAT (Sprint 28) | `EWCMlpInt8Classifier` | binaire sigmoïde (`fc3` → 1) | **AUROC** (normal-vs-faute) | `_get_tasks` (benchmark) |
| PTQ (Sprint 39) | `EWCMlpMulticlass` | 2-logits | **F1_faulty** | `load_condition_arrays` |

**Décision Sprint 46** : la **voie de référence par modèle est la voie QAT binaire
(AUROC)** pour EWC, car c'est la seule qui possède les trois moments dans une **unique
classe de modèle**. Le noyau PTQ `int8_c_emulation` (générique sur la dimension de sortie)
est **adapté** au chemin `both`/`after` en réutilisant `EWCHeadWeights.from_state_dict` sur
les poids `fc1/fc2/fc3` — noms **identiques** dans `EWCMlpClassifier` (FP32) et
`EWCMlpInt8Classifier` (QAT), donc l'extraction est valable pour les deux. Pour une tête
binaire (`fc3` de sortie 1), le logit unique sert directement de score AUROC (transformation
monotone → AUROC invariante). La métrique reportée reste **native au modèle** : AUROC pour
EWC, erreur de reconstruction / F1 pour TinyOL. Ce choix est tracé ici pour éviter toute
comparaison AUROC ↔ F1 trompeuse. `TODO(arnaud)` : valider ce choix de métrique de référence.

---

## 4. Clé `quant_moment` et presets `after_scheme`

Les configs `configs/quant_moment/*.yaml` portent une clé de moment unique :

```yaml
model: ewc                      # ewc | tinyol
dataset: monitoring             # monitoring | pronostia
extends: ewc_int8_monitoring    # réutilise archi + hyperparamètres EWC existants
quant_moment: both              # before | after | both  (+ fp32 = baseline implicite)
after_scheme: per_tensor_calib  # only for after/both : legacy_c | per_tensor_calib | per_channel_int8 | q15
seed: 42
metric: auroc                   # auroc (ewc) | recon_error / f1 (tinyol)
```

- `quant_moment` mappe vers un chemin d'exécution dans `run_s46_quant_moment.py` (S4602)
  et peut être surchargé par `--moment {fp32,before,after,both,all}` sur la CLI.
- `after_scheme` réutilise les presets `QuantConfig` existants de `int8_c_emulation.py`
  (source unique — pas de redéfinition) :

  | `after_scheme` | preset `QuantConfig` | clé RAM/BOPs (`run_s39_quant_sweep`) | Format ([`quantization_strategies.md`](quantization_strategies.md)) |
  |----------------|----------------------|--------------------------------------|--------|
  | `legacy_c` | `QuantConfig.legacy_c()` | `int8_legacy` | `int8_ptq_legacy` (§3) |
  | `per_tensor_calib` | `QuantConfig.per_tensor_calib()` | `int8` | `int8_v2` (§4) |
  | `per_channel_int8` | `QuantConfig.per_channel_int8()` | `int8_perchannel` | `int8_v2` (§4) |
  | `q15` | `QuantConfig.q15()` | `q15` | `q15` (§5) |

Aucun hyperparamètre en dur : la source de vérité reste les configs (conforme
[`CLAUDE.md` § Reproductibilité](../../CLAUDE.md)).

---

## 5. Renvois

- **`before` / QAT** — Sprint 28 (benchmark INT8 vs FP32), `exp_S28_PC_ewc_hdc/`,
  [`benchmark_int8_fp32.py`](../../scripts/benchmark_int8_fp32.py).
- **`after` / PTQ** — Sprint 39 (émulateur bit-exact + ablation de la perte F1),
  `exp_S39_ablation/`, `exp_S39_quant_sweep/`, [`int8_c_emulation.py`](../../src/utils/int8_c_emulation.py).
- **Format Q15** — Sprint 34 (Mahalanobis Q15, récupération grande dynamique),
  `exp_S34_maha_q15/`.
- **HDC / TinyOL INT8** — Sprint 24, [`quantization_strategies.md`](quantization_strategies.md)
  (§4 `int8_v2`, §6 `int16_am`).
- **Axe format (orthogonal)** — [`quantization_strategies.md`](quantization_strategies.md) (S4202).
- **Chemin `both` (nouveau, S4602)** — [`run_s46_quant_moment.py`](../../scripts/run_s46_quant_moment.py).
- **Positionnement Gap 3** — [`triple_gap.md`](../triple_gap.md), [`CLAUDE.md` § Gap 3](../../CLAUDE.md).
