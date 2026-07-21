# S4601 — Taxonomie des moments de quantification + mapping par modèle

| Champ | Valeur |
|-------|--------|
| **Sprint** | 46 |
| **Priorité** | 🔴 Critique — fonde le vocabulaire et le périmètre du sprint ; tout le harnais et les figures en découlent. |
| **Statut** | 📝 Doc — spec complète ; implémentation à venir |
| **Durée estimée** | 4h |
| **Dépendances** | Sprint 28 ✅ (`EWCMlpInt8Classifier`) · Sprint 39 ✅ (`int8_c_emulation.py`) · Sprint 34 ✅ (Q15 Maha) · Sprint 24 ✅ (HDC/TinyOL INT8) |
| **Fichiers cibles** | `docs/context/quantization_moments.md` (nouveau), référencé par `docs/sprints/sprint_46/S4601_cadrage_taxonomie.md` |
| **Références** | S2800 (QAT PC) · S3901/S3904 (audit + ablation PTQ) · `docs/context/quantization_strategies.md` (S4202) · CLAUDE.md § « Gap 3 » |

---

## Contexte

Le mot « quantification » recouvre au moins deux **moments** distincts dans le cycle de vie d'un modèle,
et le projet les a implémentés séparément sans jamais poser le vocabulaire commun. Cette tâche fige la
**taxonomie** et le **mapping par modèle**, pour que le harnais S4602 et les figures S4606 parlent le
même langage. Elle produit un document de contexte réutilisable (`quantization_moments.md`), à l'image de
`quantization_strategies.md` (S4202) mais orienté *moment* plutôt que *format*.

## Spec

### 1. Définition des trois moments

| Moment | Nom | Définition opérationnelle | Implémentation existante |
|--------|-----|---------------------------|--------------------------|
| **before** | QAT (quantization-aware training) | Fake-quant inséré dans le forward **pendant** l'entraînement ; gradients FP32 via straight-through estimator. Le modèle « apprend » avec le bruit de quantification. | `src/models/ewc/ewc_mlp_int8.py::EWCMlpInt8Classifier` · `src/models/tinyol/tinyol_int8.py` |
| **after** | PTQ (post-training quantization) | On entraîne en FP32, puis on quantifie les poids **du modèle figé**. Aucun ré-apprentissage. | `src/utils/int8_c_emulation.py::forward_quant` (per-tensor calibré, per-canal, Q15) |
| **both** | QAT → export PTQ | On entraîne avec fake-quant (before), on **extrait les poids appris**, puis on les passe dans le **noyau PTQ** du firmware (after). C'est le chemin réel de déploiement. | **à câbler (S4602)** — `EWCHeadWeights.from_state_dict` accepte `fc1/fc2/fc3` que `EWCMlpInt8Classifier` expose |

> **Point d'honnêteté** : `before` (fake-quant à l'inférence) est une **borne haute** — la carte
> n'exécute jamais de fake-quant, elle exécute un noyau entier. `both` est la seule colonne **fidèle au
> déploiement**. `after` isole l'effet de la PTQ sans le bénéfice de l'entraînement conscient.

### 2. Mapping par modèle (quels moments s'appliquent)

| Modèle | before (QAT) | after (PTQ) | both | Justification |
|--------|:---:|:---:|:---:|---------------|
| **EWC** | ✅ | ✅ | ✅ | Vraie boucle d'entraînement à fake-quant (`EWCMlpInt8Classifier`) + noyau PTQ bit-exact (`int8_c_emulation`). Axe 3-way complet. |
| **TinyOL** | ✅ | ✅ | ✅ | Fake-quant dans la boucle online (`OtOHeadInt8.update_int8`) + calibration PTQ des activations. Axe 3-way exerçable. |
| **HDC** | ✖ structurel | ✖ structurel | ✖ | **Nativement entier** : vecteurs int8 ±1, mémoire associative int16. La quantification n'est pas une *conversion* mais la *structure* du modèle → métrique INT8 ≡ FP32 par construction. Pas d'axe avant/après. |
| **Mahalanobis** | ✖ pas d'entraînement | ✅ (INT8 / Q15) | ✖ | Détecteur **sans entraînement par gradient** (fit statistique). Son axe pertinent n'est pas le moment mais le **format** : INT8 casse Σ⁻¹ (grande dynamique) / Q15 récupère (Sprint 34). |

**Conséquence de cadrage** : la grille 3-way ne concerne que **EWC + TinyOL × {Monitoring, Pronostia}**.
HDC et Maha sont documentés en **contexte** (S4605), avec cellules explicitement **N/A** (structurel /
hors-axe), **jamais remplies par un chiffre artificiel**.

### 3. Réconciliation voie / métrique de référence par modèle

Les deux voies historiques divergent (à documenter, pas à masquer) :

| Voie | Modèle-classe | Tête | Métrique | Loader |
|------|---------------|------|----------|--------|
| QAT (S28) | `EWCMlpInt8Classifier` | binaire sigmoïde | **AUROC** (normal-vs-faute) | `_get_tasks` (benchmark) |
| PTQ (S39) | `EWCMlpMulticlass` | 2-logits | **F1_faulty** | `load_condition_arrays` |

**Décision Sprint 46** : la **voie de référence par modèle est la voie QAT binaire (AUROC)** pour EWC,
car c'est celle qui possède les trois moments dans une seule classe de modèle. Le noyau PTQ
`int8_c_emulation` (tête 2-logits) est **adapté** au chemin `both` en réutilisant `from_state_dict` sur
les poids `fc1/fc2/fc3` de `EWCMlpInt8Classifier` (mêmes noms). La métrique reportée reste **native au
modèle** : AUROC pour EWC, erreur de reconstruction / F1 pour TinyOL. Ce choix est tracé ici pour éviter
toute comparaison AUROC↔F1 trompeuse. `TODO(arnaud)` : valider ce choix de métrique de référence.

### 4. Clé de configuration `quant_moment`

Introduire une clé unique dans les configs `configs/quant_moment/*.yaml` :

```yaml
model: ewc            # ewc | tinyol
dataset: monitoring   # monitoring | pronostia
quant_moment: both    # before | after | both  (+ fp32 comme baseline implicite)
after_scheme: per_tensor_calib   # only for after/both: legacy_c | per_tensor_calib | per_channel_int8 | q15
seed: 42
metric: auroc         # auroc (ewc) | recon_error/f1 (tinyol)
```

`quant_moment` mappe vers un chemin d'exécution dans `run_s46_quant_moment.py` (S4602) ; `after_scheme`
réutilise les presets `QuantConfig` existants de `int8_c_emulation.py`. Aucun hyperparamètre en dur : la
source de vérité reste les configs (conforme CLAUDE.md).

## Format de sortie

Document `docs/context/quantization_moments.md` structuré :

```
# Moments de quantification — taxonomie CL-Embedded
## 1. Les trois moments (before / after / both)   -> table + schéma cycle de vie
## 2. Mapping par modèle (EWC/TinyOL 3-way ; HDC/Maha N/A)  -> table justifiée
## 3. Métrique & voie de référence par modèle       -> table de réconciliation
## 4. Clé `quant_moment` et presets after_scheme     -> exemple YAML
## 5. Renvois : S28 (QAT), S39 (PTQ), S34 (Q15), S24 (HDC/TinyOL INT8)
```

## Contraintes

- Aucun chiffre de résultat dans ce document (taxonomie pure) — les valeurs viennent de S4603–S4605.
- Le mapping HDC/Maha = N/A doit être **justifié** (structurel / hors-axe), pas laissé vide.
- Vocabulaire cohérent avec `quantization_strategies.md` (format) : ce doc traite le **moment**, l'autre
  le **format** — les deux sont orthogonaux et se citent mutuellement.

## Vérification

```bash
# Le document de contexte existe et couvre les 5 sections
test -f docs/context/quantization_moments.md
grep -c "before\|after\|both" docs/context/quantization_moments.md   # > 0
grep -c "quant_moment" docs/context/quantization_moments.md          # > 0
# Mapping HDC/Maha marqué N/A avec justification
grep -i "structurel\|Q15" docs/context/quantization_moments.md
```

---

## Résolution (implémentée)

✅ **Implémenté**. Document de contexte `docs/context/quantization_moments.md` créé (taxonomie
pure, aucun chiffre de résultat), calqué sur le style de `quantization_strategies.md` (S4202).

- **§1** — table des trois moments (`before`/`after`/`both`) + définition opérationnelle +
  implémentation existante + schéma cycle de vie ASCII + point d'honnêteté (borne haute /
  isole PTQ / fidèle déploiement).
- **§2** — mapping par modèle : EWC ✅ / TinyOL ✅ (3-way) ; **HDC N/A structurel** (nativement
  entier) ; **Mahalanobis N/A hors-axe** (pas d'entraînement gradient → axe = format INT8/Q15).
  Chaque N/A justifié, jamais vide.
- **§3** — réconciliation voie/métrique : voie de référence EWC = **QAT binaire (AUROC)** ;
  noyau PTQ adapté au chemin `both` via `from_state_dict` (noms `fc1/fc2/fc3` identiques FP32/QAT) ;
  logit binaire unique = score AUROC (transformation monotone). `TODO(arnaud)` conservé.
- **§4** — clé `quant_moment` + table de mapping `after_scheme` → preset `QuantConfig` → clé
  RAM/BOPs (`run_s39_quant_sweep`) → format (`quantization_strategies.md`).
- **§5** — renvois S28/S39/S34/S24 + axe orthogonal (format vs moment).

**Vérification** :

```
$ test -f docs/context/quantization_moments.md && echo OK           # OK
$ grep -c quant_moment docs/context/quantization_moments.md         # > 0
$ grep -ci "structurel\|Q15" docs/context/quantization_moments.md   # > 0
```
