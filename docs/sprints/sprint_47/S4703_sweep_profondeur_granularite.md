# S4703 — Sweep profondeur × granularité (EWC × Monitoring/Pronostia)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 47 |
| **Priorité** | 🔴 Critique — cœur expérimental du sprint : identifie le « cliff » en bits et l'effet de la granularité. |
| **Statut** | 📝 Doc — spec complète ; implémentation à venir |
| **Durée estimée** | 7h |
| **Dépendances** | S4702 (émulateur étendu + harnais) · S4701 (clés config) |
| **Fichiers cibles** | `configs/quant_depth/ewc_{monitoring,pronostia}_*.yaml`, `experiments/exp_S47_depth/` |
| **Références** | Baselines FP32/INT8 : `exp_S39_ablation/` (S4202) ; voie AUROC S28 |

---

## Contexte

Toute la quantification EWC du projet vit à **8 bits** (INT8) ou **16 bits** (Q15). On ne sait pas **jusqu'où
descendre** avant que l'AUROC s'effondre, ni **combien de bits la per-channel rachète** face à la per-tensor.
Cette tâche balaie ces deux variables frontalement.

## Spec

### 1. Grille expérimentale

**EWC × {Monitoring, Pronostia} × `weight_bits` {8, 6, 4, 3, 2, ternaire, binaire} × `granularity` {per_tensor, per_channel}**
= 2 × 7 × 2 = **28 cellules** (+ 2 baselines FP32 implicites). Activations 8-bit calibrées, symétrie `symmetric`
(l'axe symétrie est isolé en S4704). Seed 42.

Configs générées (une par cellule) `configs/quant_depth/ewc_<dataset>_<bits>_<gran>.yaml`, portant les clés
S4701. Exemple :

```yaml
model: ewc
dataset: pronostia
weight_bits: 3
granularity: per_channel
symmetry: symmetric
act_bits: 8
seed: 42
metric: auroc
```

### 2. Sorties par cellule

`experiments/exp_S47_depth/exp_S47_ewc_<dataset>_<bits>_<gran>.json` :
- `auroc_fp32`, `auroc_quant`, `delta_auroc` (quant − fp32),
- `agreement_vs_fp32` (taux d'accord de prédiction),
- `ram_weight_bytes_theoretical`, `ram_ratio_vs_fp32` (bit-packé, cf. S4701 §2 — **théorique**),
- `config_snapshot`.

Tous `null` tant que le harnais n'a pas tourné.

### 3. Analyses attendues (à remplir depuis les JSON, jamais en dur)

- **Courbe AUROC vs bits**, une ligne par granularité, par dataset → **identification du « cliff »** (le nombre
  de bits où `delta_auroc` franchit un seuil de dégradation, p. ex. −0,02).
- **Écart per-channel − per-tensor** en fonction des bits : hypothèse H1 = la per-channel **repousse le cliff**
  de plusieurs bits (elle isole les canaux à grande dynamique). À confirmer/infirmer par les chiffres.
- **Ratio RAM théorique** en regard : à AUROC préservée, quel est le **plus petit `weight_bits`** viable, et son
  gain RAM (÷8 à INT4, ÷16 à INT2) — **sous réserve de kernel bit-packé** (Sprint 48).

### 4. Table de résultats (gabarit — valeurs `pending`)

| Dataset | granularité | 8 bits | 6 | 4 | 3 | 2 | ternaire | binaire |
|---------|-------------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Monitoring | per_tensor | pending | pending | pending | pending | pending | pending | pending |
| Monitoring | per_channel | pending | pending | pending | pending | pending | pending | pending |
| Pronostia | per_tensor | pending | pending | pending | pending | pending | pending | pending |
| Pronostia | per_channel | pending | pending | pending | pending | pending | pending | pending |

(cellules = `delta_auroc` ; remplies uniquement par exécution du harnais.)

## Contraintes

- **Seed fixe 42** (reproductibilité) ; `config_snapshot` obligatoire par cellule.
- Aucune valeur numérique écrite dans ce doc ni dans les figures avant exécution (`pending`/`null`).
- Réutiliser la même tête FP32 de référence pour toutes les cellules d'un dataset (isole la seule variable
  profondeur/granularité).

## Vérification

```bash
python scripts/run_s47_quant_depth.py --sweep configs/quant_depth/   # ou boucle sur les configs
ls experiments/exp_S47_depth/exp_S47_ewc_*.json | wc -l              # 28 cellules
# Cohérence : delta_auroc renseigné, ram_ratio croît quand weight_bits décroît
python -c "import json,glob; [print(json.load(open(f))['weight_bits'], json.load(open(f))['granularity']) for f in glob.glob('experiments/exp_S47_depth/*.json')]"
```

---

## Résolution (implémentée)

_À compléter lors de l'implémentation._
