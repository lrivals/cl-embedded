# S5005 — Analyse coût/bénéfice de la quantification INT8

| Champ | Valeur |
|-------|--------|
| **Sprint** | 50 |
| **Priorité** | 🟡 Moyen — synthèse décisionnelle demandée par le CR. |
| **Statut** | ✅ Implémenté — `docs/context/int8_cost_benefit.md` : table décision (RAM poids ÷4 / F1 préservé S46 / latence +50 % FPU / énergie « à mesurer ») + nuance honnête RAM poids ≠ RAM système totale (S49) + reproduit-littérature vs contribution propre + recommandation. 0 chiffre en dur. |
| **Durée estimée** | 2h |
| **Dépendances** | S5002 (énergie) · S5004 (latence INT8) · `exp_S49_ram/` (RAM totale, Sprint 49) · benchmarks accuracy (S28/S36/S46) |
| **Fichiers cibles** | `docs/context/int8_cost_benefit.md` |
| **Références** | CR §5 « Analyse coût/bénéfice » + « Justification dans le contexte du stage » |

## Contexte

Répondre à la question du CR : **le gain de RAM (÷4) vaut-il la légère chute d'accuracy et le surcoût éventuel
de latence/énergie ?** Et expliciter **ce qui est reproduit de la littérature** vs **la contribution propre**.

## Spec

### 1. Tableau de décision (par modèle)

| Dimension | FP32 | INT8 | Verdict |
|-----------|------|------|---------|
| RAM totale | réf (S49) | ÷4 (S49) | gain majeur |
| Accuracy / F1 / AUROC | réf | légère baisse (S46 : préservée avec calib v2) | acceptable |
| Latence | réf | ≈ ou surcoût (S5004, paradoxe FPU) | neutre/négatif sur FPU |
| Énergie/inférence | réf | à statuer (S5002) | dépend µJ mesurés |

Valeurs **rechargées** depuis les JSON S49/S50/S28/S46 ; aucune en dur.

### 2. Reproduit vs contribution (CR §5)

| Reproduit (littérature) | Contribution propre |
|--------------------------|---------------------|
| Ravaglia 2021 : INT8 ÷4 RAM, perte < 0.26 % | Application à **EWC** (régularisation) sur **PRONOSTIA** |
| Capogrosso 2024 : taxonomie PTQ/QAT | Compromis **mesuré sur carte réelle** NUCLEO-F439ZI |
| Zhu/Lin : QAS, distorsion gradient INT8 | Paradoxe **latence FPU** (INT8 ≠ gain latence sans NPU) documenté |
| Benatti 2019, Giménez 2022 : on-device, PTQ MCU | Score système intégrant énergie réelle (→ Sprint 51) |

### 3. Recommandation

Synthèse : sur cette carte FPU, INT8 = **gain RAM, pas latence** → pertinent quand la RAM est le goulot ; le
gain latence attendrait une carte INT8 natif (perspective CR).

## Format de sortie

`docs/context/int8_cost_benefit.md` (tableau décision + reproduit/contribution + recommandation).

## Contraintes

- 0 chiffre en dur (renvois/rechargements).
- Honnêteté sur le paradoxe FPU (ne pas survendre INT8 côté latence).

## Vérification

```bash
test -f docs/context/int8_cost_benefit.md
grep -i "Ravaglia\|contribution\|PRONOSTIA\|FPU" docs/context/int8_cost_benefit.md
```
