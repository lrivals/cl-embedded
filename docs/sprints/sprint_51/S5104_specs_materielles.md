# S5104 — Spécifications matérielles (PC + NUCLEO) pour la soutenance

| Champ | Valeur |
|-------|--------|
| **Sprint** | 51 |
| **Priorité** | 🟢 Bas — contexte de comparaison pour les slides (CR §1 + §7). |
| **Statut** | 📝 Doc — spec complète ; implémentation à venir |
| **Durée estimée** | 2h |
| **Dépendances** | `configs/hw_profile_f439zi.yaml` ✅ · specs PC (à relever sur la machine) |
| **Fichiers cibles** | `docs/context/hardware_comparison.md` (nouveau) |
| **Références** | CR §1 « Interprétation » · §7 action 🟢 |

## Contexte

Le CR souligne que la comparaison PC↔board n'est **pas directement exploitable** : le PC multi-cœur partage les
tâches (overhead de scheduling) là où la carte exécute séquentiellement sur un cœur dédié. Pour la soutenance,
il faut **détailler les deux matériels** afin de contextualiser honnêtement les écarts observés.

## Spec

### 1. Fiche des 2 matériels

| Caractéristique | PC | NUCLEO-F439ZI |
|-----------------|----|--------------|
| CPU / cœurs | (à relever : modèle, nb cœurs, fréq.) | Cortex-M4 mono-cœur @ 180 MHz |
| RAM | (à relever) | 256 Ko (192 SRAM + 64 CCM) |
| FPU | oui | oui (simple précision) |
| NPU | non | non |
| OS / scheduling | multitâche préemptif (overhead) | bare-metal séquentiel |
| Mesure latence | horloge OS (bruitée) | DWT cycle-exact |

Relever les specs PC réelles (CPU, cœurs, fréquence, RAM, OS) — pas de valeur inventée ; `<à relever>` sinon.

### 2. Interprétation (CR §1)

- La latence plus élevée sur PC ≈ overhead scheduling multi-cœur, **pas** une infériorité de la carte.
- Conséquence : comparer PC↔board comme « qui est meilleur » n'a pas de sens → **classer à plateforme fixée**
  (renvoi S5103) et réorienter vers board↔board (renvoi S5105).

## Format de sortie

`docs/context/hardware_comparison.md` : fiche 2 matériels + interprétation + renvois S5103/S5105.

## Contraintes

- Specs PC réelles ou `<à relever>` (aucune invention).
- Message aligné CR §1 (PC↔board non exploitable).

## Vérification

```bash
test -f docs/context/hardware_comparison.md
grep -i "Cortex-M4\|multi-cœur\|scheduling\|DWT" docs/context/hardware_comparison.md
```
