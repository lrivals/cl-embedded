# S4501 — Sélection des détecteurs à porter & cadrage protocole

| Champ | Valeur |
|-------|--------|
| **Sprint** | 45 |
| **Priorité** | 🔴 Critique — décide *quoi* porter et *comment* l'activer sans casser le protocole. |
| **Statut** | 📝 Doc — spec ; sélection finale à valider par la reco S44 mesurée. |
| **Durée estimée** | 3h |
| **Dépendances** | Sprint 44 S4406 ✅ (reco MCU tracée) · `firmware/.../src/pipeline.c` ✅ (protocole UART v3, nibble saturé) · Sprint 29 S2912 / Sprint 38 S3803 (précédents de sélection à la compilation) |
| **Fichiers cibles** | ce document (cadrage) → consommé par S4502/S4503 |
| **Références** | `docs/context/drift_detectors.md` (§ reco MCU) · CLAUDE.md § « ne pas bypasser le protocole UART » |

---

## Contexte

Le S44 a classé les détecteurs par portabilité. Cette tâche **fige la liste** à porter et **cadre
l'intégration** dans le firmware, en respectant l'invariant clé du projet : le nibble de flags UART est
**saturé** (0x10..0xF0 attribués) → aucun nouveau flag protocole, sélection à la **compilation**.

## Spec

### 1. Sélection (justifiée par les chiffres S44)

Retenir, par ordre de priorité (à confirmer par la mesure S4504) :

| Détecteur | Famille | Justification portage | Décision |
|-----------|---------|----------------------|:--------:|
| **Page-Hinkley** | supervisé O(1) | état minimal, latence négligeable | ✅ porter |
| **DDM** (et/ou **EDDM**) | supervisé O(1) | état O(1), référence supervisée | ✅ porter |
| **PSI / Jensen-Shannon** | non-supervisé O(bins) | état borné indépendant de W, sans label → autonomie | ✅ porter |
| **`SlidingWindowDriftDetector`** | baseline O(W) | déjà porté (`drift_detector.c`, S3803) | ✅ référence |
| **ADWIN** | non-supervisé O(log W) | à valider budget `.bss` (borne de buckets) | 🟡 conditionnel |
| **KSWIN / KS / MMD** | test deux-échantillons O(W)+tri | coût/tri élevé | ⚠️ PC-only si non tenable |

La liste finale est **traçable** à un chiffre de `exp_S44_PC_*` (état/latence proxy) — pas un choix
arbitraire.

### 2. Cadrage protocole

- **Pas de nouveau flag UART.** Activation par `-DDRIFT_DETECT` (build-time), comme `-DEWC_AUTO_UPDATE`
  (S3803) et `-DMAHA_INT8` (S2912). Build par défaut strictement inchangé → 0 régression.
- **Choix du détecteur porté** par `-DDRIFT_METHOD=<page_hinkley|ddm|psi|...>` (ou compilation séparée
  par méthode) — une méthode par binaire, mesurée indépendamment (miroir `run_s29_board_extend`).
- **Réponse UART V3 réutilisée** (23 B) : le verdict de drift est remonté en **réinterprétant un champ du
  snapshot** (précédent S3805 : `snap.auroc ← verdict`) sous `-DDRIFT_DETECT` uniquement → wire format
  inchangé, `sensor_stream.py` intact.
- **Verdict 3→2 niveaux** : le PC produit NORMAL/WARNING/DRIFT ; le board remonte NORMAL/DRIFT (WARNING
  agrégé à NORMAL ou remonté selon `TODO(arnaud)` S4500).
- **Source du signal board** : par défaut, brancher les détecteurs **non-supervisés** sur le **score Maha
  existant** (`maha_score`, déjà calculé dans le pipeline → 0 coût d'acquisition) ; les **supervisés** sur
  le flux d'erreur `1[pred ≠ g_recv_label]` (label déjà transmis).

## Contraintes

- Zéro modification du wire format UART (CLAUDE.md).
- Sélection à la compilation uniquement ; le build par défaut ne lie pas de nouveau coût `.bss`.
- Toute taille (fenêtre, bins, buckets) surchargeable en `#define`.

## Critères d'acceptation

1. Liste finale des détecteurs à porter figée, chaque entrée tracée à un chiffre S44.
2. Mécanisme d'activation (`-DDRIFT_DETECT` + `-DDRIFT_METHOD`) spécifié, sans flag UART neuf.
3. Stratégie de remontée du verdict (réinterprétation de champ snapshot) décrite, wire format préservé.
4. Source du signal (score Maha / flux d'erreur) tranchée pour chaque famille.
