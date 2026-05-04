# S14-02 — Config YAML `ewc_oneclass_config.yaml`

| Champ | Valeur |
|-------|--------|
| **ID** | S14-02 |
| **Sprint** | Sprint 14 |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 1h |
| **Dépendances** | S14-01 |
| **Fichier cible** | `configs/ewc_oneclass_config.yaml` |

---

## Objectif

Créer le fichier de configuration YAML pour `EWCOneClassDetector` avec des valeurs par défaut respectant la contrainte 64 Ko RAM, et des sections dataset-spécifiques pour Monitoring, Pronostia et CWRU.

---

## Contenu attendu

```yaml
# EWC One-Class Detector — Anomaly Detection Configuration
# Contrainte : RAM totale modèle ≤ 64 Ko (FP32)

MODEL:
  HIDDEN_DIM: 32          # neurons couche cachée encodeur/décodeur
  LATENT_DIM: 8           # dimension espace latent
  LAMBDA_EWC: 400.0       # poids régularisation EWC (0 = pas d'EWC)
  THRESHOLD_PERCENTILE: 95  # percentile MSE train pour seuil anomalie

TRAINING:
  N_EPOCHS: 20
  LR: 0.001
  BATCH_SIZE: 32
  DEVICE: "cpu"

DATASETS:
  monitoring:
    INPUT_DIM: 4
    # RAM estimée : (4*32 + 32*8 + 8*32 + 32*4) * 2 * 4 = 2 048 B @ FP32
  pronostia:
    INPUT_DIM: 13
    HIDDEN_DIM: 64          # override pour plus grande capacité
    # RAM estimée : (13*64 + 64*16 + 16*64 + 64*13) * 2 * 4 ≈ 13 Ko @ FP32
  cwru:
    INPUT_DIM: 9
    HIDDEN_DIM: 32
    # RAM estimée : (9*32 + 32*8 + 8*32 + 32*9) * 2 * 4 ≈ 5 Ko @ FP32

REPRODUCIBILITY:
  SEED: 42
```

---

## Critères d'acceptation

- [ ] `configs/ewc_oneclass_config.yaml` charge sans erreur avec PyYAML
- [ ] Constantes `HIDDEN_DIM`, `LATENT_DIM`, `LAMBDA_EWC`, `THRESHOLD_PERCENTILE` présentes sous `MODEL:`
- [ ] Section `DATASETS:` avec les 3 datasets et leur `INPUT_DIM` respectif
- [ ] Commentaires RAM estimée présents pour chaque dataset
- [ ] `SEED: 42` présent (reproductibilité)

## Statut

⬜ À faire
