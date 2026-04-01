# Contexte du projet — Vue d'ensemble

## Titre et périmètre

**Projet** : Apprentissage Incrémental pour Systèmes Embarqués à Ressources Limitées  
**Stage M2** : ISAE-SUPAERO (DISC) × ENAC (LII) × Edge Spectrum  
**Période** : 16 mars – 6 août 2026  
**Auteur** : Léonard Rivals

Ce dépôt couvre exclusivement l'**Objectif 1** du stage : implémentation et comparaison de méthodes de continual learning (CL) pour systèmes embarqués, avec application à la maintenance prédictive industrielle.

---

## Problème scientifique

Les systèmes industriels embarqués font face à un problème fondamental : les modèles de machine learning déployés sur microcontrôleurs sont **statiques**. Lorsque la distribution des données change (usure d'un équipement, changement de conditions opérationnelles, nouveau type de machine), le modèle ne peut pas s'adapter sans être entièrement réentraîné hors-device et re-flashé.

Ce réentraînement batch est :
- **Coûteux** (mobilise une infrastructure de calcul)
- **Lent** (indisponibilité du capteur)
- **Impossible** dans certains contextes (déploiements isolés, contraintes de bande passante)

Le **continual learning (apprentissage incrémental)** propose une alternative : le modèle s'adapte directement sur le capteur, en ligne, échantillon par échantillon ou tâche par tâche, sans oublier les connaissances acquises (problème du *catastrophic forgetting*).

---

## Positionnement scientifique — Le triple gap

La revue systématique de 20 articles (corpus complet au 1er avril 2026) révèle qu'**aucun travail** ne satisfait simultanément trois critères :

### Gap 1 — Données industrielles réelles
La quasi-totalité des travaux valident leurs méthodes CL sur des datasets d'images (CIFAR, ImageNet, CORe50) ou des données simulées. Aucun ne démontre les performances sur des séries temporelles industrielles de maintenance prédictive avec un protocole reproductible public.

**Référence de positionnement** : FEMTO PRONOSTIA (Nectoux et al., 2012) — dataset de dégradation de roulements, accès public, benchmark scientifique reconnu.

### Gap 2 — Contrainte mémoire sub-100 Ko avec chiffres précis
Les travaux TinyML CL reportent au mieux des contraintes de quelques Mo (LifeLearner : 212 Ko SRAM sur STM32H747). Aucun ne démontre un CL complet sous 100 Ko avec des mesures RAM précises par composant.

**Cible du stage** : STM32N6, ~64 Ko RAM, avec profiling mémoire composant par composant.

### Gap 3 — Quantification INT8 pendant l'entraînement incrémental
La quantification INT8 est universellement appliquée à l'inférence (post-training quantization, PTQ), jamais à la backpropagation incrémentale. QLR-CL (Ravaglia et al., 2021) quantifie le buffer de rejeu en UINT8 mais maintient la backprop en FP32.

**Exploration du stage** : tester la backprop INT8 sur un MLP minimal (< 500 paramètres) pour attaquer ce gap.

---

## Taxonomie des méthodes CL (De Lange et al., 2021)

```
Continual Learning
├── Regularization-based
│   ├── EWC (Kirkpatrick et al., 2017)          ← M2 de ce projet
│   ├── Online EWC (Schwarz et al., 2018)
│   ├── SI (Zenke et al., 2017)
│   ├── LwF (Li & Hoiem, 2018)
│   ├── MAS (Aljundi et al., 2018)
│   └── Gradient Monitoring (Shah & Schwung, 2025)
│
├── Replay-based
│   ├── Experience Replay (résumé d'échantillons)
│   ├── QLR-CL (Ravaglia et al., 2021)          ← rejeu latent UINT8
│   └── LifeLearner (Kwon et al., 2023)
│
├── Architecture-based
│   ├── TinyOL (Ren et al., 2021)               ← M1 de ce projet
│   ├── HDC (Benatti et al., 2019)              ← M3 de ce projet
│   └── Progressive Networks
│
└── Hybrid
    ├── AR1* (Pellegrini et al., 2021)
    └── Adaptive CL (Wu et al., 2025)
```

---

## Scénarios CL implémentés

| Scénario | Description | Dataset | Modèle |
|----------|-------------|---------|--------|
| **Domain-Incremental** | Même tâche, distribution change par domaine | Dataset 2 — Monitoring | M2 (EWC), M3 (HDC) |
| **Domain-Incremental temporel** | Drift continu sans frontière de tâche explicite | Dataset 1 — Pump | M1 (TinyOL) |

---

## Encadrants

| Nom | Rôle | Affiliation | Contact prioritaire pour |
|-----|------|-------------|--------------------------|
| **Arnaud Dion** | Superviseur principal | ISAE-SUPAERO (DISC) | Analyse comparative, positionnement scientifique |
| **Dorra Ben Khalifa** | Co-superviseure | ENAC (LII) | Hardware, quantification, faisabilité MCU |
| **Frédéric Zbierski** | Superviseur industriel | Edge Spectrum | Cas d'usage industriel, registre applicatif |

---

## Livrables du stage

| Livrable | Deadline | Statut |
|---------|---------|--------|
| Manuscrit préliminaire (revue de littérature) | 15 avril 2026 | En cours |
| Prototype Python PC (3 modèles) | ~15 mai 2026 | 🔲 Ce dépôt |
| Portage MCU STM32N6 | ~15 juin 2026 | 🔲 Phase 2 |
| Rapport final | 6 août 2026 | 🔲 |

---

## Questions techniques ouvertes (à résoudre avec Dorra)

1. **NPU STM32N6** : confirmer que le NPU est strictement inférence-only (pas de backward graph)
2. **Compatibilité rejeu latent + INT8** : les activations quantifiées en UINT8 sont-elles compatibles avec une backprop en FP32 sur les couches suivantes ?
3. **RAM applicative réelle** : quelle mémoire SRAM reste disponible après HAL, DMA, pile système ?
4. **Sparsité des activations** : les données industrielles (vs images) produisent-elles des activations ReLU suffisamment sparses pour justifier une compression de buffer ?
