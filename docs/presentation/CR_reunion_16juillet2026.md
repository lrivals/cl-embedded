# Compte rendu — Réunion de suivi de stage

**Date :** 16 juillet 2026 — 15h00 à 16h30
**Format :** Visioconférence — tous présents
**Thèmes :** Comparaison PC / carte embarquée · Mesure RAM complète · Quantification INT8 · Références bibliographiques

---

## 1. Comparaison PC vs carte embarquée

### Contexte de l'expérience

- Modèles évalués : **EWC** et **Mahalanobis**
- Configuration : 5 features, dataset **PRONOSTIA Monitoring**
- Conditions contrôlées : même sélection de groupes de données pour l'entraînement, même protocole de mise à jour et d'évaluation — base de comparaison équitable.

### Résultats observés

| Métrique | PC | Carte |
|----------|----|-------|
| Précision | maintenue | maintenue (légère différence) |
| RAM utilisée | légèrement différente | légèrement différente |
| Latence | **plus grande sur PC** | plus faible |

### Interprétation

La latence plus élevée sur PC s'explique par l'architecture multi-cœur : le partage des tâches entre cœurs introduit de l'overhead de scheduling là où la carte exécute séquentiellement sur un seul cœur dédié. Cette comparaison PC/carte n'est donc **pas directement exploitable** : les conditions d'exécution matérielles sont trop différentes pour en tirer des conclusions généralisables.

**Conclusion principale :** l'expérience montre que le modèle peut fonctionner sur carte seule, sans PC sur place — ce qui valide l'objectif d'embarquabilité.

### Réorientation de la démarche

> La mise en place expérimentale (mêmes données, même évaluation) est à **réutiliser pour comparer deux cartes entre elles**, ce qui sera plus pertinent et directement interprétable.

---

## 2. Mesure de la RAM — méthode complète

### Formule retenue

```
RAM totale = .data + .bss + pic de pile
```

- **`.data`** : données initialisées (variables globales avec valeur initiale)
- **`.bss`** : données non initialisées / non constantes (zéro au démarrage)
- **Pic de pile** : maximum atteint par la call stack durant l'exécution

> ⚠️ **Correction importante :** la mesure précédente n'incluait pas le pic de pile — les valeurs de RAM rapportées étaient **incomplètes**. À corriger dans toutes les expériences.

### Méthode : stack painting

Technique retenue pour mesurer le pic de pile :

1. Remplir la zone réservée de la pile avec un **motif binaire connu** (ex. `0xDEADBEEF`) avant l'exécution.
2. Après exécution, scanner la pile pour trouver la première adresse où le motif est encore intact.
3. Le **pic de pile = taille réservée − zone intacte**.

Cette méthode donne une mesure non-intrusive et précise sans nécessiter de débogueur matériel.

### Corrections à apporter aux plots

- Retirer les étiquettes d'étapes liées à des architectures de réseaux de neurones (incohérentes avec les modèles utilisés) — les plots seront plus lisibles et plus cohérents.
- Adapter les instants de mesure du pic de pile aux phases réelles d'exécution des modèles.
- **Remplacer** la mention « RAM modulable » par **« mémoire non constante »** dans tous les documents.

---

## 3. Métrique globale d'évaluation système

### Problème

Aucune métrique unique ne capture simultanément toutes les dimensions de performance. Il faut définir un **indicateur composite** adapté au contexte embarqué.

### Proposition

Définir un score système intégrant :

| Dimension | Métrique |
|-----------|----------|
| Mémoire | RAM totale (`.data + .bss + pic de pile`) |
| Calcul | Latence d'inférence (ms) |
| Énergie | Énergie par inférence (µJ ou mJ) |
| Modèle | Nombre de paramètres / MACs |
| Performance | Accuracy sur le jeu de test |

> L'objectif est de pouvoir **classer les configurations** (modèle × carte × encodage) selon un critère global plutôt que de présenter des métriques isolées.

---

## 4. Mesure de l'énergie électrique

- Récupérer la carte à l'**ENAC** pour les mesures de consommation électrique.
- Utiliser le setup logiciel posté sur le lien partagé.
- **Isoler les mesures** par composant autant que possible (MCU seul, périphériques, capteurs) pour des données détaillées.
- Objectif : obtenir l'énergie consommée **par inférence** pour chaque modèle et configuration.

---

## 5. Quantification FP32 → INT8

### Approche retenue

Quantification proposée pour **EWC** : passage de FP32 vers INT8 (v2 du système, fonctionnel).

### Résultats

| Métrique | FP32 | INT8 | Évolution |
|----------|------|------|-----------|
| Accuracy | référence | légère baisse | ≈ maintenue |
| RAM | référence | **÷ 4** | gain majeur |
| Latence processeur | référence | à détailler | à analyser |

### Point ouvert — latence INT8

La carte n'est pas nativement INT8 (processeur FP32 avec FPU) — convertir vers INT8 implique des étapes supplémentaires de déquantification/requantification dans le pipeline.

> **Tâche :** détailler précisément les étapes supplémentaires introduites par la quantification dans le pipeline d'inférence et quantifier leur coût en cycles.

### Analyse coût/bénéfice

Évaluer si le gain de RAM (÷4) justifie :

- la légère chute d'accuracy,
- le surcoût éventuel de latence lié aux conversions.

### Justification dans le contexte du stage

Il faut expliciter dans le rapport :

- **Ce qu'on a reproduit** depuis la littérature (voir sources ci-dessous).
- **Notre contribution spécifique** : application à EWC sur PRONOSTIA, mesure du compromis sur carte réelle.

### Perspectives

- Tester sur une carte **sans FPU native** (INT8 natif) pour évaluer le gain réel de latence.
- Explorer d'autres schémas de quantification (INT4, mixte).

---

## 6. Références bibliographiques mobilisées

### Quantification et TinyML

| # | Référence | Lien avec nos travaux |
|---|-----------|----------------------|
| 1 | Ravaglia et al. (2021) — *TinyML Platform for On-Device CL with Quantized Latent Replays* | Quantification 8 bits pour CL sur MCU ultra-basse conso ; gain mémoire ÷4, perte < 0.26 % |
| 2 | Capogrosso et al. (2024) — *ML-oriented Survey on TinyML* | Taxonomie PTQ / QAT ; quantification comme levier de déploiement sur MCU |
| 3 | Zhu et al. (2024) — *On-device Training: A First Overview* | QAS pour corriger la distorsion de gradient en INT8 ; systèmes Mandheling, MiniLearn, TTE |
| 5 | Lin et al. (2023) — *Tiny Machine Learning: Progress and Futures* (MIT) | QAS + réduction mémoire d'entraînement ÷2077 ; MCUNet |

### Apprentissage en ligne embarqué

| # | Référence | Lien avec nos travaux |
|---|-----------|----------------------|
| 4 | Benatti et al. (2019) — *Online Learning of EMG Gestures on PULP* | Apprentissage en ligne sur puce, budget énergétique 10 mJ ; quantification des niveaux d'entrée |
| 6 | Giménez et al. (2022) — *On-Device Training on MCU with Federated Learning* | PTQ post-entraînement sur MCU ; entraînement fédéré |

> Ces sources serviront à **contextualiser et justifier** les choix de quantification dans le rapport et la soutenance.

---

## 7. Actions à mener

| Priorité | Tâche | Contexte |
|----------|-------|---------|
| 🔴 Haut | Relancer toutes les expériences avec les conditions définies (RAM complète incluse) | Remplace les mesures incomplètes |
| 🔴 Haut | Créer un fichier de documentation détaillée des consommations RAM par expérience | Export structuré pour le rapport |
| 🔴 Haut | Récupérer la carte à l'ENAC pour mesurer l'énergie électrique | Setup logiciel sur le lien partagé |
| 🟡 Moyen | Détailler le coût de latence INT8 (étapes de conversion dans le pipeline) | Tâche ouverte de la réunion |
| 🟡 Moyen | Corriger les plots (retirer étiquettes réseau de neurones, adapter instants de mesure) | Amélioration lisibilité |
| 🟡 Moyen | Remplacer « RAM modulable » par « mémoire non constante » dans tous les documents | Correction terminologique |
| 🟢 Bas | Détailler les spécificités matérielles du PC dans les slides de soutenance | Contexte de comparaison |
| 🟢 Bas | Définir le score système composite (RAM + latence + énergie + accuracy) | À formaliser pour le rapport |
| 🟢 Bas | Explorer d'autres schémas de quantification (INT4, mixte, carte sans FPU) | Perspectives |

---

## Références (formats complets)

1. Ravaglia, L., Rusci, M., Nadalini, D., Capotondi, A., Conti, F., & Benini, L. (2021). *A TinyML Platform for On-Device Continual Learning with Quantized Latent Replays.* arXiv:2110.10486.
2. Capogrosso, L., Cunico, F., Cheng, D. S., Fummi, F., & Cristani, M. (2024). *A Machine Learning-oriented Survey on Tiny Machine Learning.* arXiv:2309.11932v2.
3. Zhu, S., Voigt, T., Ko, J., & Rahimian, F. (2024). *On-device Training: A First Overview on Existing Systems.* ACM. https://doi.org/10.1145/3696003
4. Benatti, S., Montagna, F., Kartsch, V., Rahimi, A., Rossi, D., & Benini, L. (2019). *Online Learning and Classification of EMG-Based Gestures on a Parallel Ultra-Low Power Platform Using Hyperdimensional Computing.* IEEE Trans. Biomed. Circuits Syst., 13(3), 516–528. https://doi.org/10.1109/TBCAS.2019.2914476
5. Lin, J., Zhu, L., Chen, W.-M., Wang, W.-C., & Han, S. (2023). *Tiny Machine Learning: Progress and Futures.* IEEE Circuits and Systems Magazine. https://doi.org/10.1109/MCAS.2023.3302182
6. Giménez, N. L., Grau, M. M., Centelles, R. P., & Freitag, F. (2022). *On-Device Training of Machine Learning Models on Microcontrollers with Federated Learning.* Electronics, 11(4), 573. https://doi.org/10.3390/electronics11040573
