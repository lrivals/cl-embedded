# Analyse des Datasets — CL-Embedded

> Auteur : Léonard Rivals | Stage ISAE-SUPAERO / ENAC | Mise à jour : avril 2026

---

## Table des matières

1. [PRONOSTIA (IEEE PHM 2012)](#1-pronostia-ieee-phm-2012)
   - [Origine et contexte](#origine-et-contexte)
   - [Dispositif expérimental](#dispositif-expérimental)
   - [Structure des fichiers](#structure-des-fichiers)
   - [Conditions opératoires et inventaire des roulements](#conditions-opératoires-et-inventaire-des-roulements)
   - [Pipeline de prétraitement dans ce projet](#pipeline-de-prétraitement-pronostia)
2. [CWRU Bearing Dataset](#2-cwru-bearing-dataset)
   - [Origine et contexte](#origine-et-contexte-1)
   - [Dispositif expérimental](#dispositif-expérimental-1)
   - [Taxonomie des défauts et fichiers](#taxonomie-des-défauts-et-fichiers)
   - [Pipeline de prétraitement dans ce projet](#pipeline-de-prétraitement-cwru)
3. [Comparaison synthétique des deux datasets](#3-comparaison-synthétique)
4. [Expériences réalisées dans ce projet](#4-expériences-réalisées-dans-ce-projet)
   - [Scénarios CL testés](#scénarios-cl-testés)
   - [Résultats — PRONOSTIA by_condition](#résultats--pronostia-by_condition)
   - [Résultats — CWRU by_fault_type](#résultats--cwru-by_fault_type)
   - [Résultats — CWRU by_severity](#résultats--cwru-by_severity)
   - [Tableau récapitulatif global](#tableau-récapitulatif-global)

---

## 1. PRONOSTIA (IEEE PHM 2012)

### Origine et contexte

| Champ | Valeur |
|-------|--------|
| **Nom officiel** | FEMTO-ST IEEE PHM 2012 Data Challenge Dataset |
| **Organisme** | Institut FEMTO-ST, Besançon (Franche-Comté Électronique Mécanique Thermique et Optique) |
| **Publication de référence** | Nectoux et al., *PRONOSTIA: An Experimental Platform for Bearings Accelerated Life Test*, IEEE PHM 2012, Denver, CO |
| **Usage principal** | Pronostic de durée de vie restante (RUL — Remaining Useful Life) |
| **Licence** | Libre (diffusé publiquement lors du challenge, citer la publication ci-dessus) |

PRONOSTIA est une plateforme expérimentale conçue pour soumettre des roulements à billes à des conditions de charge et de vitesse accélérées, jusqu'à leur défaillance complète. Chaque essai couvre un cycle de vie entier d'un roulement, de l'état sain à la casse. Le dataset a servi de référence pour la compétition IEEE PHM 2012 Data Challenge.

### Dispositif expérimental

- **Composant testé** : roulement à billes (ball bearing)
- **2 capteurs vibratoires** : accéléromètre horizontal (acc_H) + accéléromètre vertical (acc_V)
- **Fréquence d'échantillonnage** : **25 600 Hz** (25,6 kHz)
- **Mode d'acquisition** : rafales de 0,1 s toutes les 10 s — soit **2 560 échantillons par fichier**
- **Capteur température** : enregistrement synchrone (600 échantillons par fichier température)
- **Amplitudes de vibration** : −48 à +48 g (état sain : ≈ ±2 g, proche défaillance : ≈ ±48 g)

### Structure des fichiers

```
Pronostia dataset/
├── Learning_set/        ← 6 roulements d'entraînement (cycle complet jusqu'à casse)
│   ├── Bearing1_1/      ← 2803 fichiers acc_*.csv + 466 fichiers temp_*.csv
│   ├── Bearing1_2/
│   ├── Bearing2_1/
│   ├── Bearing2_2/
│   ├── Bearing3_1/
│   └── Bearing3_2/
├── Test_set/            ← 11 roulements de test (arrêtés avant casse complète)
│   ├── Bearing1_3/ ... Bearing1_7/
│   ├── Bearing2_3/ ... Bearing2_7/
│   └── Bearing3_3/
├── Full_Test_Set/
├── IEEEPHM2012-Challenge-Details.pdf
└── README.md

binaries/                ← 6 fichiers .npy concaténés (Learning_set uniquement)
├── 0.npy  ←→  Bearing1_1   (shape : 7 175 680 × 6, ~345 MB)
├── 1.npy  ←→  Bearing1_2   (shape : 2 229 760 × 6, ~107 MB)
├── 2.npy  ←→  Bearing2_1   (shape : 2 332 160 × 6, ~112 MB)
├── 3.npy  ←→  Bearing2_2   (shape : 2 040 320 × 6, ~98 MB)
├── 4.npy  ←→  Bearing3_1   (shape : 1 318 400 × 6, ~63 MB)
└── 5.npy  ←→  Bearing3_2   (shape : 4 190 720 × 6, ~201 MB)
Total binaries : ~926 MB (float64, 6 colonnes)
```

**Format d'un fichier CSV d'accélération (6 colonnes, sans en-tête)** :

| Col | Contenu | Unité |
|-----|---------|-------|
| 0 | Heure | h |
| 1 | Minute | min |
| 2 | Seconde | s |
| 3 | Microseconde | µs |
| **4** | **Accélération horizontale** | **g** |
| **5** | **Accélération verticale** | **g** |

### Conditions opératoires et inventaire des roulements

Le dataset couvre 3 régimes de fonctionnement distincts (charge/vitesse), ce qui constitue le fondement du scénario CL « by_condition » :

| Condition | Vitesse | Charge radiale | Roulements |
|-----------|---------|----------------|------------|
| **Cond. 1** | 1 800 rpm | 4 000 N | Bearing1_x |
| **Cond. 2** | 1 650 rpm | 4 200 N | Bearing2_x |
| **Cond. 3** | 1 500 rpm | 5 000 N | Bearing3_x |

**Durée de vie mesurée (Learning_set)** :

| Bearing | Condition | Fichiers acc | Durée de vie |
|---------|-----------|:------------:|:------------:|
| Bearing1_1 | Cond. 1 | 2 803 | **4,7 min** |
| Bearing1_2 | Cond. 1 | 871 | 1,5 min |
| Bearing2_1 | Cond. 2 | 911 | 1,5 min |
| Bearing2_2 | Cond. 2 | 797 | 1,3 min |
| Bearing3_1 | Cond. 3 | 515 | 0,9 min |
| Bearing3_2 | Cond. 3 | 1 637 | 2,7 min |

**Test_set** (11 roulements, durées tronquées avant casse) :

| Bearing | Condition | Fichiers acc | Durée |
|---------|-----------|:------------:|:-----:|
| Bearing1_3 | Cond. 1 | 1 802 | 3,0 min |
| Bearing1_4 | Cond. 1 | 1 139 | 1,9 min |
| Bearing1_5 | Cond. 1 | 2 302 | 3,8 min |
| Bearing1_6 | Cond. 1 | 2 302 | 3,8 min |
| Bearing1_7 | Cond. 1 | 1 502 | 2,5 min |
| Bearing2_3 | Cond. 2 | 1 202 | 2,0 min |
| Bearing2_4 | Cond. 2 | 612 | 1,0 min |
| Bearing2_5 | Cond. 2 | 2 002 | 3,3 min |
| Bearing2_6 | Cond. 2 | 572 | 1,0 min |
| Bearing2_7 | Cond. 2 | 172 | **0,3 min** |
| Bearing3_3 | Cond. 3 | 352 | 0,6 min |

**Total** : 17 roulements — 21 493 fichiers acc — ~55 millions d'échantillons bruts

### Pipeline de prétraitement (Pronostia)

*(implémenté dans `src/data/pronostia_dataset.py`)*

```
.npy brut (6 colonnes)
  → extraction canaux acc_H et acc_V
  → fenêtrage sans overlap : WINDOW_SIZE = 2 560 pts (= 0,1 s @ 25,6 kHz)
  → extraction de 13 features par fenêtre :
      6 stats × 2 canaux = 12 features
      [mean, std, RMS, kurtosis, peak, crest_factor]  par canal
      + 1 feature de position temporelle relative (0→1)
  → labellisation binaire :
      derniers 10% des fenêtres → label = 1 (pré-défaillance)
      premiers 90% → label = 0 (état sain)
  → normalisation Z-score (mean/std fittés sur Condition 1 uniquement)
  → split stratifié train/val (80/20)
```

**Pourquoi 10% comme seuil de défaillance ?**  
La dégradation d'un roulement PRONOSTIA s'accélère fortement dans la phase finale — cette convention est standard dans la littérature PHM pour définir la zone de pré-défaillance sans disposer d'annotation manuelle de l'instant de défaut.

---

## 2. CWRU Bearing Dataset

### Origine et contexte

| Champ | Valeur |
|-------|--------|
| **Nom officiel** | Case Western Reserve University Bearing Data Center Dataset |
| **Organisme** | CWRU, en partenariat avec Rockwell (système IQ PreAlert) |
| **Usage principal** | Diagnostic de pannes (fault detection & classification) |
| **Statut** | Référence de la communauté — >15 000 citations, benchmark incontournable du Bearing Fault Diagnosis |

Ce dataset a été produit pour valider et améliorer les techniques d'évaluation de l'état des moteurs. Les défauts sont introduits artificiellement par électro-érosion (EDM machining) à taille contrôlée, sur trois localisations possibles du roulement.

### Dispositif expérimental

- **Moteur** : 2 HP avec charge variable (ici : 1 HP)
- **Vitesse de rotation** : ~1 772 rpm (1 HP)
- **Fréquence d'échantillonnage** : **48 000 Hz** (48 kHz)
- **Durée par enregistrement** : ~10 secondes (~485 000 échantillons)
- **Capteurs** : 2 accéléromètres — Drive End (DE) + Fan End (FE)
- **Tailles de défaut** : 0,007'' / 0,014'' / 0,021'' (0,178 / 0,356 / 0,533 mm)
- **Localisations** : bille (Ball), bague intérieure (Inner Race), bague extérieure (Outer Race)

### Taxonomie des défauts et fichiers

**10 classes au total (3 localisations × 3 tailles + état normal)** :

| Classe | Localisation | Taille défaut | Échantillons |
|--------|-------------|:-------------:|:------------:|
| Normal | — | — | 460 |
| Ball_007 | Bille | 0,007'' | 460 |
| Ball_014 | Bille | 0,014'' | 460 |
| Ball_021 | Bille | 0,021'' | 460 |
| IR_007 | Bague intérieure | 0,007'' | 460 |
| IR_014 | Bague intérieure | 0,014'' | 460 |
| IR_021 | Bague intérieure | 0,021'' | 460 |
| OR_007 | Bague extérieure | 0,007'' | 460 |
| OR_014 | Bague extérieure | 0,014'' | 460 |
| OR_021 | Bague extérieure | 0,021'' | 460 |

**Dataset parfaitement balancé** : 230 fenêtres/classe dans le CSV features, 460 dans le NPZ CNN.

#### Fichiers disponibles

**Données brutes** (`raw/*.mat`, format MATLAB) :

| Fichier | Signaux | Durée |
|---------|---------|-------|
| `Time_Normal_1_098.mat` | DE + FE | ~10 s |
| `IR007_1_110.mat` ... `OR021_6_1_239.mat` | DE + FE + RPM | ~10 s |

**Données prétraitées** :

| Fichier | Format | Shape | Description |
|---------|--------|-------|-------------|
| `feature_time_48k_2048_load_1.csv` | CSV | 2 300 × 10 | 9 features temporelles + colonne `fault` |
| `CWRU_48k_load_1_CNN_data.npz` | NPZ | data:(4 600, 32, 32) + labels:(4 600,) | Matrices 2D pour CNN |

**9 features extraites par fenêtre de 2 048 points** (= 0,043 s @ 48 kHz) :
`max`, `min`, `mean`, `sd`, `RMS`, `skewness`, `kurtosis`, `crest factor`, `form factor`

### Pipeline de prétraitement (CWRU)

*(implémenté dans `src/data/cwru_dataset.py`)*

```
feature_time_48k_2048_load_1.csv  (2300 lignes × 9 features)
  → lecture pandas + sélection des colonnes features
  → labellisation binaire : Normal_1 → 0, tout défaut → 1
  → regroupement selon le scénario CL :
      by_fault_type : Ball | Inner Race | Outer Race  (3 tâches)
      by_severity   : 0.007" | 0.014" | 0.021"       (3 tâches)
  → Normal réparti uniformément entre les 3 tâches (shuffle, seed=42)
  → StandardScaler fitté sur train de la Task 0 uniquement
  → split stratifié train/val (80/20, seed=42)
```

**Deux scénarios CL distincts** :
- **by_fault_type** : le modèle apprend d'abord à reconnaître les défauts de bille, puis de bague intérieure, puis de bague extérieure → test la généralisation inter-type
- **by_severity** : le modèle apprend d'abord les petits défauts (0,007''), puis moyens (0,014''), puis grands (0,021'') → test l'adaptation à la sévérité croissante

---

## 3. Comparaison synthétique

| Critère | PRONOSTIA | CWRU |
|---------|-----------|------|
| **Tâche** | Pronostic / détection anomalie (RUL, binaire) | Diagnostic (classification multi-classe) |
| **Signal source** | Vibration brute, acquisition continue | Vibration brute, état stationnaire |
| **Fréquence** | 25,6 kHz | 48 kHz |
| **Nb de roulements** | 17 (6 train + 11 test) | 10 (1 par classe) |
| **Durée par run** | 0,3 à 4,7 min (cycle complet → casse) | ~10 s (état stationnaire stable) |
| **Labels** | Binaire (sain/pré-défaillance) | 10 classes (3 loc × 3 tailles + Normal) |
| **Balancement** | Non (durées très hétérogènes entre bearings) | Parfaitement balancé (460/classe) |
| **Scénario CL naturel** | Domain-incremental par condition opératoire | Domain-incremental par type ou sévérité |
| **Nombre de tâches** | 3 (Cond. 1 → 2 → 3) | 3 (Ball → IR → OR, ou 007 → 014 → 021) |
| **Taille totale** | ~926 MB (binaries, float64) | ~38 MB (NPZ + CSV) |
| **Format source** | CSV + NPY | MAT (MATLAB) + CSV + NPZ |
| **Utilisation** | Détection anomalie vibratoire | Classification état roulement |

---

## 4. Expériences réalisées dans ce projet

### Scénarios CL testés

Chaque dataset a été évalué avec les 6 modèles du projet dans un schéma **domain-incremental à 3 tâches séquentielles** (sans accès aux données passées). Trois baselines sont systématiquement comparées :

- **Naive** (fine-tuning pur) : entraîne séquentiellement sans aucune protection → mesure le catastrophic forgetting maximal
- **Joint** (upper bound) : accès à toutes les données simultanément → performance plafond
- **CL** (modèle évalué) : méthode incrémentale testée

**6 modèles évalués** :

| ID | Modèle | Famille | Paramètres clés |
|----|--------|---------|-----------------|
| EWC | EWC Online + MLP | Regularization-based | λ=500, hidden=32, lr=1e-3 |
| HDC | Hyperdimensional Computing | Architecture-based | D=1024 dim, bipolar {−1,+1} |
| TinyOL | TinyOL + tête OtO | Architecture-based | encoder=391 params, OtO head |
| KMeans | K-Means online | Unsupervised baseline | k=2 (sain/défaut) |
| Mahalanobis | Distance de Mahalanobis | Unsupervised baseline | fit covariance online |
| DBSCAN | Density-Based Clustering | Unsupervised baseline | eps adaptatif |

**Métriques systématiquement reportées** :

| Métrique | Description |
|----------|-------------|
| `AA` (Average Accuracy) | Accuracy moyenne sur toutes tâches après entraînement complet |
| `AF` (Average Forgetting) | Chute moyenne d'accuracy entre pic et fin par tâche |
| `BWT` (Backward Transfer) | Impact de l'apprentissage futur sur tâches passées (négatif = oubli) |
| `ram_peak_bytes` | RAM maximale mesurée à l'exécution (tracemalloc) |
| `inference_latency_ms` | Latence forward pass (moyenne sur 100 runs) |
| `n_params` | Nombre de paramètres entraînables |
| `within_budget_64ko` | Contrainte STM32N6 respectée ? |

---

### Résultats — PRONOSTIA by_condition

**Scénario** : Task 1 = Cond. 1 (1 800 rpm) → Task 2 = Cond. 2 (1 650 rpm) → Task 3 = Cond. 3 (1 500 rpm)  
**Tâche** : détection binaire pré-défaillance sur 13 features vibratoires  
**Experiments** : `exp_044` à `exp_055`

#### Baseline single-task (toutes conditions mélangées)

| Modèle | Accuracy | F1 | AUC-ROC | RAM peak | Latence |
|--------|:--------:|:--:|:-------:|:--------:|:-------:|
| EWC (exp_044) | 96,0% | 0,758 | 0,902 | 1 171 B | 0,047 ms |
| TinyOL (exp_046) | 94,9% | 0,670 | **0,968** | 944 B | **0,009 ms** |
| KMeans (exp_047) | 90,0% | 0,375 | 0,901 | 5 620 B | 0,516 ms |
| Mahalanobis (exp_048) | 88,7% | 0,280 | 0,885 | 1 756 B | 0,008 ms |
| DBSCAN (exp_049) | 88,5% | 0,258 | 0,844 | **267 416 B** ❌ | 0,362 ms |
| HDC (exp_045) | 63,6% | 0,285 | 0,674 | 14 640 B | 0,046 ms |

#### Évaluation Continual Learning (by_condition)

| Modèle | AA | AF | BWT | RAM peak | Latence | Params | Dans 64 Ko |
|--------|:--:|:--:|:---:|:--------:|:-------:|:------:|:----------:|
| **EWC** (exp_050) | **98,2%** | **0,0%** | +0,49% | **1 171 B** | **0,035 ms** | 993 | ✅ |
| *Naive (référence)* | *91,7%* | *10,8%* | −10,8% | — | — | — | — |
| *Joint (plafond)* | *99,2%* | *0%* | — | — | — | — | — |
| TinyOL (exp_052) | 93,0% | 2,0% | −2,0% | **3 698 B** | **0,009 ms** | 542 | ✅ |
| KMeans (exp_053) | 89,0% | 3,1% | −3,1% | 5 574 B | 0,319 ms | 26 | ✅ |
| DBSCAN (exp_055) | 90,1% | 0,0% | +0,51% | 121 024 B ❌ | 0,271 ms | 11 479 | ❌ |
| HDC (exp_051) | 80,5% | 4,5% | −4,5% | 14 504 B | 0,119 ms | 2 048 | ✅ |
| Mahalanobis (exp_054) | 79,3% | 16,9% | −16,9% | 1 756 B | 0,008 ms | 182 | ✅ |

**Points saillants** :
- EWC est le meilleur modèle sur Pronostia : AA=98,2%, AF=0% — il ne souffre quasiment pas d'oubli catastrophique alors que Naive perd 10,8%
- Le naive forgetting est massif sur Pronostia (−10,8%) car les 3 conditions ont des distributions vibratoires très différentes
- TinyOL est le plus rapide (0,009 ms) et le plus léger (3 698 B), au prix d'une accuracy un peu inférieure
- Mahalanobis souffre d'oubli catastophique (+17%) — la covariance fittée sur une condition est peu transférable
- DBSCAN dépasse le budget 64 Ko (121 KB) → incompatible MCU

---

### Résultats — CWRU by_fault_type

**Scénario** : Task 1 = Ball → Task 2 = Inner Race → Task 3 = Outer Race  
**Tâche** : classification binaire (sain/défaut) sur 9 features temporelles  
**Experiments** : `exp_074` à `exp_079`

| Modèle | AA | AF | BWT | RAM peak | Latence | Params | Dans 64 Ko |
|--------|:--:|:--:|:---:|:--------:|:-------:|:------:|:----------:|
| **EWC** (exp_074) | **100%** | **0%** | **0%** | **1 171 B** | 0,020 ms | 865 | ✅ |
| TinyOL (exp_076) | 96,6% | 0,2% | +0,65% | 4 055 B | **0,005 ms** | 397 | ✅ |
| HDC (exp_075) | 93,5% | 4,5% | −3,9% | 7 848 B | 0,052 ms | 1 024 | ✅ |
| Mahalanobis (exp_078) | 31,6% | 1,3% | +28,6% | 1 644 B | 0,004 ms | 90 | ✅ |
| KMeans (exp_077) | 15,2% | 1,9% | +3,9% | 5 432 B | 0,161 ms | 18 | ✅ |
| DBSCAN (exp_079) | 12,6% | 4,5% | −4,5% | 16 788 B | 0,129 ms | 1 116 | ✅ |

**Points saillants** :
- EWC atteint 100% sur ce scénario — les 3 types de défaut sont très séparables dans l'espace des features temporelles, et EWC les mémorise parfaitement
- Les méthodes non supervisées (KMeans, DBSCAN) échouent sur ce scénario multi-classes car elles ne peuvent pas distinguer 3+ types de défauts avec seulement 2 clusters
- TinyOL est le plus rapide (0,005 ms) avec seulement 397 paramètres

---

### Résultats — CWRU by_severity

**Scénario** : Task 1 = 0,007'' → Task 2 = 0,014'' → Task 3 = 0,021''  
**Tâche** : classification binaire sur 9 features temporelles  
**Experiments** : `exp_080` à `exp_085`

| Modèle | AA | AF | BWT | RAM peak | Latence | Params | Dans 64 Ko |
|--------|:--:|:--:|:---:|:--------:|:-------:|:------:|:----------:|
| **EWC** (exp_080) | **95,2%** | **0%** | +0,65% | **1 171 B** | 0,038 ms | 865 | ✅ |
| *Naive (référence)* | *91,8%* | *3,9%* | −3,2% | — | — | — | — |
| *Joint (plafond)* | *97,4%* | *0%* | — | — | — | — | — |
| TinyOL (exp_082) | 90,0% | 0% | +1,3% | 4 055 B | **0,009 ms** | 397 | ✅ |
| HDC (exp_081) | 89,2% | 1,9% | −0,65% | 7 848 B | 0,089 ms | 1 024 | ✅ |
| Mahalanobis (exp_084) | 39,4% | 9,1% | +39,6% | 1 644 B | 0,009 ms | 90 | ✅ |
| KMeans (exp_083) | 30,3% | 6,5% | +28,6% | 5 432 B | 0,300 ms | 18 | ✅ |
| DBSCAN (exp_085) | 12,1% | 29,2% | −1,3% | 31 474 B | 0,221 ms | 2 439 | ✅ |

**Points saillants** :
- Le scénario by_severity est plus difficile que by_fault_type : les features temporelles simples peinent à distinguer différentes sévérités d'un même type de défaut
- EWC gagne encore nettement, avec 0% d'oubli et un AA supérieur à Naive de +3,4 points
- Les méthodes non supervisées continuent d'échouer (DBSCAN : 12%, AF=29%) — la séparabilité binaire sain/défaut change avec la sévérité et ces méthodes ne s'adaptent pas

---

### Tableau récapitulatif global

| Dataset | Scénario | Modèle | AA | AF | RAM peak | Latence | MCU ✅ |
|---------|----------|--------|:--:|:--:|:--------:|:-------:|:------:|
| Pronostia | by_condition | **EWC** | **98,2%** | 0,0% | 1 171 B | 0,035 ms | ✅ |
| Pronostia | by_condition | TinyOL | 93,0% | 2,0% | 3 698 B | **0,009 ms** | ✅ |
| Pronostia | by_condition | KMeans | 89,0% | 3,1% | 5 574 B | 0,319 ms | ✅ |
| Pronostia | by_condition | DBSCAN | 90,1% | 0,0% | 121 024 B | 0,271 ms | ❌ |
| Pronostia | by_condition | HDC | 80,5% | 4,5% | 14 504 B | 0,119 ms | ✅ |
| Pronostia | by_condition | Mahalanobis | 79,3% | 16,9% | 1 756 B | 0,008 ms | ✅ |
| CWRU | by_fault_type | **EWC** | **100%** | **0%** | 1 171 B | 0,020 ms | ✅ |
| CWRU | by_fault_type | TinyOL | 96,6% | 0,2% | 4 055 B | **0,005 ms** | ✅ |
| CWRU | by_fault_type | HDC | 93,5% | 4,5% | 7 848 B | 0,052 ms | ✅ |
| CWRU | by_fault_type | Mahalanobis | 31,6% | 1,3% | 1 644 B | 0,004 ms | ✅ |
| CWRU | by_fault_type | KMeans | 15,2% | 1,9% | 5 432 B | 0,161 ms | ✅ |
| CWRU | by_fault_type | DBSCAN | 12,6% | 4,5% | 16 788 B | 0,129 ms | ✅ |
| CWRU | by_severity | **EWC** | **95,2%** | **0%** | 1 171 B | 0,038 ms | ✅ |
| CWRU | by_severity | TinyOL | 90,0% | 0% | 4 055 B | **0,009 ms** | ✅ |
| CWRU | by_severity | HDC | 89,2% | 1,9% | 7 848 B | 0,089 ms | ✅ |
| CWRU | by_severity | Mahalanobis | 39,4% | 9,1% | 1 644 B | 0,009 ms | ✅ |
| CWRU | by_severity | KMeans | 30,3% | 6,5% | 5 432 B | 0,300 ms | ✅ |
| CWRU | by_severity | DBSCAN | 12,1% | 29,2% | 31 474 B | 0,221 ms | ✅ |

**Budget MCU** : RAM ≤ 65 536 B (64 Ko), latence ≤ 100 ms — contrainte STM32N6

---

### Conclusions générales

1. **EWC domine sur tous les scénarios** : AA > 95% systématiquement, AF ≈ 0% — la régularisation Fisher protège efficacement les poids importants pour les tâches passées, même avec un MLP minimal (< 1 000 paramètres, < 4 Ko RAM).

2. **TinyOL offre le meilleur compromis vitesse/mémoire** : le modèle le plus rapide (0,005–0,009 ms), le plus léger (< 5 Ko), avec des performances acceptables (90–97%). Idéal pour le déploiement MCU temps-réel.

3. **HDC est robuste mais consommateur** : bonnes performances (80–93%) et zéro oubli sur certains scénarios, mais 14 Ko de RAM (D=2048) — compatible 64 Ko mais sans marge.

4. **Les méthodes non supervisées (KMeans, Mahalanobis, DBSCAN) peinent sur les scénarios multi-classes** : conçues pour la détection d'anomalie binaire, elles ne peuvent pas discriminer plusieurs types ou sévérités de défauts. KMeans et DBSCAN chutent à 12–30% sur CWRU.

5. **DBSCAN est incompatible MCU sur Pronostia** : 121 Ko de RAM pour 11 000+ "paramètres" (vecteurs de support) — dépasse le budget de 64 Ko.

6. **L'oubli catastrophique est réel** : sur Pronostia by_condition, Naive perd 10,8% d'accuracy — EWC le réduit à 0%. Sur CWRU by_severity, Naive perd 3,9% — EWC le réduit à 0%.
