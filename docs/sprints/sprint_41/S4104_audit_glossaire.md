# S4104 — Audit glossaire & acronymes (`Manuscrit_Final_Rivals/glossary_entries.tex`)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 41 |
| **Statut** | ✅ Audit réalisé (3 juillet 2026) — entrées à ajouter/corriger livrées ci-dessous |
| **Règle** | Le .tex n'est modifié que sur instruction explicite ; ce fichier est la source |

## 1. État des lieux

`glossary_entries.tex` (version Overleaf, **en retard sur le PDF rendu**) : ~50 entrées couvrant
le CL général. Le PDF intermédiaire rendu contenait déjà des acronymes absents du .tex
(FPU, DSP, SRAM, GOPS, FIM, SGD, DBSCAN, PCA, VAE, IoT, MTSAD, CMSIS-NN…) — la version Overleaf
doit être resynchronisée en plus des ajouts ci-dessous.

## 2. Entrées OBSOLÈTES ou à corriger

1. **`STM32N6`** : décrit comme « cible matérielle du stage » — **faux depuis le Sprint 16**.
   → Remplacer par une entrée `NUCLEO-F439ZI` (reprendre la définition du glossaire du PDF
   intermédiaire, qui l'avait déjà) ; supprimer STM32N6 ou la reformuler « cible originale,
   indisponible ».
2. **`LifeLearner`** : description vague (« framework hybride ») — préciser 212 Ko SRAM / STM32H747
   si l'entrée est gardée.
3. **`quantizationINT8`** : « accélère l'inférence » — nuancer : *sur ce projet, gain RAM sans gain
   de latence sur Cortex-M4 FPU (déquantification FP32)* ; ou laisser générique et nuancer en texte.

## 3. Acronymes à AJOUTER (présents dans le PDF rendu, absents du .tex)

```latex
\newacronym{FPU}{FPU}{Floating-Point Unit}
\newacronym{DSP}{DSP}{Digital Signal Processor}
\newacronym{SRAM}{SRAM}{Static Random Access Memory}
\newacronym{GOPS}{GOPS}{Giga Operations Per Second}
\newacronym{FIM}{FIM}{Fisher Information Matrix}
\newacronym{SGD}{SGD}{Stochastic Gradient Descent}
\newacronym{DBSCAN}{DBSCAN}{Density-Based Spatial Clustering of Applications with Noise}
\newacronym{PCA}{PCA}{Principal Component Analysis}
\newacronym{VAE}{VAE}{Variational Autoencoder}
\newacronym{IoT}{IoT}{Internet of Things}
\newacronym{MTSAD}{MTSAD}{Multivariate Time Series Anomaly Detection}
\newacronym{HDC}{HDC}{Hyperdimensional Computing}
\newacronym{PdM}{PdM}{maintenance pr\'edictive (\emph{Predictive Maintenance})}
\newacronym{SoC}{SoC}{System on Chip}
```

## 4. Nouvelles entrées — termes apparus dans la contribution (ch. 4–7)

Acronymes :

```latex
\newacronym{AUROC}{AUROC}{Area Under the Receiver Operating Characteristic curve}
\newacronym{RMSE}{RMSE}{Root Mean Square Error}
\newacronym{PTQ}{PTQ}{Post-Training Quantization}
\newacronym{QAT}{QAT}{Quantization-Aware Training}
\newacronym{DWT}{DWT}{Data Watchpoint and Trace (compteur de cycles ARM Cortex-M)}
\newacronym{UART}{UART}{Universal Asynchronous Receiver-Transmitter}
\newacronym{CRC}{CRC}{Cyclic Redundancy Check}
\newacronym{CCM}{CCM}{Core Coupled Memory}
```

Entrées de glossaire (définitions — formulations à ajuster à la rédaction) :

```latex
\newglossaryentry{Q15}{
  name = {Q15},
  description = {Format virgule fixe sur 16 bits sign\'es (1 bit de signe, 15 bits fractionnaires),
                 utilis\'e ici pour repr\'esenter des tenseurs \`a grande dynamique avec une
                 fid\'elit\'e sup\'erieure \`a l'INT8 affine},
  sort = {q15}}
\newglossaryentry{fakequant}{
  name = {fake-quant},
  description = {Simulation de la quantification pendant l'entra\^inement (quantification puis
                 d\'equantification des poids/activations \`a chaque pas), permettant au mod\`ele
                 d'apprendre sous contrainte de pr\'ecision r\'eduite (base du QAT)},
  sort = {fakequant}}
\newglossaryentry{bss}{
  name = {.bss},
  description = {Section m\'emoire regroupant les variables statiques non initialis\'ees ;
                 sa taille, born\'ee par les symboles \'editeur de liens
                 \texttt{\_sbss}/\texttt{\_ebss}, mesure l'empreinte RAM statique du firmware},
  sort = {bss}}
\newglossaryentry{watermark}{
  name = {stack watermark},
  description = {Technique de mesure du pic d'utilisation de la pile : la zone de pile est
                 pr\'e-remplie d'un motif connu, la plus haute adresse \'ecras\'ee \`a
                 l'ex\'ecution donne le pic r\'eel},
  sort = {watermark}}
\newglossaryentry{parite}{
  name = {parit\'e PC\textleftrightarrow{}board},
  description = {Protocole de validation du portage : les m\^emes poids et les m\^emes
                 \'echantillons sont \'evalu\'es sur PC (Python) et sur la carte (C), et les
                 pr\'edictions sont compar\'ees une \`a une (parit\'e exacte en inf\'erence
                 gel\'ee, approch\'ee en apprentissage en ligne float32 vs float64)},
  sort = {parite}}
\newglossaryentry{P50P99}{
  name = {P50 / P99},
  description = {Percentiles 50 (m\'ediane) et 99 de la distribution des latences mesur\'ees},
  sort = {p50p99}}
\newglossaryentry{perchannel}{
  name = {quantification per-channel},
  description = {Sch\'ema de quantification o\`u chaque canal (ligne de la matrice de poids)
                 poss\`ede sa propre \'echelle, par opposition \`a une \'echelle unique
                 par tenseur (per-tensor) ; r\'eduit l'erreur quand les dynamiques varient
                 fortement entre canaux},
  sort = {per-channel}}
```

## 5. Termes utilisés mais volontairement SANS entrée (définis en texte à la première occurrence)

`domain-incremental` / `class-incremental` (acronymes DomainIL/ClassIL existent déjà),
`gate de nouveauté` (perspectives seulement), `BOPs` (seulement si le ch. 7 le mentionne —
sinon omettre).

## 6. Checklist S4110

- [ ] Chaque acronyme utilisé dans les md finaux a une entrée (grep croisé md ↔ .tex).
- [ ] Aucune entrée obsolète restante (STM32N6, description LifeLearner).
- [ ] Premier usage de chaque acronyme = forme longue (géré par le package glossaries).
