# S04 — Bibliographie de la soutenance

Toutes les références citées dans le dossier de soutenance (`S00`–`S03`), plus les entrées
qui manquaient au dépôt et dont la soutenance a besoin : CWRU, Paderborn, CMSIS-NN, les
détecteurs de dérive de la slide de secours B10, et les crédits d'images.

Fichier compagnon : [`references_soutenance.bib`](references_soutenance.bib) — même contenu,
compilable si le deck passe par LaTeX/Beamer.

---

## Portée et convention

**Convention de clés retenue = celle du manuscrit**
(`docs/rapport_de_stage/manuscrit_overleaf/references.bib`), désignée comme la convention de
référence par l'audit [`S4103_audit_biblio.md`](../sprints/sprint_41/S4103_audit_biblio.md).
Les entrées reprises de ce fichier y sont **copiées à l'identique** : pour les corriger, éditer
la source et recopier — jamais l'inverse.

`S00_index.md` et `S02_plan_presentation.md` écrivent aujourd'hui les clés de l'**article**
(`docs/article/ewc_int8_mcu/references.bib`). Table de correspondance :

| Clé écrite dans `S00`/`S02` | Clé du manuscrit (à utiliser) |
|---|---|
| `Kirkpatrick2017EWC` | `Kirkpatrick2017` |
| `Ren2021TinyOL` | `Ren2021` |
| `Ravaglia2021QLRCL` | `Ravaglia2021` |
| `Kwon2023LifeLearner` | `Kwon2023` |
| `Benatti2019HDC` | `Benatti2019` |
| `DeLange2021Survey` | `DeLange2021` |
| `Capogrosso2023TinyML` | `Capogrosso2023` |
| `Nectoux2012Pronostia` | `Nectoux2012` |

Gabarit de chaque entrée ci-dessous :
**Clé** — Auteurs, *Titre*, venue, année. · *Cité :* origine dans le dossier.

---

## A — L'état de l'art embarqué (slide 5)

Les quatre travaux du tableau `S02:128-133`. C'est la **colonne de droite** de ce tableau — ce
que chacun laisse ouvert — qui construit le triple gap de la slide 6.

**`Ren2021`** — H. Ren, D. Anicic, T. A. Runkler, *TinyOL: TinyML with Online-Learning on
Microcontrollers*, arXiv:2103.08295, 2021.
· *Cité :* `S02:130` (tableau), `S01:348` (Q-23, réserve d'architecture), `S01:587-594` (Q-41,
« qu'y a-t-il de nouveau chez vous »), `S01:604`.
· **Attention à l'homonymie** : *TinyOL* désigne à la fois ce travail et le modèle M1 du projet,
qui s'en inspire sans en être le portage. La ligne « TinyOL » du tableau 5.1 est notre
auto-encodeur `k → 32 → 16 → k`, pas celui de l'article.

**`Ravaglia2021`** — L. Ravaglia, M. Rusci, D. Nadalini, A. Capotondi, F. Conti, L. Benini,
*A TinyML Platform for On-Device Continual Learning with Quantized Latent Replays*,
IEEE JETCAS 11(4):789–802, 2021. DOI 10.1109/JETCAS.2021.3121554.
· *Cité :* `S02:131` (tableau), `S01:602` (Q-42), `S01:667` (Q-48, quantification du buffer de
rejeu).
· **Le `.bib` de l'article porte une venue erronée** (TCSVT) ; la venue correcte est JETCAS,
comme ci-dessus et comme dans le `.bib` du manuscrit. Ne pas recopier depuis l'article.

**`Benatti2019`** — S. Benatti, F. Montagna, V. Kartsch, A. Rahimi, D. Rossi, L. Benini,
*Online Learning and Classification of EMG-Based Gestures on a Parallel Ultra-Low Power Platform
Using Hyperdimensional Computing*, IEEE TBioCAS, 2019.
· *Cité :* `S02:132` (tableau) — la référence du modèle M3 (HDC).

**`Kwon2023`** — Y. D. Kwon, J. Chauhan, H. Jia, S. I. Venieris, C. Mascolo,
*LifeLearner: Hardware-Aware Meta Continual Learning System for Embedded Computing Platforms*,
ACM SenSys, 2023.
· *Cité :* `S02:133` (tableau), `S01:394-402` (Q-26, « 212 Ko sur Cortex-M7 contre vos 105 Ko »),
`S01:603`.

---

## B — Apprentissage continu : fondements et méthodes citées

**`Kirkpatrick2017`** — J. Kirkpatrick, R. Pascanu, N. Rabinowitz *et al.*,
*Overcoming Catastrophic Forgetting in Neural Networks*, PNAS, 2017.
· **La référence la plus exposée de la soutenance** : trois figures projetées en sont extraites
(§ H), et l'argument de la slide 4 se construit *contre* son panneau A.
· *Cité :* `S02:90` (slide 4, contrepoint publié), `S02:104` (le régime *permuted MNIST* opposé
au class-incremental CWRU), `S02:126` (slide 5), `S02:199` (slide 9, schéma des bassins de
perte), `S01:167` (Q-10, l'approximation diagonale de Fisher « retenue par Kirkpatrick
lui-même »), `S00:94-106`.

**`Zenke2017`** — F. Zenke, B. Poole, S. Ganguli, *Continual Learning Through Synaptic
Intelligence*, ICML, 2017.
· *Cité :* `S01:651-655` (Q-47) — accumulation du gradient le long de la trajectoire.

**`Aljundi2018MAS`** — R. Aljundi, F. Babiloni, M. Elhoseiny, M. Rohrbach, T. Tuytelaars,
*Memory Aware Synapses: Learning What (not) to Forget*, ECCV, 2018. arXiv:1711.09601.
· *Cité :* `S01:651-655` (Q-47) — sensibilité de la sortie.
· Le `.bib` du manuscrit contient (ou contenait) un doublon `Aljundi2018` mal typé, signalé par
l'audit S4103 § 3.1 ; c'est bien `Aljundi2018MAS` qu'il faut citer.

**`Li2018LwF`** — Z. Li, D. Hoiem, *Learning Without Forgetting*, IEEE TPAMI, 2018.
· *Cité :* `S01:651-657` (Q-47) — distillation depuis le modèle précédent, donc **empreinte
doublée**, ce qui est exactement l'argument embarqué qui l'écarte.

**`DeLange2021`** — M. De Lange, R. Aljundi, M. Masana *et al.*, *A Continual Learning Survey:
Defying Forgetting in Classification Tasks*, IEEE TPAMI, 2021.
· Taxonomie régularisation / rejeu / architecture qui structure le chapitre 2 et la slide 9.

**`Rebuffi2017`** — S.-A. Rebuffi, A. Kolesnikov, G. Sperl, C. H. Lampert, *iCaRL: Incremental
Classifier and Representation Learning*, CVPR, 2017.
**`LopezPaz2017`** — D. Lopez-Paz, M. Ranzato, *Gradient Episodic Memory for Continual
Learning*, NeurIPS, 2017.
· Les deux références de rejeu à avoir sous la main si le jury demande pourquoi le rejeu a été
écarté (`S01:667`, Q-48 : le stockage d'exemplaires est la famille la plus difficile à porter).

**Baselines non supervisées** — la slide 9 met Mahalanobis au focus, et Q-34 (`S01:500-510`)
attaque frontalement ce choix :

- **`Mahalanobis1936`** — P. C. Mahalanobis, *On the Generalised Distance in Statistics*,
  Proceedings of the National Institute of Sciences of India, 1936. — le modèle M4 porté sur
  carte : une moyenne et une inverse de covariance, pas de rétropropagation, pas de stockage
  d'exemplaires, 3 à 5 µs par inférence.
- **`Liu2008IsolationForest`** — F. T. Liu, K. M. Ting, Z.-H. Zhou, *Isolation Forest*, IEEE ICDM,
  2008. — **nommé par le jury dans Q-34**, jamais implémenté ici : à avoir sous la main
  uniquement pour ne pas être pris au dépourvu. L'auto-encodeur de la même question est, lui,
  bien présent dans le dispositif — c'est TinyOL.

---

## C — TinyML / embarqué

**`Lin2023`** — J. Lin, L. Zhu, W.-M. Chen, W.-C. Wang, S. Han, *Tiny Machine Learning: Progress
and Futures*, IEEE Circuits and Systems Magazine 23(3):8–34, 2023. DOI 10.1109/MCAS.2023.3302182.
· *Cité :* `S01:191` (Q-05) — **la définition du TinyML opposable au jury** : pas de DRAM, pas de
système d'exploitation, moins de 256 Ko de SRAM. C'est elle qui légitime la NUCLEO-F439ZI.

**`Capogrosso2023`** — L. Capogrosso, F. Cunico, D. S. Cheng, F. Fummi, M. Cristani,
*A Machine Learning-Oriented Survey on Tiny Machine Learning*, IEEE Access, 2023.

**`Pellegrini2021`** — L. Pellegrini, V. Lomonaco, G. Graffieti, D. Maltoni, *Continual Learning
at the Edge: Real-Time Training on Smartphone Devices*, ESANN, 2021.

**`Lai2018CMSISNN`** — L. Lai, N. Suda, V. Chandra, *CMSIS-NN: Efficient Neural Network Kernels
for Arm Cortex-M CPUs*, arXiv:1801.06601, 2018. — **entrée nouvelle**, absente de tous les `.bib`
du dépôt alors que CMSIS-NN est cité deux fois en prose.
· *Cité :* `S02:442` (slide 24, piste non explorée), `S01:673-679` (Q-49) — les instructions SIMD
du Cortex-M4 accéléreraient l'étage MAC (≈ 6 785 cycles), pas l'étage de requantification
(≈ 3 777 cycles), qui est un arrondi flottant par neurone.

---

## D — Quantification (slides 22–24)

**`Jacob2018`** — B. Jacob, S. Kligys, B. Chen *et al.*, *Quantization and Training of Neural
Networks for Efficient Integer-Arithmetic-Only Inference*, CVPR, 2018.
· La référence du QAT et du schéma affine échelle + point zéro — donc de l'argument de la
slide 23 : l'effondrement venait d'une **échelle fixe 1/128 non calibrée**, pas de la
quantification.

**`Krishnamoorthi2018`** — R. Krishnamoorthi, *Quantizing Deep Convolutional Networks for
Efficient Inference: A Whitepaper*, arXiv:1806.08342, 2018.
· La référence à citer pour l'opposition **par tenseur / par canal** et pour PTQ contre QAT.

---

## E — Maintenance prédictive et séries temporelles

**`Hurtado2023`** — J. Hurtado, A. Salvati, A. Cossu, A. Carta, D. Bacciu, *Continual Learning
for Predictive Maintenance: Overview and Challenges*, Intelligent Systems with Applications, 2023.
· La référence qui pose le cadre CL × PdM, donc le « ce qu'il laisse ouvert » de la ligne HDC.

**`BesnardRagot2024`** — Q. Besnard, N. Ragot, *Continual Learning for Time Series Forecasting:
A First Survey*, arXiv, 2024.

**`BerghoutBenbouzid2022`** — T. Berghout, M. Benbouzid, *A Systematic Guide for Predicting
Remaining Useful Life with Machine Learning*, Electronics, 2022.
· Utile sur CMAPSS (Q-33, `S01:488-498`) : le RUL est une régression, et une décision par
échantillon isolé sans fenêtre temporelle y est structurellement désavantagée.

---

## F — Jeux de données (slide 8)

| Jeu | Clé | Référence |
|---|---|---|
| D4 Pronostia | `Nectoux2012` | Nectoux *et al.*, *PRONOSTIA: An Experimental Platform for Bearings Accelerated Degradation Tests*, IEEE PHM, 2012 |
| D5 CMAPSS | `Saxena2008` | Saxena, Goebel, Simon, Eklund, *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*, PHM, 2008 |
| D6 Paderborn | `Lessmeier2016` | Lessmeier, Kimotho, Zimmer, Sextro, *Condition Monitoring of Bearing Damage in Electromechanical Drive Systems…*, PHM Society European Conf., 2016 |
| D3 CWRU | `SmithRandall2015` · `CWRU_DataCenter` | Smith & Randall, *Rolling Element Bearing Diagnostics Using the CWRU Data: A Benchmark Study*, MSSP 64–65:100–131, 2015 · données : Case Western Reserve University Bearing Data Center |
| D1 Pump (Kaggle) | — | **slug manquant** |
| D2 Monitoring (Kaggle) | — | **slug manquant** |

**Trois points de vigilance :**

1. **CWRU et Paderborn n'avaient d'entrée dans aucun `.bib` du dépôt** — elles sont ajoutées ici
   depuis l'audit S4103 § 4 et depuis le docstring de `src/data/paderborn_loader.py:29`. Or CWRU
   porte l'argument central de la slide 14 (`S02:282` : « signaler en passant que ce cas vient de
   CWRU ») et Paderborn a sa propre slide 16.
2. **Pour CWRU, citer l'article de référence, pas seulement le Data Center** : `SmithRandall2015`
   est l'étude de référence sur ces données ; `CWRU_DataCenter` est la source des fichiers bruts.
3. **Les deux jeux Kaggle n'ont ni slug ni URL nulle part dans le dépôt** — cf. le `TODO(arnaud)`
   de `docs/context/datasets.md:9` et `:376`. **Aucune URL n'est fabriquée ici.** Si le jury
   demande la provenance de D1 ou D2, la réponse honnête est « jeu Kaggle, le slug exact n'est pas
   consigné dans le dossier de données ». À citer en note de bas de page avec URL et date d'accès
   une fois renseigné.

---

## G — Détection de dérive · slide de secours B10 · **hors manuscrit**

Cette section couvre la famille de détecteurs des sprints 43–45, que `S00:48-51` classe
explicitement en **réserve** : elle n'est pas dans le manuscrit, donc le jury ne peut pas
l'anticiper. Aucune de ces références n'a d'entrée dans les `.bib` du projet ; elles sont créées
ici.

**Supervisés à état constant** (`S03:220-221` — 16 à 20 octets d'état, coût indépendant du
nombre de variables ; les trois portés sur carte sont Page-Hinkley, DDM et PSI) :

- **`Gama2004DDM`** — J. Gama, P. Medas, G. Castillo, P. Rodrigues, *Learning with Drift
  Detection*, SBIA, LNCS 3171:286–295, 2004. — **DDM**.
- **`BaenaGarcia2006EDDM`** — M. Baena-García *et al.*, *Early Drift Detection Method*,
  4ᵉ atelier ECML PKDD sur la découverte de connaissances dans les flux, 2006. — **EDDM**.
- **`Page1954`** — E. S. Page, *Continuous Inspection Schemes*, Biometrika 41(1/2):100–115, 1954.
  — le test **CUSUM/Page-Hinkley**. C'est le détecteur de la cellule de validation carte du
  sprint 45 (parité de verdict carte ↔ PC = 1,000).

**Non supervisés** (`S03:222-223` — coût qui croît avec la dimensionnalité, jusqu'à 245 Ko d'état
pour ADWIN sur un jeu à 128 variables, donc non portables) :

- **`Bifet2007ADWIN`** — A. Bifet, R. Gavaldà, *Learning from Time-Changing Data with Adaptive
  Windowing*, SIAM SDM, 2007. — **ADWIN**.
- **`Raab2020KSWIN`** — C. Raab, M. Heusinger, F.-M. Schleif, *Reactive Soft Prototype Computing
  for Concept Drift Streams*, Neurocomputing 416, 2020. — **KSWIN**.
- **`Massey1951KS`** — F. J. Massey, *The Kolmogorov-Smirnov Test for Goodness of Fit*, JASA
  46(253):68–78, 1951. — le test **KS**.
- **`Gretton2012MMD`** — A. Gretton, K. M. Borgwardt, M. J. Rasch, B. Schölkopf, A. Smola,
  *A Kernel Two-Sample Test*, JMLR 13:723–773, 2012. — **MMD**.
- **PSI** (*population stability index*) : **pas de référence fondatrice canonique.** C'est un
  indicateur issu de la pratique du *credit scoring*, pas d'un article. Le présenter comme tel
  plutôt que de lui attribuer une paternité incertaine. La limite mesurée de `S03:228-231` — PSI
  tire son signal du détecteur de Mahalanobis, dont l'inverse de covariance est en O(k²), d'où le
  débordement SRAM à k = 128 — se défend sans référence externe.

---

## H — Crédits d'images

**Règle générale** : toutes les figures `sNN_*.png` / `bNN_*.png` de `docs/figures/soutenance/`
sont **produites par le dépôt** depuis les JSON d'`experiments/`
(`python scripts/generate_figures.py --catalog soutenance --style slide`, `S00:81`) — travail
propre, aucun crédit tiers à porter. Les quatre images ci-dessous sont les **seules** exceptions :
elles sont déposées à la main et ne doivent jamais être supprimées lors d'une régénération
(`S00:94-97`).

| Fichier | Slide | Source | Statut |
|---|---|---|---|
| `KirkArticle_Frontpage.png` | 5 | `Kirkpatrick2017` — page de titre | reproduite avec citation |
| `KirkArticle_Plot.png` | 4 | `Kirkpatrick2017` — panneau A | reproduite avec citation |
| `KirkArticle_SchemaLoss.png` | 9 | `Kirkpatrick2017` — bassins de perte | reproduite avec citation |
| `MaintenancePredictive_Intro.png` | 2 | `BlueDocker_PdM` — [bluedocker.com — *La maintenance prédictive, un enjeu pour l'industrie*](https://bluedocker.com/maintenance-predictive-enjeu-industrie/), consulté le 22 août 2026 | illustration tierce |

**Les trois captures de Kirkpatrick sont citées, pas reproduites comme travail propre** — il faut
les **annoncer à l'oral** (`S00:105`). Une formulation suffit : *« cette figure vient de l'article
de Kirkpatrick, elle est ici comme contrepoint publié à ma propre mesure. »* C'est d'autant plus
nécessaire slide 4, où l'argument consiste précisément à opposer notre courbe à la leur
(`S02:99-108`).

Pour la slide 2, porter le nom de domaine en petit sous l'image ; c'est une illustration
d'accroche, pas une figure de résultat.

---

## I — Champs incertains

Conformément à la règle du projet — *aucun chiffre inventé*, ici étendue à la bibliographie —
les champs qui n'ont pas pu être vérifiés portent un commentaire `% à vérifier` **dans l'entrée
`.bib` elle-même** :

| Entrée | Champ à vérifier |
|---|---|
| `CWRU_DataCenter` | année de mise en ligne, URL courante du Data Center |
| `BaenaGarcia2006EDDM` | pages |
| `Raab2020KSWIN` | pages |

Par ailleurs, plusieurs entrées **reprises du manuscrit** sont volontairement incomplètes
(volume, numéro, pages, DOI absents) : c'est l'état de la source, signalé par l'en-tête de
`references.bib` lui-même. Elles ne sont pas complétées ici pour ne pas faire diverger les deux
fichiers ; toute complétion doit se faire dans le `.bib` du manuscrit, puis être recopiée.

---

## Utilisation

- **Deck LaTeX/Beamer** : `\bibliography{references_soutenance}`, puis `\cite{Kirkpatrick2017}`
  etc. — les clés sont celles du manuscrit, donc les mêmes que dans le mémoire.
- **Deck non LaTeX** : une slide finale « Références » reprenant les sections A à E suffit ; les
  crédits d'images du § H vont **sur les slides concernées**, pas en fin de deck.
- **Révision** : les lignes *Cité :* permettent de remonter de chaque référence à la question de
  `S01` ou à la slide de `S02` qui la mobilise.
