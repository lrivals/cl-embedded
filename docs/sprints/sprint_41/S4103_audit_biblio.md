# S4103 — Audit bibliographie (`Manuscrit_Final_Rivals/references.bib`)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 41 |
| **Statut** | ✅ Audit réalisé (3 juillet 2026) — entrées à ajouter livrées en BibTeX ci-dessous |
| **Règle** | Le .bib de l'Overleaf n'est modifié que sur instruction explicite ; ce fichier est la source |

## 1. État des lieux

`references.bib` : 27 entrées, organisées par familles (surveys, régularisation, rejeu,
architecture, TinyML, séries temporelles/PdM, datasets). Bonne couverture du cœur CL/TinyML.

⚠️ Les clés diffèrent de celles listées dans CLAUDE.md (`Ren2021TinyOL` → `Ren2021`,
`Kirkpatrick2017EWC` → `Kirkpatrick2017`, `DeLange2021Survey` → `DeLange2021`,
`Capogrosso2023TinyML` → `Capogrosso2023`, `Ravaglia2021QLRCL` → `Ravaglia2021`,
`Kwon2023LifeLearner` → `Kwon2023`, `Benatti2019HDC` → `Benatti2019`). **Convention du manuscrit
final = les clés du .bib Overleaf** (colonnes de droite).

## 2. Problèmes détectés dans l'existant

1. **Doublon** : `Aljundi2018MAS` (l. 83, @inproceedings, correct) et `Aljundi2018` (l. 249,
   @article avec booktitle en champ journal — incorrect). → **Supprimer `Aljundi2018`**, garder
   `Aljundi2018MAS`.
2. **Wu2025 vs intermédiaire** : le manuscrit intermédiaire cite « Wu et al. (2024) » ; le .bib a
   `Wu2025` (TII). Vérifier l'année réelle de publication (TII vol. 2024 ou 2025) et harmoniser.
3. **Ravaglia2021** : journal indiqué « IEEE TCSVT » ; la publication réelle est *IEEE JETCAS*
   (Journal on Emerging and Selected Topics in Circuits and Systems) — à corriger.
4. **Lin2024** : l'intermédiaire cite Lin et al. (2023), IEEE Circuits and Systems Magazine ; le
   .bib pointe l'arXiv 2024. Choisir la version journal (2023) de préférence.
5. Champs `TODO` (DOI, volumes, pages) signalés en tête de fichier — passe Zotero à prévoir en S4110.

## 3. Entrées MANQUANTES (citées dans l'intermédiaire rendu, absentes du .bib Overleaf)

Nécessaires si le ch. 2 conserve la section MTSAD (recommandé en condensé) :

```bibtex
@article{Belay2023,
  author  = {Mohammed Ayalew Belay and Sindre Stenen Blakseth and Adil Rasheed and Pierluigi Salvo Rossi},
  title   = {Unsupervised Anomaly Detection for {IoT}-Based Multivariate Time Series: Existing Solutions, Performance Analysis and Future Directions},
  journal = {Sensors},
  volume  = {23},
  number  = {5},
  pages   = {2844},
  year    = {2023},
}

@article{Park2018,
  author  = {Daehyung Park and Yuuna Hoshi and Charles C. Kemp},
  title   = {A Multimodal Anomaly Detector for Robot-Assisted Feeding Using an {LSTM}-Based Variational Autoencoder},
  journal = {IEEE Robotics and Automation Letters},
  year    = {2018},
}

@inproceedings{Su2019,
  author    = {Ya Su and Youjian Zhao and Chenhao Niu and Rong Liu and Wei Sun and Dan Pei},
  title     = {Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network},
  booktitle = {ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD)},
  year      = {2019},
}

@inproceedings{Zong2018,
  author    = {Bo Zong and Qi Song and Martin Renqiang Min and Wei Cheng and
               Cristian Lumezanu and Daeki Cho and Haifeng Chen},
  title     = {Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2018},
}
```

Optionnelles (si le ch. 2 garde CURL/CaSSLe — recommandation : 1 phrase, refs conservées) :

```bibtex
@misc{Rao2019,
  author       = {Dushyant Rao and Francesco Visin and Andrei A. Rusu and Yee Whye Teh and
                  Razvan Pascanu and Raia Hadsell},
  title        = {Continual Unsupervised Representation Learning},
  howpublished = {arXiv:1910.14481},
  year         = {2019},
}

@inproceedings{Fini2022,
  author    = {Enrico Fini and Victor G. Turrisi da Costa and Xavier Alameda-Pineda and
               Elisa Ricci and Karteek Alahari and Julien Mairal},
  title     = {Self-Supervised Models are Continual Learners},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2022},
}
```

## 4. Entrées MANQUANTES — nouveaux besoins du manuscrit final

**Datasets (ch. 4, tableau des 6)** — Saxena2008 (CMAPSS) et Nectoux2012 (Pronostia) existent déjà ✅ :

```bibtex
@article{Lessmeier2016,
  author  = {Christian Lessmeier and James Kuria Kimotho and Detmar Zimmer and Walter Sextro},
  title   = {Condition Monitoring of Bearing Damage in Electromechanical Drive Systems by Using
             Motor Current Signals of Electric Motors: A Benchmark Data Set for Data-Driven Classification},
  journal = {Proceedings of the European Conference of the PHM Society},
  year    = {2016},
}

@article{SmithRandall2015,
  author  = {Wade A. Smith and Robert B. Randall},
  title   = {Rolling Element Bearing Diagnostics Using the {Case Western Reserve University} Data:
             A Benchmark Study},
  journal = {Mechanical Systems and Signal Processing},
  volume  = {64--65},
  pages   = {100--131},
  year    = {2015},
}
```

Datasets Kaggle (D1 Pump, D2 Monitoring) : pas d'entrée BibTeX académique — citer en **note de bas
de page avec URL et date d'accès** (pratique standard).

**Quantification (ch. 2 §quantification + ch. 7)** :

```bibtex
@inproceedings{Jacob2018,
  author    = {Benoit Jacob and Skirmantas Kligys and Bo Chen and Menglong Zhu and
               Matthew Tang and Andrew Howard and Hartwig Adam and Dmitry Kalenichenko},
  title     = {Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2018},
}

@misc{Krishnamoorthi2018,
  author       = {Raghuraman Krishnamoorthi},
  title        = {Quantizing Deep Convolutional Networks for Efficient Inference: A Whitepaper},
  howpublished = {arXiv:1806.08342},
  year         = {2018},
}
```

**Baseline Mahalanobis (ch. 2/4)** :

```bibtex
@article{Mahalanobis1936,
  author  = {Prasanta Chandra Mahalanobis},
  title   = {On the Generalized Distance in Statistics},
  journal = {Proceedings of the National Institute of Sciences of India},
  volume  = {2},
  number  = {1},
  pages   = {49--55},
  year    = {1936},
}
```

## 5. Checklist S4110 (fin de sprint)

- [ ] Compléter les champs `TODO` (DOI/volumes/pages) via Zotero.
- [ ] Supprimer le doublon `Aljundi2018` ; corriger `Ravaglia2021` (JETCAS) ; trancher `Wu2025`/2024 ; `Lin2024`→2023.
- [ ] Vérifier que chaque `\cite` des md pointe une clé existante (grep croisé md ↔ .bib).
- [ ] Vérifier les métadonnées des entrées ci-dessus avant intégration (années/venues à re-contrôler en ligne).
