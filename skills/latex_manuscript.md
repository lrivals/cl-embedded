# Skill : Rédaction LaTeX — Manuscrit

> **Usage** : Générer ou réviser des sections du manuscrit LaTeX à partir des résultats expérimentaux.  
> **Déclencheur** : "rédige la section [X] du manuscrit" / "intègre ces résultats dans le LaTeX" / "génère le tableau comparatif"

---

## Structure du manuscrit (rappel)

```
main.tex
├── sections/sec1_introduction.tex
├── sections/sec2_contexte.tex
├── sections/sec3_methodes_CL.tex
│   ├── sec3.1 — Définitions et taxonomie        [✅ v3 rédigé]
│   └── sec3.2 — Méthodes par régularisation     [🔄 en cours]
├── sections/sec4_embarque_tinyml.tex
├── sections/sec5_etude_comparative.tex
└── sections/sec6_discussion.tex
```

---

## Conventions LaTeX du projet

### Citations
```latex
% Format : (Auteur, Année)
\cite{Kirkpatrick2017EWC}           → (Kirkpatrick et al., 2017)
\cite{Ren2021TinyOL}                → (Ren et al., 2021)
\cite{DeLange2021Survey}            → (De Lange et al., 2021)
```

### Glossaire (macros)
```latex
\acrfull{CL}     → Continual Learning (CL) [première occurrence]
\acrshort{CL}    → CL                       [occurrences suivantes]
\gls{ewc}        → EWC avec définition au survol
```

### Tableaux (format booktabs obligatoire)
```latex
\begin{table}[htbp]
  \centering
  \caption{Comparaison des méthodes CL pour systèmes embarqués}
  \label{tab:comparaison_methodes}
  \begin{tabular}{lcccc}
    \toprule
    Méthode & RAM (Ko) & AF & Gap 1 & Gap 2 & Gap 3 \\
    \midrule
    EWC Online & 9 & 0.05 & ❌ & ✅ & ❌ \\
    HDC & 12 & 0.00 & ❌ & ✅ & ❌ \\
    TinyOL & 7 & -- & ❌ & ✅ & ❌ \\
    \bottomrule
  \end{tabular}
\end{table}
```

---

## Intégration des résultats expérimentaux

### Template de paragraphe de résultats

```latex
% Template à remplir avec les valeurs de experiments/exp_XXX/results/metrics.json

La méthode \acrshort{EWC} appliquée sur le Dataset 2 (équipements industriels)
atteint une accuracy moyenne (\textit{Average Accuracy}) de \textbf{0.87}
après trois domaines séquentiels, contre 0.71 pour le fine-tuning naïf
et 0.92 pour l'entraînement joint (borne supérieure).
L'oubli moyen (\textit{Average Forgetting}, AF) s'établit à 0.05,
confirmant l'efficacité du terme de régularisation élastique.
L'empreinte mémoire mesurée est de \textbf{9.1 Ko} (RAM totale),
soit un facteur 7 en deçà de la cible de 64 Ko du STM32N6,
ce qui répond partiellement au Gap 2 identifié dans notre analyse
comparative (\autoref{sec:triple_gap}).
```

### Tableau comparatif final (template pour Section 5)

```latex
\begin{table*}[htbp]
  \centering
  \caption{Synthèse comparative des méthodes d'apprentissage incrémental
           pour systèmes embarqués à ressources limitées.
           \textbf{AF} : Average Forgetting.
           \textbf{RAM} : pic mémoire mesuré sur PC (proxy embarqué).
           Les cases \textcolor{green}{✓} et \textcolor{red}{✗} indiquent
           si la méthode contribue au gap correspondant.}
  \label{tab:synthese_comparative}
  \begin{tabular}{lccccccc}
    \toprule
    Méthode & Taxonomie & RAM (Ko) & AF & Gap 1 & Gap 2 & Gap 3 & Score \\
    \midrule
    Fine-tuning naïf (baseline) & — & — & élevé & ✗ & ✗ & ✗ & 0/3 \\
    EWC Online \cite{Kirkpatrick2017EWC} & Régularisation & 9 & faible & ✗ & ✓ & ✗ & 1/3 \\
    HDC \cite{Benatti2019HDC} & Architecture & 12 & nul & ✗ & ✓ & ✗ & 1/3 \\
    TinyOL \cite{Ren2021TinyOL} & Architecture & 7 & modéré & ✗ & ✓ & ✗ & 1/3 \\
    \textbf{TinyOL + UINT8} (ce travail) & Hybride & 8 & modéré & ✓* & ✓ & ✓* & 2–3/3 \\
    \bottomrule
    \multicolumn{8}{l}{\small * partiellement — voir Section~\ref{sec:discussion}}
  \end{tabular}
\end{table*}
```

---

## Style de rédaction

### Règles générales
- Rédiger en **français**, termes techniques entre parenthèses en anglais à la 1ère occurrence
- Pas de bullet points dans le corps du texte — utiliser des listes `enumerate` ou de la prose
- Longueur des sections :
  - Sec 3.x (méthodes) : 600–1000 mots
  - Sec 4 (embarqué) : 800–1200 mots
  - Sec 5 (comparatif) : 1000–1500 mots + tableaux
  - Sec 6 (discussion) : 600–900 mots

### Structure type d'une sous-section méthode

```
1. Problème motivant la méthode (1-2 phrases)
2. Principe de fonctionnement (2-4 phrases, formule si pertinente)
3. Faisabilité embarquée (RAM, type d'opération, lattence) 
4. Lien avec le triple gap (1-2 phrases)
5. Limitations identifiées (1-2 phrases)
```

---

## Commande de génération de contenu LaTeX

Quand on demande à Claude de rédiger une section, fournir :

```
Section cible : [sec X.Y]
Contenu source : [fiche de lecture + résultats expérimentaux JSON]
Style : [académique / synthétique / comparatif]
Longueur cible : [X mots]
Termes à faire apparaître : [liste de termes techniques]
Citations à inclure : [liste de clés BibTeX]
```

---

## Ne jamais faire dans le LaTeX

- ❌ Inventer des valeurs numériques (toujours issues de `experiments/`)
- ❌ Citer un article sans avoir sa clé dans `references.bib`
- ❌ Utiliser `\textbf{}` pour surligner des résultats non significatifs
- ❌ Créer des tableaux sans `\label{}` (non référençable)
- ❌ Reformuler des chiffres de la littérature différemment de leur publication
