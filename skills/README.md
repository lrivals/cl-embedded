# Skills — Index et Guide d'utilisation

> Ce dossier contient des guides de prompting pour optimiser l'utilisation de Claude
> dans ce projet. Chaque skill cible un type de tâche spécifique.
>
> **Principe** : mentionner le skill pertinent dans ta demande pour que Claude
> adopte immédiatement le bon format, les bonnes contraintes et les bonnes vérifications.

---

## Index des skills

| Fichier | Quand l'utiliser | Déclencheur type |
|---------|-----------------|-----------------|
| `sprint_generation.md` | Planifier le travail de la semaine | "Génère le sprint N" |
| `model_implementation.md` | Écrire ou réviser du code de modèle | "Implémente le fichier X.py" |
| `cl_evaluation.md` | Calculer ou interpréter des métriques CL | "Interprète ces résultats" |
| `embedded_portability.md` | Vérifier la portabilité MCU d'un modèle | "Ce code est-il compatible STM32N6 ?" |
| `latex_manuscript.md` | Rédiger des sections du manuscrit | "Rédige la section 3.2 en LaTeX" |
| `code_review.md` | Réviser du code existant | "Révise ce fichier Python" |
| `update_pipeline_diagram.md` | Synchroniser le diagramme pipeline avec le dépôt | "Mets à jour le diagramme pipeline" |

---

## Utilisation dans VSCode avec Claude Code

### Méthode 1 — Référence directe dans le prompt
```
[Skill: model_implementation]
Implémente src/models/hdc/hdc_classifier.py selon la spec dans docs/models/hdc_spec.md.
```

### Méthode 2 — Contexte implicite (Claude Code lit CLAUDE.md)
Claude Code lira automatiquement `CLAUDE.md` au démarrage. Les skills sont mentionnés
dans `CLAUDE.md` — Claude les appliquera si la demande correspond.

### Méthode 3 — Demande explicite de lecture
```
Lis skills/cl_evaluation.md puis interprète le fichier experiments/exp_001/results/metrics.json
```

---

## Économie de tokens — Bonnes pratiques

Ces skills évitent de ré-expliquer le contexte à chaque session :

**Au lieu de** :
> "Je travaille sur un projet de continual learning pour MCU STM32N6 avec 64 Ko de RAM,
> j'utilise PyTorch, je veux implémenter EWC avec SGD, sans BatchNorm, avec des annotations
> mémoire sur chaque couche..."

**Dire simplement** :
> "[Skill: model_implementation] Implémente ewc_mlp.py"

Claude Code accède au contexte complet via `CLAUDE.md` + le skill ciblé.

---

## Skills à créer selon l'avancement du projet

| Priorité | Skill futur | Phase |
|:--------:|-------------|-------|
| 🔴 | `data_pipeline.md` — conventions loaders + feature engineering | Sprint 1 |
| 🟡 | `experiment_runner.md` — format scripts CLI + argparse | Sprint 2 |
| 🟡 | `onnx_export.md` — procédure d'export et validation | Sprint 4 |
| 🟢 | `github_release.md` — conventions README public + GitHub Actions | Phase 3 |
| 🟢 | `supervisor_email.md` — rédaction emails encadrants (Arnaud, Dorra) | Continu |
