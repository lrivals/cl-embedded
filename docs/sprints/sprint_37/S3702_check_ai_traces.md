# S3702 — `scripts/check_ai_traces.py` (scanner de traces)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 37 |
| **Priorité** | 🔴 Critique — garantit qu'aucune trace ne franchit l'export. |
| **Statut** | ✅ Implémenté |
| **Durée estimée** | 2h |
| **Fichiers cibles** | `scripts/check_ai_traces.py` |
| **Dépendances** | `configs/gitlab_release.yaml` (S3701) ✅ · PyYAML ✅ |

## Contexte

Scanner réutilisable, utilisé à deux endroits : (1) **gate dur** appelé par
`prepare_gitlab_release.py` sur l'export ; (2) brique du **garde-fou** d'ajouts futurs.

## Spec

- `scan_tree(root, config, source_mode)` → liste de `Finding(path, line_no, pattern, excerpt)`.
- Ignore les répertoires de cache/données (`.git`, `__pycache__`, `data`, …) et les extensions
  binaires (`.png`, `.pkl`, `.elf`, …).
- Applique `forbidden_patterns` moins `allowlist` ligne par ligne sur les fichiers texte.
- `source_mode=True` : tolère les `exclude_paths` (zones internes connues côté dépôt de travail) ;
  `source_mode=False` (export) : un `exclude_path` encore présent est signalé comme anomalie.
- CLI : `python scripts/check_ai_traces.py [TREE] [--config] [--source] [--quiet]`.
- Exit 0 si propre, 1 sinon, avec rapport `fichier:ligne [pattern] extrait`.

## Validation

- `check_ai_traces.py <export propre>` → exit 0.
- Trace semée (`# made with Claude`) → exit 1 + ligne fautive listée.
