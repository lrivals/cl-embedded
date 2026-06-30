# S3404 — Notebook streaming

| Champ | Valeur |
|-------|--------|
| **Sprint** | 34 |
| **Priorité** | 🟡 Important |
| **Statut** | ✅ Implémenté |
| **Durée estimée** | 2h |
| **Dépendances** | S3401 (modèle) · S3403 (mesures board) |
| **Fichiers cibles** | `notebooks/cl_eval/streaming/comparison.ipynb` |
| **Références** | `notebooks/cl_eval/cwru_by_severity/comparison.ipynb` (pattern notebook existant) |

---

## Contexte

Synthèse visuelle de l'étude streaming/buffer : relier le modèle analytique (S3401) aux
mesures réelles de saturation (S3403), et documenter par écrit la question ouverte du
multi-stream concurrent (restée analytique ce sprint, cf. `docs/sprints/sprint_34/
S3400_sprint_34.md`).

---

## Spec

Sections attendues :

1. **Débit max vs débit d'acquisition** par modèle (EWC/HDC/TinyOL/Mahalanobis), courbe
   `debit_max(latence_inf)` superposée aux points mesurés `f_acq` réels.
2. **Courbes latence vs stride** — issues de `experiments/exp_S34_streaming/`.
3. **Occupation buffer vs W** — `.bss` mesuré par config de fenêtre.
4. **Frontière temps-réel** — zone `debit_streaming <= debit_max` (sûr) vs au-delà
   (saturation observée S3403).
5. **Note analytique multi-stream concurrent** (cellule markdown) : estimation du nombre de
   flux simultanés supportables d'après le budget SRAM et la latence cumulée, sans
   prototype firmware (hors périmètre, cf. question ouverte `TODO(arnaud)` du sprint).
6. **Synthèse écrite** liant au critère Gap 2 (`FIXME(gap2)` : latence < 100 ms pour toutes
   les configs testées).

**Règles** : notebook exécutable de bout en bout, déplacé dans `notebooks/` directement
(jamais ailleurs), tout chiffre affiché vient d'une exécution réelle de S3401/S3403.

---

## Vérification

```bash
jupyter nbconvert --to notebook --execute notebooks/cl_eval/streaming/comparison.ipynb \
    --output comparison.ipynb
```
