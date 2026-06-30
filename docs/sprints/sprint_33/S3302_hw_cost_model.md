# S3302 — `src/evaluation/hw_cost_model.py` + `configs/hw_profile_f439zi.yaml`

| Champ | Valeur |
|-------|--------|
| **Sprint** | 33 |
| **Priorité** | 🔴 Critique — bloquant pour S3308 (notebook), entrée du modèle d'autonomie S3307 |
| **Statut** | ✅ Implémenté (23 juin 2026) |
| **Durée estimée** | 3h |
| **Dépendances** | S3301 (FLOPs/MACs disponibles) |
| **Fichiers cibles** | `src/evaluation/hw_cost_model.py`, `configs/hw_profile_f439zi.yaml` |
| **Références** | `src/evaluation/compute_cost.py` (S3301), `firmware/stm32f4_blink/inc/profiling.h:17` (`SYSCLK_HZ = 180000000U`) |

---

## Contexte

Le CR du 19 mai 2026 demande « une formule pour estimer le nombre de calculs en fonction du
matériel ». Aucun fichier `hw_profile*.yaml` n'existe dans `configs/` (confirmé) : ce sprint
crée le module de coût temps-HW/FLOPS-W et sa config dédiée. C'est un **proxy** analytique
(pas une mesure), à ne pas confondre avec les mesures réelles de S3305/S3306.

---

## Spec

```python
# src/evaluation/hw_cost_model.py

def estimate_inference_time(macs: int, flops_peak: float, efficacite: float) -> float:
    """T_HW (s) ~= (2 x macs) / (flops_peak x efficacite).
    efficacite in [0.1, 0.6] (proxy documenté, cf. CR 19 mai — incertitude assumée).
    """
    flops = 2 * macs
    return flops / (flops_peak * efficacite)

def flops_per_watt(flops_peak: float, puissance_watts: float) -> float:
    return flops_peak / puissance_watts

def throughput(temps_inference_s: float) -> float:
    """Inférences/seconde."""
    return 1.0 / temps_inference_s
```

```yaml
# configs/hw_profile_f439zi.yaml — NUCLEO-F439ZI, Cortex-M4 @ 180 MHz, pas de NPU
hardware:
  sysclk_hz: 180000000          # cf. profiling.h SYSCLK_HZ — ne pas dupliquer en dur
  flops_peak_fp32: <à_renseigner>   # FLOPs/s théorique FPU simple précision
  flops_peak_int8: <à_renseigner>   # FLOPs/s effectif (pas d'accélérateur INT8 dédié sur M4)
  efficacite:
    fp32: 0.3   # eff e [0.1, 0.6], valeur par défaut TODO(arnaud)
    int8: 0.3
  puissance_watts:
    actif_mA: <à_mesurer>    # croisé avec S3305/S3306 (LPM01A)
    veille_uA: <à_mesurer>
    tension_v: 3.3
```

**Règles** :
- Toute constante matérielle (FLOPS_peak, efficacité, tension/courant nominaux) vient de
  `configs/hw_profile_f439zi.yaml`, **jamais en dur dans le code** (règle CLAUDE.md). Le
  module `hw_cost_model.py` ne contient que des formules paramétrées.
- Documenter explicitement que `T_HW` est un **proxy** (incertitude liée à `efficacite`,
  cf. CR) — ne pas le présenter comme une mesure équivalente à la latence DWT réelle
  (`profiling.c`).
- Réutiliser `SYSCLK_HZ` (180 MHz, déjà défini dans `profiling.h:17`) comme référence
  documentaire dans le YAML, sans le redéfinir indépendamment côté firmware.

---

## Vérification

```bash
python -c "from src.evaluation.hw_cost_model import estimate_inference_time, flops_per_watt, throughput; \
print(estimate_inference_time(1000, 2e8, 0.3))"

pytest tests/test_hw_cost_model.py -v   # S3309 : bornes eff, FLOPS/W > 0, throughput cohérent
```
