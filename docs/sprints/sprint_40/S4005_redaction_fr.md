# S4005 — Rédaction de l'article (version française)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 40 |
| **Priorité** | 🔴 Critique |
| **Statut** | ✅ Implémenté — `main_fr.tex` + 7 sections FR, tables chiffres canoniques (adossés JSON), 5 figures, distinction mesuré/émulé, board v2 « à mesurer » ; `make fr` OK |
| **Durée estimée** | ~10h |
| **Dépendances** | S4004 (squelette) · S4003 (figures) |
| **Fichiers cibles** | `docs/article/ewc_int8_mcu/main_fr.tex` + `sections/` FR |
| **Références** | S4000 (message scientifique) · exp_S36 · exp_S39 |

## Contexte

Rédiger la version française complète. **Tous les chiffres proviennent du notebook S4003** (donc des JSON).
Le tableau ci-dessous fixe les **valeurs canoniques attendues** (issues de `exp_S36_summary.json` et
`exp_S39_ablation/`) pour relecture — mais le `.tex` doit les référencer via les figures/tables générées,
pas les figer à la main.

## Chiffres canoniques (relecture — ne pas hardcoder, vérifier ≡ JSON)

### Parité & performance FP32 PC↔board (Sprint 36)
| Dataset | Cond. | acc PC | acc board frozen | Δacc | F1 PC/board | parité frozen | parité online |
|---------|:----:|:-----:|:----------------:|:----:|:-----------:|:-------------:|:-------------:|
| Pronostia | 5feat | 0.9887 | 0.9821 | 0.0066 | 0.9164 | 1.000 | 0.975 |
| Pronostia | all | 0.9834 | 0.9831 | 0.0003 | 0.9180 | 1.000 | 0.963 |
| Monitoring | 5feat | — | 0.9846 | — | 0.9194 | 1.000 | 0.989 |

Latences board : frozen 48–65 µs · online 239–340 µs (inf. + MAJ) ≪ 100 ms (**Gap 2**). `.bss` 100–145 Ko < 256 Ko.

### INT8 vs FP32 board — legacy (Sprint 36, effondrement)
| Dataset | Cond. | F1 FP32 | F1 INT8 legacy | accord INT8↔FP32 | RAM ÷ |
|---------|:----:|:-------:|:--------------:|:----------------:|:-----:|
| Pronostia | 5feat | ≈0.916 | **0.138** | 0.736 | ×4 |
| Pronostia | 5feat (online) | — | **0.085** | 0.867 | ×4 |
| Monitoring | 5feat | ≈0.919 | 0.05–0.15 | 0.595 | ×4 |

### Échelle d'ablation (Sprint 39, émulateur — explique la cause)
| Schéma | Pronostia F1 | Monitoring F1 |
|--------|:-----------:|:-------------:|
| FP32 | 0.9616 | 0.9194 |
| legacy_c | 0.0663 | 0.1178 |
| **per_tensor_calib** | **0.9462** (+0.88) | **0.9201** (+0.88) |
| per_channel_int8 | 0.9426 | 0.9187 |
| q15 | 0.9616 | 0.9194 |

### Récupération board v2 (Sprint 40, S4002)
→ **`"à mesurer"`** tant que la carte n'a pas streamé. Ne rien affirmer de chiffré côté board v2 avant
l'exécution réelle.

## Consignes de rédaction

- Ton académique, français, structure S4004.
- **Distinction explicite** dans le texte et les légendes : « mesuré sur carte » (Sprint 36 FP32 + legacy,
  Sprint 40 v2) vs « émulé PC bit-exact » (Sprint 39 ablation/récupération).
- Assumer le **paradoxe latence** INT8 (RAM ÷4 sans accélération FPU) — le présenter comme constat honnête +
  piste SIMD, pas le masquer.
- Insérer les figures depuis `docs/figures/sprint40_article/`.

## Vérification

```bash
cd docs/article/ewc_int8_mcu && make fr    # main_fr.pdf compile sans erreur
```
