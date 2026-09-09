# Analyse coût/bénéfice de la quantification INT8 (S5005)

> **Réponse au CR du 16 juillet 2026 (§5)** : *le gain de RAM (÷4) vaut-il la légère chute
> d'accuracy et le surcoût éventuel de latence/énergie ?* Et : *qu'est-ce qui est reproduit de la
> littérature, qu'est-ce qui est notre contribution propre ?*
>
> Document de synthèse décisionnelle. **Chaque chiffre porte sa source** (JSON d'expérience) et
> provient d'une exécution — aucun résultat écrit à la main. Voir aussi
> [`int8_latency_breakdown.md`](int8_latency_breakdown.md) (détail latence),
> [`quantization_strategies.md`](quantization_strategies.md) (les 6 stratégies),
> [`ram_report.md`](ram_report.md) (RAM totale S49).

## 1. Tableau de décision (EWC INT8 v2, board réelle NUCLEO-F439ZI)

| Dimension | FP32 (réf) | INT8 v2 | Verdict | Source |
|-----------|-----------|---------|---------|--------|
| **RAM (poids du modèle)** | 2 688–2 816 B | 672–704 B (**÷4**) | ✅ **gain majeur** | `exp_S40_board_v2/results_per_channel_*_frozen.json` (`ram_ratio_fp32_over_quant=4.0`) |
| **RAM système totale** | ≈ réf | ≈ réf (ratio ≈ 1.0) | ⚠️ le gain porte sur les **poids**, pas le budget total (dominé par des buffers fixes) | `exp_S49_ram/summary.json` (`ratio_int8_vs_fp32≈1.0002`) |
| **Accuracy / F1** | F1 ≈ 0.92 | F1 = 0.921 (monitoring) / 0.907 (pronostia) | ✅ **préservée** (kernel v2 calibré) | `exp_S46_board/{monitoring,pronostia}_both.json` |
| **Latence / inférence** | 48–50 µs | 74 µs (**+24 à +26 µs**) | ❌ **surcoût** (paradoxe FPU) | `exp_S50_int8_latency/{dataset}.json` |
| **Énergie / inférence** | *à mesurer* | *à mesurer* | ⏳ **à statuer** (banc LPM01A non posé) | `exp_S50_energy/summary.json` |

**Lecture.** L'INT8 v2 **préserve la métrique** (F1 ≈ FP32, grâce à la calibration par-canal +
`act_max` du Sprint 39/40) et **divise par 4 la RAM des poids** ; mais il **coûte ~50 % de latence en
plus** sur cette carte à FPU (la requantification `FP32→INT8` ajoute ~3 800 cycles/inférence, cf.
[`int8_latency_breakdown.md`](int8_latency_breakdown.md)), et son **impact énergétique reste à
mesurer**. Toutes les latences restent **≪ 100 ms → Gap 2 ✅**.

> **Nuance d'honnêteté clé.** Le « ÷4 » célèbre de la littérature s'applique aux **poids**. Sur ce
> système, la **RAM totale** (`.data + .bss + pic de pile`, S49) est dominée par des tampons de taille
> fixe → son ratio int8/fp32 est ≈ 1.0. L'INT8 est donc pertinent quand **le stockage des poids** est
> le goulot (grands modèles), moins quand c'est le budget système global.

## 2. Reproduit (littérature) vs contribution propre (CR §5)

| Reproduit de la littérature | Contribution propre de ce travail |
|------------------------------|------------------------------------|
| **Ravaglia 2021** (`Ravaglia2021QLRCL`) : INT8 ÷4 RAM, perte < 0.26 % | Application à **EWC** (méthode de **régularisation**, non couverte) sur **PRONOSTIA** (roulements FEMTO) — F1 préservé mesuré board |
| **Capogrosso 2024** (`Capogrosso2023TinyML`) : taxonomie PTQ / QAT | Compromis **mesuré sur carte réelle** NUCLEO-F439ZI (RAM, F1, latence DWT) — pas seulement émulé |
| **Zhu / Lin** : QAS, distorsion du gradient INT8 | **Paradoxe latence FPU documenté et chiffré** : INT8 ≠ gain latence **sans NPU** (breakdown déquant/MAC/requant, S5004) |
| **Benatti 2019** (`Benatti2019HDC`), **Giménez 2022** : on-device, PTQ MCU | Score système intégrant l'**énergie réelle** (chaîne LPM01A prête, → mesure Sprint 50/51) |

**Ce que la littérature ne dit pas et que l'on montre** : sur un Cortex-M4 **à FPU sans accélérateur
entier**, la quantification est un **coût net en latence** (pas seulement « neutre »). Le bénéfice est
strictement la **RAM des poids** ; le gain latence attendrait une carte **INT8 natif / NPU** (la
STM32N6 visée initialement, indisponible) ou des noyaux **SIMD CMSIS-NN**.

## 3. Recommandation

- **Utiliser l'INT8 v2 calibré** quand **le stockage des poids** est contraignant (modèle volumineux,
  Flash/RAM serrée) : il préserve la F1 (Δ ≈ 0) et divise les poids par 4.
- **Ne pas attendre de gain de latence** de l'INT8 sur cette carte FPU : au contraire, ~+50 %
  (requant/déquant). Si la **latence** est le budget critique et qu'elle est déjà ≪ 100 ms (c'est le
  cas ici), **rester en FP32** est légitime.
- **Énergie** : décision différée — la chaîne de mesure existe (`scripts/run_s50_energy.py`), les
  champs restent `"à mesurer"` tant que le banc **X-NUCLEO-LPM01A** n'est pas posé (aucun µJ inventé).
  Question ouverte : l'INT8 réduit-il malgré tout les µJ (moins d'octets déplacés) malgré le surcoût
  de cycles ? À trancher sur mesure réelle (Sprint 50/51).

En un mot : sur NUCLEO-F439ZI, **INT8 = gain RAM (poids), pas latence** ; pertinent quand la RAM est
le goulot, à éviter si l'on optimise la latence — et l'axe énergie reste à instrumenter.
