# Coût de latence INT8 — breakdown déquant / MAC / requant (S5004)

> **Réponse au point ouvert du CR du 16 juillet 2026** : « le processeur est FP32 (FPU) ; passer
> en INT8 ajoute des étapes de quantification/déquantification — détailler ces étapes et quantifier
> leur coût en cycles ». Ce document **mesure** ces étapes au DWT sur la NUCLEO-F439ZI réelle.
>
> **Chaque chiffre porte sa source** (`experiments/exp_S50_int8_latency/{dataset}.json`, agrégat
> `ewc.json`) et provient d'un **run board réel** (`scripts/run_s50_int8_latency.py`). Aucun cycle
> n'est écrit sans mesure. Régénérer : `python scripts/run_s50_int8_latency.py`.

## 1. Où la quantification s'insère dans le pipeline d'inférence

Le kernel EWC INT8 v2 ([`firmware/stm32f4_blink/src/ewc_head_int8_v2.c`](../../firmware/stm32f4_blink/src/ewc_head_int8_v2.c),
`ewc_int8_v2_forward`) enchaîne, par couche, trois familles d'opérations :

```
  entrée FP32
     │
     ▼  ┌─ REQUANT ──┐  quantif entrée FP32 → INT8  (quant_val = lroundf(x/scale))
        └────────────┘
     │
     ▼  ┌─ MAC ──────┐  produit scalaire ENTIER  acc(int32) = Σ w_q[j][i]·a_q[i]
        └────────────┘
     │
     ▼  ┌─ DÉQUANT ──┐  int → FP32 sur FPU  val = (float)acc·scale_w[j]·scale_act + b[j]  (+ ReLU)
        └────────────┘
     │
     ▼  ┌─ REQUANT ──┐  requantif activation FP32 → INT8 pour la couche suivante
        └────────────┘
   … (couches 1, 2 ; couche 3 = MAC + déquant, pas de requant en sortie : logits FP32)
```

- **REQUANT** (`FP32 → INT8`) = les appels `quant_val` (entrée + activations) — contiennent un
  `lroundf`, opération coûteuse.
- **MAC** = l'accumulation **entière** `int32` — ne bénéficie d'aucune accélération vs FP32 sur le
  Cortex-M4 (la FPU fait un MAC flottant tout aussi vite).
- **DÉQUANT** (`int → FP32`) = le rescale flottant `(float)acc·scale·scale + b` + ReLU.

Le **FP32 de référence** ([`ewc_head.c`](../../firmware/stm32f4_blink/src/ewc_head.c), `ewc_head_forward`)
n'a **ni requant ni déquant** : il fait directement le MAC flottant sur la FPU.

### Instrumentation

Trois accumulateurs de **cycles DWT bruts** (`seg_dequant_cycles`, `seg_mac_cycles`,
`seg_requant_cycles`) sont ajoutés à `ProfilingState` et remplis par `ewc_int8_v2_forward`, le tout
**entièrement sous `-DINT8_SEGMENT_PROFILE`** (build par défaut strictement inchangé, `.bss` défaut
invariant à 105 036 B, `make test` inchangé). On reporte des **cycles bruts** (et non des µs) : la
conversion µs de `profiling.c` divise par 180 et **tronque** → un segment de quelques dizaines de
cycles arrondirait à 0 µs. Les 3 compteurs sont remontés dans la réponse V3 (23 B, **wire format
inchangé**) via les slots `[acc][auroc][forgetting]` réinterprétés côté hôte (précédent S3805/S4502).

## 2. Coûts mesurés (board réelle NUCLEO-F439ZI, DWT, 180 MHz)

Cycles DWT p50 (800 inférences frozen, 0 erreur CRC). Source : `exp_S50_int8_latency/{dataset}.json`.

| Segment | monitoring (k=4) | pronostia (k=5) | Nature |
|---------|:----------------:|:---------------:|--------|
| REQUANT (FP32→INT8, `lroundf`) | 3 760 cyc (≈ 20.9 µs) | 3 795 cyc (≈ 21.1 µs) | **surcoût INT8** |
| MAC (entier) | 6 841 cyc (≈ 38.0 µs) | 6 729 cyc (≈ 37.4 µs) | ≈ identique à FP32 |
| DÉQUANT (int→FP32, FPU) | 200 cyc (≈ 1.1 µs) | 200 cyc (≈ 1.1 µs) | **surcoût INT8** |
| **Σ segments** | 10 801 cyc | 10 724 cyc | — |

Latences totales par inférence (µs, p50) :

| Total | monitoring | pronostia |
|-------|:----------:|:---------:|
| INT8 v2 | 74 µs | 74 µs |
| FP32 (même flash, mêmes échantillons, flag 0x10) | 48 µs | 50 µs |
| **INT8 − FP32** | **+26 µs** | **+24 µs** |

Les deux restent **≪ 100 ms → Gap 2 ✅** ([`triple_gap.md`](../triple_gap.md)).

## 3. Interprétation — le paradoxe latence FPU

Le résultat **confirme et chiffre le paradoxe du Sprint 29** : sur ce Cortex-M4 **doté d'une FPU**,
l'INT8 **n'accélère pas** l'inférence — il la **ralentit** (+24 à +26 µs, soit ~+50 %). La raison,
maintenant mesurée :

1. le **MAC entier** (le seul segment que l'INT8 pourrait accélérer) **ne gagne rien** : la FPU
   exécute un MAC flottant à la même vitesse (~6 800 cyc dans les deux cas) ;
2. l'INT8 **ajoute** la **requantification** (`FP32 → INT8`, avec `lroundf`), qui coûte à elle seule
   ~3 800 cyc (≈ 21 µs) — c'est le poste dominant du surcoût — plus la **déquantification** (~200 cyc).

Autrement dit, sur une carte FPU sans NPU ni SIMD entier, **la quantification est un coût net en
latence**. Le gain de l'INT8 est **ailleurs** : la **RAM ÷ 4** (poids `int8` vs `float32`, Gap 3,
Sprints 28/49). Un vrai gain de latence exigerait :

- une carte à **INT8 natif / accélérateur entier** (perspective CR — la STM32N6 à NPU visée
  initialement, indisponible), ou
- des noyaux **SIMD CMSIS-NN** (`__SMLAD` : 4 MAC int8/cycle) exploitant le DSP du M4 — piste future.

## 4. Portée — pourquoi EWC seulement

Le breakdown déquant/MAC/requant **par couche** n'a de sens que pour un kernel de type MLP quantifié.
Les autres modèles portés en INT8 ont une structure différente :

- **Mahalanobis INT8/Q15** = **N/A par construction** : le kernel déquantifie `mu`/`sigma_inv` puis
  calcule une **distance** (`mahalanobis_int8.c`/`mahalanobis_q15.c`) — pas de MAC ni de requant par
  couche à isoler ;
- **HDC** : projection binaire/entière, pas de déquant/requant flottant par couche ;
- **TinyOL** : archi board distincte, non exportable (cf. Sprints 29/32).

La mesure porte donc sur **EWC INT8 v2** (`per_channel_int8`), le kernel canonique du projet, sur les
deux datasets de référence **monitoring** (D2) et **pronostia** (D4, le dataset « contribution propre »).

## 5. Reproductibilité

```bash
python scripts/run_s50_int8_latency.py                 # monitoring + pronostia
python -c "import json;d=json.load(open('experiments/exp_S50_int8_latency/ewc.json'));\
print({s: d['segments'][s]['cycles_p50_mean_over_datasets'] for s in ('dequant','mac','requant')})"
```

Firmware : `make EXTRA_CFLAGS="-DEWC_INT8_V2 -DINT8_SEGMENT_PROFILE" EWC_IN=<k> all flash`.
Le build par défaut (sans le flag) est **inchangé** — l'instrumentation est intégralement gardée.
