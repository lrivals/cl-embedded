# S5308 — Isolation par composant + autonomie à duty-cycle réaliste

| Champ | Valeur |
|-------|--------|
| **Sprint** | 53 |
| **Priorité** | 🟠 Important |
| **Statut** | 📝 Spec |
| **Durée estimée** | 2 h (30 min banc + 1 h 30 analyse/écriture) |
| **Dépendances** | S5302 (repos WFI) · S5304 (pente) |
| **Fichiers cibles** | `experiments/exp_S53_components/` · `configs/hw_profile_f439zi.yaml` · `configs/energy_campaign_s50.yaml` |
| **Références** | CR 16 juillet (« isoler les mesures par composant ») · `lpm01a_setup.md:151-169` |

## Contexte

Deux dettes distinctes, réglées par la même session de banc.

**1. `by_component` est entièrement `"à mesurer"`.** Le CR du 16 juillet demandait
explicitement d'isoler MCU / périphériques / capteurs. Les 8 cellules du Sprint 50 portent
`mcu: "à mesurer"`, `periph: "à mesurer"`, `sensor: "na"` (capteurs simulés par UART — ce
`"na"` est correct et le reste). La sonde mesure le **rail VDD_MCU entier** : l'isolation
ne peut donc être qu'**différentielle**, par A/B d'extinction.

Un seul A/B a été tenté : `-DETH_PHY_POWERDOWN` → 63,24 mA, *plus haut* que les 59,07 mA de
référence. Le doc note honnêtement que **ce n'était pas un A/B strict**
(`lpm01a_setup.md:151-169`) : les deux mesures ne venaient pas de la même session, donc
l'anomalie d'ordre de S5301 pouvait entièrement l'expliquer. **À refaire proprement.**

**2. L'autonomie publiée n'est pas une autonomie de déploiement.**
`exp_S50_energy/autonomy.json` porte 4,30–4,74 h @220 mAh et 195–215 h @10 000 mAh, avec un
bloc `regime_mesure` qui avertit — correctement — qu'il s'agit du régime de banc (flux
continu à 100 Hz), pas d'un duty-cycle réaliste. `configs/energy_campaign_s50.yaml:21`
porte `inference_period_s: 1.0` avec un `TODO(fred)` jamais tranché.

## Spec

### 1. A/B d'isolation, **une seule session, ordre contre-balancé**

Appliquer la leçon de S5301 : alterner les conditions, ne jamais les mesurer en blocs
successifs.

| Condition | Levier | Ce qu'elle isole |
|-----------|--------|------------------|
| PHY Ethernet actif / en veille | `-DETH_PHY_POWERDOWN` (2 flashes) | consommation du PHY LAN8742A sur le rail VDD_MCU |
| LED PA5 allumée / éteinte | pilotable sans reflash | consommation d'une LED — le témoin de sanité de l'A/B |
| UART actif / inactif | **gratuit** : point `rate = 0` de S5304 | coût du trafic UART hors calcul |
| Repos scrutation / repos WFI | `-DUART_WFI_IDLE` (S5302) | coût du busy-wait à 180 MHz |

La LED sert de **contrôle positif** : son écart est connu et calculable (~1–3 mA selon la
résistance de la NUCLEO). Si l'A/B ne le voit pas, la méthode n'est pas assez sensible pour
voir le PHY non plus, et il faut le dire.

### 2. Remplir le profil matériel

`configs/hw_profile_f439zi.yaml:34-36` porte `actif_mA: null` et `veille_uA: null` depuis
le Sprint 33. Les remplir avec les valeurs **mesurées** :

- `actif_mA` ← courant sous charge d'inférence, à cadence documentée dans le champ voisin.
- `veille_uA` ← repos WFI (S5302). Si S5302 n'aboutit pas, ce champ **reste `null`** — la
  scrutation active à 54,8 mA n'est pas une veille et ne doit pas être écrite comme telle.

Consigner aussi le résultat de `by_component` dans les cellules, en remplaçant les
`"à mesurer"` par les valeurs différentielles **ou** par une `na_reason` mise à jour si
l'A/B n'est pas concluant.

### 3. Autonomie à duty-cycle réaliste

Avec le repos WFI (S5302) et la pente (S5304), l'autonomie d'un déploiement s'écrit :

```
I_moy(f_usage) = I_repos_WFI + pente · f_usage
Autonomie_h    = Capacité_mAh / I_moy_mA
```

Balayer `f_usage ∈ {0,1 ; 1 ; 10 ; 100} Hz` × 5 capacités de batterie
(`hw_profile_f439zi.yaml:116-122`) × modèles. C'est enfin l'autonomie d'un **système
déployé** — celle qu'un industriel lit — et non celle du régime de banc.

Renseigner `configs/energy_campaign_s50.yaml:21` (`inference_period_s`) avec les valeurs
balayées, en laissant le `TODO(fred)` ouvert sur le **choix** du profil d'usage : mesurer
plusieurs points ne dispense pas de faire valider lequel correspond au cas d'usage
industriel visé.

### 4. Sortie

`experiments/exp_S53_components/ab_isolation.json` (conditions, ordre de session, deltas et
leur significativité) et `autonomy_duty_cycle.json` (matrice f_usage × capacité × modèle,
avec `regime: "duty-cycle modélisé depuis pente mesurée"`).

## Critères d'acceptation

- [ ] A/B PHY refait **en session unique, ordre contre-balancé** ; le résultat S50
      (63,24 mA) est soit confirmé, soit corrigé, avec la raison.
- [ ] Contrôle positif LED : l'écart est détecté, ou l'insuffisance de sensibilité est
      énoncée.
- [ ] `actif_mA` renseigné ; `veille_uA` renseigné **seulement** si le WFI a abouti, sinon
      laissé `null` avec un commentaire.
- [ ] `by_component.{mcu,periph}` renseigné ou N/A avec raison **mise à jour**.
- [ ] `sensor` reste `"na"` (capteurs simulés) — inchangé.
- [ ] L'autonomie duty-cycle porte un `regime` explicite et **ne remplace pas**
      `exp_S50_energy/autonomy.json` : elle s'ajoute, avec son hypothèse déclarée.

## Limite honnête à conserver

La sonde mesure un rail entier. Toute valeur `by_component` est une **différence entre deux
configurations**, pas une mesure directe d'un composant. Le formuler ainsi dans le champ
`method` de chaque cellule — c'est ce qui distingue une isolation mesurée d'une isolation
supposée.
