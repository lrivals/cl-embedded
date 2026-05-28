# S1705 — Tests C UART mock + CI Renode + MAJ roadmap

| Champ | Valeur |
|-------|--------|
| **ID** | S1705 |
| **Sprint** | Sprint 17 — Tâches transverses |
| **Priorité** | 🟡 Important |
| **Durée estimée** | 2.5h |
| **Dépendances** | S1702 ✅ (printf debug), S1704 ✅ (Renode opérationnel) |
| **Fichiers cibles** | `firmware/stm32f4_blink/tests/test_pipeline.c`, `.github/workflows/firmware.yml`, `docs/roadmap_phase2.md` |
| **Statut** | ✅ Terminé (25 mai 2026) |

---

## Objectif

Compléter l'infrastructure de qualité du firmware : ajouter un mock UART dans les tests Unity existants pour couvrir le mode debug printf (S17-08), étendre la CI GitHub Actions avec un step Renode headless (S17-14), et mettre à jour la roadmap Phase 2 pour refléter l'avancement du Sprint 17.

---

## Sous-tâches

| ID | Description | Durée |
|----|-------------|:---:|
| **S17-17** | Mock UART dans Unity — test `pipeline.c` mode debug | 1h |
| **S17-18** | Step Renode dans `.github/workflows/firmware.yml` | 1h |
| **S17-19** | MAJ `docs/roadmap_phase2.md` — Sprint 17 statut + S1008/S1009 | 0.5h |

---

## Spécification

### S17-17 : Mock UART dans Unity

**Problème** : `pipeline.c` appelle `uart_send_byte()` qui accède à `USART3->DR` — registre inexistant sur x86. Les tests Unity tournent sur x86 (CI), donc il faut mocker ce périphérique.

**Mécanisme existant** : Sprint 16 a déjà mis en place des mocks dans `firmware/stm32f4_blink/tests/` — vérifier comment `mahalanobis.c` est testé (pas de UART, donc pas de mock UART pour l'instant).

**Extension à ajouter dans `test_pipeline.c`** :

```c
/* Mock UART pour tests x86 — remplace uart_send_byte() */
#ifdef TEST_MODE

static char uart_tx_buf[256];
static int  uart_tx_idx = 0;

void uart_send_byte(uint8_t b)
{
    if (uart_tx_idx < (int)sizeof(uart_tx_buf) - 1)
        uart_tx_buf[uart_tx_idx++] = (char)b;
}

const char *uart_tx_get(void) { return uart_tx_buf; }
void uart_tx_reset(void)      { uart_tx_idx = 0; memset(uart_tx_buf, 0, sizeof(uart_tx_buf)); }

#endif /* TEST_MODE */
```

**Nouveaux tests à ajouter** :

```c
void test_pipeline_debug_printf_contains_score(void)
{
    uart_tx_reset();
    /* Simuler un appel pipeline complet avec DEBUG_PRINTF=1 */
    float features[5] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
    pipeline_process(features, 5);
    /* Vérifier que le buffer UART contient "score=" */
    TEST_ASSERT(strstr(uart_tx_get(), "score=") != NULL);
}

void test_pipeline_response_binary_9bytes(void)
{
    uart_tx_reset();
    float features[5] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
    pipeline_process(features, 5);
    /* Réponse binaire = pred(1B) + conf(4B) + lat(4B) = 9B */
    TEST_ASSERT_EQUAL_INT(9, uart_tx_idx);
}
```

Le `Makefile` doit compiler les tests avec `-DTEST_MODE=1` et `-DDEBUG_PRINTF=1`.

### S17-18 : CI GitHub Actions avec Renode

**Fichier existant** : `.github/workflows/firmware.yml` (créé en Sprint 16, `make test` 16/16 PASS)

**Step Renode à ajouter** après le step `make test` :

```yaml
- name: Install Renode
  run: |
    sudo apt-get update -qq
    sudo apt-get install -y renode || \
    (wget -q https://github.com/renode/renode/releases/download/v1.14.0/renode_1.14.0_amd64.deb \
     && sudo dpkg -i renode_1.14.0_amd64.deb)
    renode --version

- name: Build firmware for Renode
  run: make -C firmware/stm32f4_blink -j4

- name: Run Renode simulation
  run: |
    chmod +x firmware/renode/run_mahalanobis_sim.sh
    ./firmware/renode/run_mahalanobis_sim.sh
  timeout-minutes: 2
```

**Condition** : ce step ne se lance que si `run_mahalanobis_sim.sh` existe (créé en S17-14). Ajouter une vérification :
```yaml
- name: Run Renode simulation
  if: hashFiles('firmware/renode/run_mahalanobis_sim.sh') != ''
  run: ./firmware/renode/run_mahalanobis_sim.sh
```

### S17-19 : MAJ roadmap_phase2.md

Sections à mettre à jour dans `docs/roadmap_phase2.md` :

1. Entrée Sprint 17 : `⬜ À démarrer` → `✅ CLÔTURÉ` (après completion)
2. S1008 (`stm32f4_cubemx_cmake_demo`) et S1009 (`led_blink_cubemx`) : statut `🆕` → `✅` avec référence vers `S1701_cubemx_gpio_setup.md`
3. Ajouter ligne dans le tableau macro :

```markdown
Sprint 17 (20–27 mai)    → NUCLEO-F439ZI : GPIO/UART/TIM/Renode (4 objectifs)
```

---

## Implémentation

### S17-17 : Tests Unity UART mock (1h)

```bash
# Vérifier les tests existants
cat firmware/stm32f4_blink/tests/test_pipeline.c

# Ajouter le mock UART et les 2 nouveaux tests (voir spécification)
# Rebuilder les tests x86
cd firmware/stm32f4_blink
make test
# Attendu : 18/18 tests PASS (16 existants + 2 nouveaux)
```

Si `pipeline_process()` n'est pas encore une fonction exposée dans `pipeline.h`, créer le wrapper :
```c
/* pipeline.h */
void pipeline_process(const float *features, int n);
```

### S17-18 : CI Renode (1h)

Éditer `.github/workflows/firmware.yml` — ajouter les steps Renode après les steps existants (build + test x86). Vérifier que le workflow passe en local avec `act` (si installé) ou pousser une branche test.

```bash
# Vérifier la syntaxe YAML
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/firmware.yml'))"
```

### S17-19 : MAJ roadmap (0.5h)

Modifier `docs/roadmap_phase2.md` :
- Sprint 17 entrée : ajouter statut + lien `docs/sprints/sprint_17/`
- S1008/S1009 : mettre à jour statut dans le tableau Sprint 16
- Note de numérotation : préciser que Sprint 17 = embedded examples NUCLEO-F439ZI

---

## Critères d'acceptation

- [x] `make test` → **24/24 PASS** (22 existants + 2 nouveaux tests UART mock pipeline)
- [x] Mock `uart_send_byte()` / `uart_getbyte()` sous `#ifdef TEST_MODE` — aucun accès registre USART3 sur x86
- [x] `test_pipeline_debug_printf_contains_score` : vérifie présence de `"score="` dans le buffer TX
- [x] `test_pipeline_response_v2_14bytes` : vérifie que `uart_send_response_v2` émet exactement 14 octets
- [x] `.github/workflows/firmware.yml` syntaxe YAML valide (step Renode déjà présent)
- [x] `docs/roadmap_phase2.md` : Sprint 17 ✅ CLÔTURÉ, O3/O4/O5 mis à jour
- [x] Le workflow CI ne régresse pas sur les 22 tests existants

---

## Statut

✅ Terminé (25 mai 2026)
