# Makefile — STM32F439ZI Blink (Phase 2 CL-Embedded)
# Cible : NUCLEO-F439ZI, Cortex-M4 @ 180 MHz, 256 Ko RAM
# Toolchain : arm-none-eabi-gcc + OpenOCD (ST-LINK intégré)

TARGET    = stm32f4_blink
BUILD_DIR = build

# ── Toolchain ────────────────────────────────────────────────────────────
CC      = arm-none-eabi-gcc
AS      = arm-none-eabi-gcc -x assembler-with-cpp
OBJCOPY = arm-none-eabi-objcopy
SIZE    = arm-none-eabi-size

# ── Sources ──────────────────────────────────────────────────────────────
C_SOURCES   = src/main.c \
              src/hw_info.c \
              src/mahalanobis.c \
              src/mahalanobis_q15.c \
              src/mahalanobis_int8.c \
              src/ewc_head.c \
              src/ewc_head_int8.c \
              src/ewc_head_int8_v2.c \
              src/ewc_head_regression.c \
              src/ewc_head_multiclass.c \
              src/tinyol.c \
              src/tinyol_int8.c \
              src/hdc.c \
              src/hdc_int8.c \
              src/meta_head.c \
              src/metrics.c \
              src/ring_buffer.c \
              src/drift_detector.c \
              src/drift/page_hinkley.c \
              src/drift/ddm.c \
              src/drift/psi.c \
              src/pipeline.c \
              src/profiling.c \
              src/eth_phy.c \
              src/syscalls.c
ASM_SOURCES = startup/startup_stm32f439xx.s

# ── Includes ─────────────────────────────────────────────────────────────
C_INCLUDES = -Iinc -Iinc/drift

# ── CPU flags ────────────────────────────────────────────────────────────
CPU      = -mcpu=cortex-m4
FPU      = -mfpu=fpv4-sp-d16
FLOAT    = -mfloat-abi=hard
MCU      = $(CPU) -mthumb $(FPU) $(FLOAT)

# ── Flags de compilation ─────────────────────────────────────────────────
CFLAGS   = $(MCU) $(C_INCLUDES) -DSTM32F439xx \
           -Wall -Wextra -Os -g3 \
           -ffunction-sections -fdata-sections \
           -fstack-usage \
           -std=gnu11
# -fstack-usage : émet un build/<unité>.su (cadre de pile par fonction). N'affecte
#   PAS le code généré (fichier annexe uniquement → .text/.data/.bss bit-identiques).
#   Exploité par scripts/stack_usage_report.py comme borne supérieure statique de pile,
#   contre-vérifiée par le pic mesuré (scripts/measure_stack_watermark.py).

# Flags additionnels passés en ligne de commande SANS clobber les flags ci-dessus.
# Ex. marqueurs énergie (S3304) :  make all EXTRA_CFLAGS=-DENERGY_MARKERS
# (NE PAS utiliser `make CFLAGS+=...` : un CFLAGS en ligne de commande écrase
#  entièrement la définition du Makefile, y compris -Iinc et les flags MCU.)
EXTRA_CFLAGS ?=
CFLAGS += $(EXTRA_CFLAGS)

# Taille de la fenêtre du buffer de streaming instrumenté (S3403). Rebuild par W :
#   make all STREAM_BUF_W=10   →  .bss varie avec W (mesuré par arm-none-eabi-size)
STREAM_BUF_W ?= 5
CFLAGS += -DSTREAM_BUF_W=$(STREAM_BUF_W)

# Dimensions d'entrée des modèles, surchargeables au build (S3506). Non définies
# par défaut → les headers inc/ gardent leur valeur native via #ifndef (ligne de
# commande compilateur identique ⇒ .bss 5feat bit-identique). Override par condition :
#   make EWC_IN=9 MAHA_DIM=9 size            # condition all/best 9-feat (CWRU)
#   make PROTO_MAX_N=21 size                 # condition all CMAPSS (21 capteurs > 16)
# Poids réels par condition regénérés ensuite via export_weights_c.py (S3507).
ifdef EWC_IN
CFLAGS += -DEWC_IN=$(EWC_IN)
endif
ifdef MAHA_DIM
CFLAGS += -DMAHA_DIM=$(MAHA_DIM)
endif
ifdef TINYOL_IN
CFLAGS += -DTINYOL_IN=$(TINYOL_IN)
endif
ifdef HDC_N_FEATURES
CFLAGS += -DHDC_N_FEATURES=$(HDC_N_FEATURES)
endif
ifdef PROTO_MAX_N
CFLAGS += -DPROTO_MAX_N=$(PROTO_MAX_N)
endif
# Capacité max de la fenêtre du gate de dérive (S3803). Couvre window_size=50 par
# défaut (DRIFT_WINDOW_MAX=64 dans drift_detector.h) ; surchargeable au build.
ifdef DRIFT_WINDOW_MAX
CFLAGS += -DDRIFT_WINDOW_MAX=$(DRIFT_WINDOW_MAX)
endif
# Sélection du détecteur de drift porté (S4502). Activation build-time (nibble UART saturé) :
#   make EXTRA_CFLAGS="-DDRIFT_DETECT -DDRIFT_METHOD=DRIFT_PSI"     # une méthode par binaire
# PSI_BINS / PSI_BLOCK_SIZE surchargeables (histogramme à bacs fixes). -DDRIFT_DETECT non posé
# par défaut → détecteurs liés mais inactifs, .bss invariant.
ifdef PSI_BINS
CFLAGS += -DPSI_BINS=$(PSI_BINS)
endif
ifdef PSI_BLOCK_SIZE
CFLAGS += -DPSI_BLOCK_SIZE=$(PSI_BLOCK_SIZE)
endif

ASFLAGS  = $(MCU) -g3

# ── Linker ───────────────────────────────────────────────────────────────
LDSCRIPT  = linker/STM32F439ZITx_FLASH.ld
LDFLAGS   = $(MCU) -specs=nano.specs -u _printf_float -T$(LDSCRIPT) \
            -Wl,--gc-sections -Wl,-Map=$(BUILD_DIR)/$(TARGET).map,--cref \
            -Wl,--print-memory-usage

# ── Objets ───────────────────────────────────────────────────────────────
OBJECTS  = $(addprefix $(BUILD_DIR)/,$(notdir $(C_SOURCES:.c=.o)))
OBJECTS += $(addprefix $(BUILD_DIR)/,$(notdir $(ASM_SOURCES:.s=.o)))

# ── Règles ───────────────────────────────────────────────────────────────
all: $(BUILD_DIR)/$(TARGET).elf $(BUILD_DIR)/$(TARGET).bin

$(BUILD_DIR)/%.o: src/%.c | $(BUILD_DIR)
	$(CC) -c $(CFLAGS) -o $@ $<

# Détecteurs de drift portés en sous-répertoire src/drift/ (S4502). Objets aplatis
# (notdir) — pas de collision de noms avec src/*.c.
$(BUILD_DIR)/%.o: src/drift/%.c | $(BUILD_DIR)
	$(CC) -c $(CFLAGS) -o $@ $<

$(BUILD_DIR)/%.o: startup/%.s | $(BUILD_DIR)
	$(AS) -c $(ASFLAGS) -o $@ $<

$(BUILD_DIR)/$(TARGET).elf: $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@
	$(SIZE) $@

$(BUILD_DIR)/$(TARGET).bin: $(BUILD_DIR)/$(TARGET).elf
	$(OBJCOPY) -O binary $< $@

$(BUILD_DIR):
	mkdir -p $@

# ── Flash via OpenOCD + ST-LINK ───────────────────────────────────────────
flash: $(BUILD_DIR)/$(TARGET).elf
	openocd -f interface/stlink.cfg \
	        -f target/stm32f4x.cfg \
	        -c "program $(BUILD_DIR)/$(TARGET).elf verify reset exit"

# ── Debug (launch depuis VS Code) ────────────────────────────────────────
debug: $(BUILD_DIR)/$(TARGET).elf
	openocd -f interface/stlink.cfg \
	        -f target/stm32f4x.cfg

# ── Taille Flash/RAM du binaire (text/data/bss) ──────────────────────────
size: $(BUILD_DIR)/$(TARGET).elf
	$(SIZE) $(BUILD_DIR)/$(TARGET).elf

clean:
	rm -rf $(BUILD_DIR)

# ── Host unit tests (x86 natif — pas de toolchain ARM requis) ────────────
# On compile uniquement les modules portables (pas pipeline.c / main.c / hw_info.c
# qui ont des dépendances HAL STM32 incompatibles avec x86).
TEST_CC  = gcc
TEST_SRC = tests/test_mahalanobis.c tests/test_mahalanobis_q15.c tests/test_mahalanobis_int8.c tests/test_ewc_head.c \
           tests/test_ewc_int8.c \
           tests/test_ewc_int8_v2.c \
           tests/test_ewc_subint8.c \
           tests/test_profiling.c tests/test_pipeline.c tests/test_tinyol.c \
           tests/test_hdc.c \
           tests/test_ring_buffer.c \
           tests/test_hdc_int8.c \
           tests/test_tinyol_int8.c \
           tests/test_metrics.c \
           tests/test_ewc_regression.c \
           tests/test_ewc_multiclass.c \
           tests/test_meta_head.c \
           tests/test_drift_detector.c \
           tests/test_drift_methods.c \
           tests/test_eth_phy.c \
           tests/test_runner.c \
           tests/unity/unity.c \
           src/mahalanobis.c src/mahalanobis_q15.c src/mahalanobis_int8.c src/ewc_head.c src/ewc_head_int8.c src/ewc_head_int8_v2.c src/ewc_head_regression.c \
           src/ewc_head_multiclass.c \
           src/tinyol.c src/tinyol_int8.c src/hdc.c src/hdc_int8.c \
           src/meta_head.c \
           src/ring_buffer.c \
           src/drift_detector.c \
           src/drift/page_hinkley.c src/drift/ddm.c src/drift/psi.c \
           src/profiling.c src/pipeline.c \
           src/metrics.c
TEST_INC = -Iinc -Iinc/drift -Itests/unity

test: $(BUILD_DIR)
	$(TEST_CC) $(TEST_SRC) $(TEST_INC) -DTEST_HOST -DTEST_MODE=1 -DDEBUG_PRINTF=1 -lm -o $(BUILD_DIR)/test_runner
	./$(BUILD_DIR)/test_runner

# Variante Q15 du kernel INT8 v2 (S3909) : recompile le runner avec -DEWC_INT8_Q15
# (typedef 16 bits, exclusif de l'int8) pour exécuter test_v2_q15_parity.
test-v2-q15: $(BUILD_DIR)
	$(TEST_CC) $(TEST_SRC) $(TEST_INC) -DTEST_HOST -DTEST_MODE=1 -DDEBUG_PRINTF=1 -DEWC_INT8_Q15 -lm -o $(BUILD_DIR)/test_runner_q15
	./$(BUILD_DIR)/test_runner_q15

# Variantes sub-INT8 (Sprint 48, S4802) : recompile le runner par profondeur de build.
# Non-packé (conteneur int8) puis packé (2/4/8 poids par octet). Parité vs émulateur.
test-sub-int4: $(BUILD_DIR)
	$(TEST_CC) $(TEST_SRC) $(TEST_INC) -DTEST_HOST -DTEST_MODE=1 -DDEBUG_PRINTF=1 -DEWC_INT4 -lm -o $(BUILD_DIR)/test_runner_int4
	./$(BUILD_DIR)/test_runner_int4

test-sub-int4-packed: $(BUILD_DIR)
	$(TEST_CC) $(TEST_SRC) $(TEST_INC) -DTEST_HOST -DTEST_MODE=1 -DDEBUG_PRINTF=1 -DEWC_INT4 -DEWC_INTx_PACKED -lm -o $(BUILD_DIR)/test_runner_int4p
	./$(BUILD_DIR)/test_runner_int4p

test-sub-int2: $(BUILD_DIR)
	$(TEST_CC) $(TEST_SRC) $(TEST_INC) -DTEST_HOST -DTEST_MODE=1 -DDEBUG_PRINTF=1 -DEWC_INT2 -lm -o $(BUILD_DIR)/test_runner_int2
	./$(BUILD_DIR)/test_runner_int2

test-sub-int2-packed: $(BUILD_DIR)
	$(TEST_CC) $(TEST_SRC) $(TEST_INC) -DTEST_HOST -DTEST_MODE=1 -DDEBUG_PRINTF=1 -DEWC_INT2 -DEWC_INTx_PACKED -lm -o $(BUILD_DIR)/test_runner_int2p
	./$(BUILD_DIR)/test_runner_int2p

test-sub-binary: $(BUILD_DIR)
	$(TEST_CC) $(TEST_SRC) $(TEST_INC) -DTEST_HOST -DTEST_MODE=1 -DDEBUG_PRINTF=1 -DEWC_INT1 -lm -o $(BUILD_DIR)/test_runner_bin
	./$(BUILD_DIR)/test_runner_bin

test-sub-binary-packed: $(BUILD_DIR)
	$(TEST_CC) $(TEST_SRC) $(TEST_INC) -DTEST_HOST -DTEST_MODE=1 -DDEBUG_PRINTF=1 -DEWC_INT1 -DEWC_INTx_PACKED -lm -o $(BUILD_DIR)/test_runner_binp
	./$(BUILD_DIR)/test_runner_binp

# Agrège toutes les variantes sub-INT8 (packé + non-packé).
test-sub-all: test-sub-int4 test-sub-int4-packed test-sub-int2 test-sub-int2-packed test-sub-binary test-sub-binary-packed

# Variante « lot d'inférences » (Sprint 53, S5302) : recompile le runner avec
# -DINFER_BATCH_N=N pour exercer les 3 invariants du lot (prédiction identique à
# N=1, UNE seule mise à jour CL, portée du lot). Le runner par défaut reste
# strictement inchangé — ces tests sont gardés sur INFER_BATCH_N > 1.
# N surchargeable : make test-batch BATCH_N=16
BATCH_N ?= 4
test-batch: $(BUILD_DIR)
	$(TEST_CC) $(TEST_SRC) $(TEST_INC) -DTEST_HOST -DTEST_MODE=1 -DDEBUG_PRINTF=1 -DINFER_BATCH_N=$(BATCH_N) -lm -o $(BUILD_DIR)/test_runner_batch
	./$(BUILD_DIR)/test_runner_batch

# Build de banc « repos réel » (Sprint 53, S5302) : endort l'attente UART.
# Vérifiable SANS carte — compile et rend la taille. Le `.bss` attendu est celui du
# build par défaut (le WFI n'ajoute pas d'état), l'écart éventuel doit être expliqué.
check-wfi:
	$(MAKE) clean
	$(MAKE) all EXTRA_CFLAGS="-DUART_WFI_IDLE"
	$(SIZE) $(BUILD_DIR)/$(TARGET).elf

# Balayage de fréquence système (Sprint 53, S5303) : les trois builds -DSYSCLK_MHZ.
# Vérifiable SANS carte — ce qui se prouve ici, c'est que les trois binaires COMPILENT et
# quelle taille ils occupent ; la fréquence réellement atteinte, elle, ne se constate qu'à
# bord (bannière `hw_info_print`, recalculée depuis PLLCFGR — cf. run_s53_freq_sweep.py).
# Le build par défaut est reconstruit en fin de cible : la cible ne laisse JAMAIS un
# binaire 45 MHz dans build/ prêt à être flashé par mégarde.
SYSCLK_LIST ?= 180 90 45
check-sysclk:
	@for f in $(SYSCLK_LIST); do \
	    echo "── SYSCLK $$f MHz ──"; \
	    $(MAKE) --no-print-directory clean >/dev/null; \
	    $(MAKE) --no-print-directory all EXTRA_CFLAGS="-DSYSCLK_MHZ=$$f" >/dev/null || exit 1; \
	    $(SIZE) $(BUILD_DIR)/$(TARGET).elf; \
	done
	@echo "── retour au build par défaut (180 MHz, sans -DSYSCLK_MHZ) ──"
	@$(MAKE) --no-print-directory clean >/dev/null
	@$(MAKE) --no-print-directory all >/dev/null
	@$(SIZE) $(BUILD_DIR)/$(TARGET).elf

TEST_MODELS_SRC = tests/test_models.c \
                  tests/unity/unity.c \
                  src/mahalanobis.c src/ewc_head.c src/tinyol.c src/metrics.c

test-models: $(BUILD_DIR)
	$(TEST_CC) $(TEST_MODELS_SRC) $(TEST_INC) -Itests \
	           -DTEST_HOST -lm -o $(BUILD_DIR)/test_models_runner
	./$(BUILD_DIR)/test_models_runner

.PHONY: all flash debug clean test test-v2-q15 test-models size \
        test-sub-int4 test-sub-int4-packed test-sub-int2 test-sub-int2-packed \
        test-sub-binary test-sub-binary-packed test-sub-all \
        test-batch check-wfi check-sysclk
