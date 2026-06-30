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
              src/pipeline.c \
              src/profiling.c \
              src/syscalls.c
ASM_SOURCES = startup/startup_stm32f439xx.s

# ── Includes ─────────────────────────────────────────────────────────────
C_INCLUDES = -Iinc

# ── CPU flags ────────────────────────────────────────────────────────────
CPU      = -mcpu=cortex-m4
FPU      = -mfpu=fpv4-sp-d16
FLOAT    = -mfloat-abi=hard
MCU      = $(CPU) -mthumb $(FPU) $(FLOAT)

# ── Flags de compilation ─────────────────────────────────────────────────
CFLAGS   = $(MCU) $(C_INCLUDES) -DSTM32F439xx \
           -Wall -Wextra -Os -g3 \
           -ffunction-sections -fdata-sections \
           -std=gnu11

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
           tests/test_runner.c \
           tests/unity/unity.c \
           src/mahalanobis.c src/mahalanobis_q15.c src/mahalanobis_int8.c src/ewc_head.c src/ewc_head_int8.c src/ewc_head_regression.c \
           src/ewc_head_multiclass.c \
           src/tinyol.c src/tinyol_int8.c src/hdc.c src/hdc_int8.c \
           src/meta_head.c \
           src/ring_buffer.c \
           src/drift_detector.c \
           src/profiling.c src/pipeline.c \
           src/metrics.c
TEST_INC = -Iinc -Itests/unity

test: $(BUILD_DIR)
	$(TEST_CC) $(TEST_SRC) $(TEST_INC) -DTEST_HOST -DTEST_MODE=1 -DDEBUG_PRINTF=1 -lm -o $(BUILD_DIR)/test_runner
	./$(BUILD_DIR)/test_runner

TEST_MODELS_SRC = tests/test_models.c \
                  tests/unity/unity.c \
                  src/mahalanobis.c src/ewc_head.c src/tinyol.c src/metrics.c

test-models: $(BUILD_DIR)
	$(TEST_CC) $(TEST_MODELS_SRC) $(TEST_INC) -Itests \
	           -DTEST_HOST -lm -o $(BUILD_DIR)/test_models_runner
	./$(BUILD_DIR)/test_models_runner

.PHONY: all flash debug clean test test-models size
