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
              src/ewc_head.c \
              src/tinyol.c \
              src/hdc.c \
              src/metrics.c \
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

clean:
	rm -rf $(BUILD_DIR)

# ── Host unit tests (x86 natif — pas de toolchain ARM requis) ────────────
# On compile uniquement les modules portables (pas pipeline.c / main.c / hw_info.c
# qui ont des dépendances HAL STM32 incompatibles avec x86).
TEST_CC  = gcc
TEST_SRC = tests/test_mahalanobis.c tests/test_ewc_head.c \
           tests/test_profiling.c tests/test_pipeline.c tests/test_tinyol.c \
           tests/test_hdc.c \
           tests/test_runner.c \
           tests/unity/unity.c \
           src/mahalanobis.c src/ewc_head.c src/tinyol.c src/hdc.c src/profiling.c src/pipeline.c \
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

.PHONY: all flash debug clean test test-models
