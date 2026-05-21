cmake_minimum_required(VERSION 3.20)

# Cross-compilation : désactiver le test de link (pas de main() cible)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

project(stm32f4_cubemx C ASM)

# ── Système cible ────────────────────────────────────────────────────────────
set(CMAKE_SYSTEM_NAME      Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)

# ── Toolchain ARM GCC ────────────────────────────────────────────────────────
set(CMAKE_C_COMPILER   arm-none-eabi-gcc)
set(CMAKE_ASM_COMPILER arm-none-eabi-gcc)
set(CMAKE_OBJCOPY      arm-none-eabi-objcopy)
set(CMAKE_SIZE         arm-none-eabi-size)

# ── Flags MCU Cortex-M4 FPU ─────────────────────────────────────────────────
set(MCU_FLAGS "-mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard")

set(CMAKE_C_FLAGS
    "${MCU_FLAGS} \
     -Wall -Wextra \
     -fdata-sections -ffunction-sections \
     -std=gnu11"
    CACHE STRING "" FORCE
)
set(CMAKE_C_FLAGS_DEBUG   "-Og -g3"    CACHE STRING "" FORCE)
set(CMAKE_C_FLAGS_RELEASE "-Os -DNDEBUG" CACHE STRING "" FORCE)

set(CMAKE_ASM_FLAGS "${MCU_FLAGS} -g3" CACHE STRING "" FORCE)

# ── Sources C générées par CubeMX ────────────────────────────────────────────
# CubeMX (Makefile toolchain) génère :
#   Core/Src/          : main.c, stm32f4xx_it.c, stm32f4xx_hal_msp.c, system_stm32f4xx.c
#   Drivers/STM32F4xx_HAL_Driver/Src/ : pilotes HAL (exclure *_template.c)
file(GLOB_RECURSE CORE_SOURCES
    CONFIGURE_DEPENDS
    "Core/Src/*.c"
)
file(GLOB_RECURSE HAL_SOURCES
    CONFIGURE_DEPENDS
    "Drivers/STM32F4xx_HAL_Driver/Src/*.c"
)
# Exclure les fichiers template HAL (ne compilent pas standalone)
list(FILTER HAL_SOURCES EXCLUDE REGEX ".*_template\\.c$")

# Fichier de démarrage ASM (racine du projet, toolchain Makefile)
set(STARTUP_SOURCE "${CMAKE_SOURCE_DIR}/startup_stm32f439xx.s")

set(ALL_SOURCES
    ${CORE_SOURCES}
    ${HAL_SOURCES}
    ${STARTUP_SOURCE}
)

# ── Includes ─────────────────────────────────────────────────────────────────
include_directories(
    Core/Inc
    Drivers/STM32F4xx_HAL_Driver/Inc
    Drivers/STM32F4xx_HAL_Driver/Inc/Legacy
    Drivers/CMSIS/Device/ST/STM32F4xx/Include
    Drivers/CMSIS/Include
)

# ── Defines ──────────────────────────────────────────────────────────────────
add_definitions(
    -DSTM32F439xx
    -DUSE_HAL_DRIVER
)

# ── Linker ───────────────────────────────────────────────────────────────────
set(LINKER_SCRIPT "${CMAKE_SOURCE_DIR}/STM32F439ZITx_FLASH.ld")

set(CMAKE_EXE_LINKER_FLAGS
    "${MCU_FLAGS} \
     -specs=nano.specs \
     -lc -lm \
     -Wl,--gc-sections \
     -Wl,-Map=${CMAKE_BINARY_DIR}/${PROJECT_NAME}.map,--cref \
     -Wl,--print-memory-usage \
     -T${LINKER_SCRIPT}"
    CACHE STRING "" FORCE
)

# ── Exécutable ───────────────────────────────────────────────────────────────
add_executable(${PROJECT_NAME}.elf ${ALL_SOURCES})

# ── Post-build : .hex, .bin, taille ─────────────────────────────────────────
add_custom_command(TARGET ${PROJECT_NAME}.elf POST_BUILD
    COMMAND ${CMAKE_OBJCOPY} -O ihex
            $<TARGET_FILE:${PROJECT_NAME}.elf>
            ${CMAKE_BINARY_DIR}/${PROJECT_NAME}.hex
    COMMAND ${CMAKE_OBJCOPY} -O binary
            $<TARGET_FILE:${PROJECT_NAME}.elf>
            ${CMAKE_BINARY_DIR}/${PROJECT_NAME}.bin
    COMMAND ${CMAKE_SIZE} $<TARGET_FILE:${PROJECT_NAME}.elf>
    COMMENT "Generating .hex / .bin / size report"
)

# ── Cible flash OpenOCD ──────────────────────────────────────────────────────
add_custom_target(flash
    DEPENDS ${PROJECT_NAME}.elf
    COMMAND openocd
            -f interface/stlink.cfg
            -f target/stm32f4x.cfg
            -c "program ${CMAKE_BINARY_DIR}/${PROJECT_NAME}.elf verify reset exit"
    COMMENT "Flashing via OpenOCD ST-LINK"
)
