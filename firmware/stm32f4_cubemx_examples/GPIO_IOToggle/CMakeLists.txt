cmake_minimum_required(VERSION 3.20)

set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

project(gpio_iotoggle C ASM)

set(CMAKE_SYSTEM_NAME      Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)

set(CMAKE_C_COMPILER   arm-none-eabi-gcc)
set(CMAKE_ASM_COMPILER arm-none-eabi-gcc)
set(CMAKE_OBJCOPY      arm-none-eabi-objcopy)
set(CMAKE_SIZE         arm-none-eabi-size)

set(MCU_FLAGS "-mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard")

set(CMAKE_C_FLAGS
    "${MCU_FLAGS} -Wall -Wextra -fdata-sections -ffunction-sections -std=gnu11"
    CACHE STRING "" FORCE
)
set(CMAKE_C_FLAGS_DEBUG   "-Og -g3"      CACHE STRING "" FORCE)
set(CMAKE_C_FLAGS_RELEASE "-Os -DNDEBUG" CACHE STRING "" FORCE)
set(CMAKE_ASM_FLAGS "${MCU_FLAGS} -g3"   CACHE STRING "" FORCE)

# ── Dossier racine du projet CubeMX (Drivers HAL partagés) ───────────────────
set(CUBEMX_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../stm32f4_cubemx")

# ── Sources ───────────────────────────────────────────────────────────────────
file(GLOB_RECURSE HAL_SOURCES
    CONFIGURE_DEPENDS
    "${CUBEMX_ROOT}/Drivers/STM32F4xx_HAL_Driver/Src/*.c"
)
list(FILTER HAL_SOURCES EXCLUDE REGEX ".*_template\\.c$")

set(ALL_SOURCES
    Src/main.c
    Src/stm32f4xx_it.c
    Src/system_stm32f4xx.c
    BSP/stm32f4xx_nucleo_144.c
    ${HAL_SOURCES}
    "${CUBEMX_ROOT}/startup_stm32f439xx.s"
)

# ── Includes ──────────────────────────────────────────────────────────────────
include_directories(
    Inc
    BSP
    ${CUBEMX_ROOT}/Drivers/STM32F4xx_HAL_Driver/Inc
    ${CUBEMX_ROOT}/Drivers/STM32F4xx_HAL_Driver/Inc/Legacy
    ${CUBEMX_ROOT}/Drivers/CMSIS/Device/ST/STM32F4xx/Include
    ${CUBEMX_ROOT}/Drivers/CMSIS/Include
)

# ── Defines ───────────────────────────────────────────────────────────────────
add_definitions(
    -DSTM32F439xx
    -DUSE_HAL_DRIVER
    -DUSE_STM32F4XX_NUCLEO_144
)

# ── Linker ────────────────────────────────────────────────────────────────────
set(LINKER_SCRIPT "${CUBEMX_ROOT}/STM32F439ZITx_FLASH.ld")

set(CMAKE_EXE_LINKER_FLAGS
    "${MCU_FLAGS} \
     -specs=nano.specs \
     -lc -lm -lnosys \
     -Wl,--gc-sections \
     -Wl,-Map=${CMAKE_BINARY_DIR}/${PROJECT_NAME}.map,--cref \
     -Wl,--print-memory-usage \
     -T${LINKER_SCRIPT}"
    CACHE STRING "" FORCE
)

# ── Exécutable ────────────────────────────────────────────────────────────────
add_executable(${PROJECT_NAME}.elf ${ALL_SOURCES})

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
