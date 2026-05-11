/*
 * Linker script STM32F439ZITx — Flash execution
 * Flash : 2 Mo @ 0x08000000
 * RAM   : 192 Ko @ 0x20000000 (SRAM1+SRAM2 contiguë — CCM 64Ko à 0x10000000 non utilisé ici)
 * Note  : STM32F439ZI expose 192 Ko au bus AHB (0x20000000–0x2002FFFF).
 *         Les 64 Ko CCM sont à 0x10000000 (inaccessibles par DMA).
 */

ENTRY(Reset_Handler)

_estack = ORIGIN(RAM) + LENGTH(RAM);   /* Haut de la pile */

MEMORY
{
  FLASH (rx) : ORIGIN = 0x08000000, LENGTH = 2048K
  RAM  (xrw) : ORIGIN = 0x20000000, LENGTH = 192K
}

SECTIONS
{
  /* Table des vecteurs + code */
  .isr_vector :
  {
    . = ALIGN(4);
    KEEP(*(.isr_vector))
    . = ALIGN(4);
  } >FLASH

  .text :
  {
    . = ALIGN(4);
    *(.text)
    *(.text*)
    *(.rodata)
    *(.rodata*)
    KEEP(*(.init))
    KEEP(*(.fini))
    . = ALIGN(4);
    _etext = .;
  } >FLASH

  /* Données initialisées (stockées en Flash, copiées en RAM au reset) */
  _sidata = LOADADDR(.data);

  .data :
  {
    . = ALIGN(4);
    _sdata = .;
    *(.data)
    *(.data*)
    . = ALIGN(4);
    _edata = .;
  } >RAM AT> FLASH

  /* Données non initialisées (zéro-initialisées par Reset_Handler) */
  .bss :
  {
    . = ALIGN(4);
    _sbss = .;
    *(.bss)
    *(.bss*)
    *(COMMON)
    . = ALIGN(4);
    _ebss = .;
  } >RAM

  /DISCARD/ :
  {
    libc.a(*)
    libm.a(*)
    libgcc.a(*)
  }
}
