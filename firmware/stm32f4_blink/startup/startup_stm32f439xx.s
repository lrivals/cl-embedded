/**
 * Fichier startup STM32F439xx — Cortex-M4
 * Source : ST Microelectronics (adapté pour projet blink minimal)
 * CMSIS-style vector table + Reset_Handler
 */

  .syntax unified
  .cpu cortex-m4
  .fpu softvfp
  .thumb

.global g_pfnVectors
.global Default_Handler

/* Symboles fournis par le linker script */
.word _sidata     /* Adresse source (Flash) des données initialisées */
.word _sdata      /* Début section .data en RAM */
.word _edata      /* Fin section .data en RAM */
.word _sbss       /* Début section .bss */
.word _ebss       /* Fin section .bss */

/**
 * Reset_Handler : point d'entrée après reset
 */
  .section .text.Reset_Handler
  .weak Reset_Handler
  .type Reset_Handler, %function
Reset_Handler:
  ldr   sp, =_estack

  /* Copier .data de Flash → RAM */
  movs  r1, #0
  b     LoopCopyDataInit

CopyDataInit:
  ldr   r3, =_sidata
  ldr   r3, [r3, r1]
  str   r3, [r0, r1]
  adds  r1, r1, #4

LoopCopyDataInit:
  ldr   r0, =_sdata
  ldr   r3, =_edata
  adds  r2, r0, r1
  cmp   r2, r3
  bcc   CopyDataInit

  /* Initialiser .bss à zéro */
  ldr   r2, =_sbss
  b     LoopFillZerobss

FillZerobss:
  movs  r3, #0
  str   r3, [r2], #4

LoopFillZerobss:
  ldr   r3, =_ebss
  cmp   r2, r3
  bcc   FillZerobss

  bl    main
  bx    lr

.size Reset_Handler, .-Reset_Handler

Default_Handler:
  b Default_Handler
.size Default_Handler, .-Default_Handler

/**
 * Table des vecteurs (STM32F439xx — 97 interruptions + 16 exceptions Cortex-M4)
 */
  .section .isr_vector,"a",%progbits
  .type g_pfnVectors, %object

g_pfnVectors:
  .word _estack
  .word Reset_Handler
  .word NMI_Handler
  .word HardFault_Handler
  .word MemManage_Handler
  .word BusFault_Handler
  .word UsageFault_Handler
  .word 0
  .word 0
  .word 0
  .word 0
  .word SVC_Handler
  .word DebugMon_Handler
  .word 0
  .word PendSV_Handler
  .word SysTick_Handler
  /* Interruptions externes (IRQ0–IRQ96) — Default_Handler par défaut */
  .rept 97
  .word Default_Handler
  .endr

.size g_pfnVectors, .-g_pfnVectors

/* Weak aliases — peuvent être redéfinis dans main.c ou autres fichiers */
  .weak NMI_Handler
  .thumb_set NMI_Handler,Default_Handler

  .weak HardFault_Handler
  .thumb_set HardFault_Handler,Default_Handler

  .weak MemManage_Handler
  .thumb_set MemManage_Handler,Default_Handler

  .weak BusFault_Handler
  .thumb_set BusFault_Handler,Default_Handler

  .weak UsageFault_Handler
  .thumb_set UsageFault_Handler,Default_Handler

  .weak SVC_Handler
  .thumb_set SVC_Handler,Default_Handler

  .weak DebugMon_Handler
  .thumb_set DebugMon_Handler,Default_Handler

  .weak PendSV_Handler
  .thumb_set PendSV_Handler,Default_Handler

  .weak SysTick_Handler
  .thumb_set SysTick_Handler,Default_Handler
