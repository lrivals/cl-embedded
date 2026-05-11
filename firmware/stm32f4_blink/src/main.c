/**
 * STM32F439ZI — Pipeline anomaly detection (S1603)
 *
 * Séquence au démarrage :
 *   1. hw_clock_init()   : HSI → PLL → SYSCLK 180 MHz
 *   2. hw_uart_init()    : USART3 @ 115200, PD8=TX PD9=RX (ST-LINK VCP)
 *   3. hw_info_collect / hw_info_print + hw_dwt_calibrate : rapport UART
 *   4. pipeline_init()   : GPIO PA5, détecteur Mahalanobis (poids Flash)
 *   5. Boucle : pipeline_run() — UART frame → inférence → réponse 9 B
 *
 * Sortie UART lisible sur : minicom -b 115200 -D /dev/ttyACM0
 */

#include "stm32f4xx.h"
#include "hw_info.h"
#include "pipeline.h"

int main(void)
{
    /* ── 1. Horloge système → 180 MHz ───────────────────────────────── */
    hw_clock_init();

    /* ── 2. UART3 @ 115200 (PD8/PD9 = ST-LINK VCP) ─────────────────── */
    hw_uart_init();

    /* ── 3. Rapport hardware + calibration DWT ──────────────────────── */
    /* MEM: HWInfo = 36 B @ FP32 — libéré en sortie de bloc */
    {
        HWInfo info;
        hw_info_collect(&info);
        hw_info_print(&info);
        hw_dwt_calibrate(info.sysclk_hz);
    }

    /* ── 4. Init pipeline (GPIO PA5, poids Mahalanobis depuis Flash) ── */
    pipeline_init();

    /* ── 5. Boucle d'inférence (bloque sur trame UART) ──────────────── */
    while (1) {
        pipeline_run();
    }
}
