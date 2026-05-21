#include "tim.h"

TIM_HandleTypeDef htim3;

/* TIM3 PWM CH1 sur PA6 @ 1 kHz — TIMxCLK = 16 MHz (HSI sans PLL)
   freq = TIMxCLK / ((PSC+1) * (ARR+1)) = 16 000 000 / (16 * 1000) = 1 000 Hz
   NOTE : le sprint doc suppose 90 MHz (APB1=45 MHz * 2 avec PLL).
          On utilise PSC=15 au lieu de PSC=89 pour rester à 1 kHz sur HSI. */
void MX_TIM3_Init(void)
{
    TIM_OC_InitTypeDef sConfigOC = {0};

    htim3.Instance               = TIM3;
    htim3.Init.Prescaler         = 15;   /* PSC=15 → div=16 ; 16 MHz/16 = 1 MHz tick */
    htim3.Init.CounterMode       = TIM_COUNTERMODE_UP;
    htim3.Init.Period            = 999;  /* ARR=999 → 1000 ticks → 1 kHz */
    htim3.Init.ClockDivision     = TIM_CLOCKDIVISION_DIV1;
    htim3.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
    HAL_TIM_PWM_Init(&htim3);

    sConfigOC.OCMode     = TIM_OCMODE_PWM1;
    sConfigOC.Pulse      = 500;          /* CCR=500 → duty 50 % (= ARR*0.5) */
    sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
    sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
    HAL_TIM_PWM_ConfigChannel(&htim3, &sConfigOC, TIM_CHANNEL_1);
}

/* HAL TIM PWM MSP : active l'horloge TIM3 et configure PA6 en AF2 (TIM3_CH1) */
void HAL_TIM_PWM_MspInit(TIM_HandleTypeDef *htim)
{
    GPIO_InitTypeDef GPIO_InitStruct = {0};
    if (htim->Instance == TIM3)
    {
        __HAL_RCC_TIM3_CLK_ENABLE();
        __HAL_RCC_GPIOA_CLK_ENABLE();

        /* PA6 → TIM3_CH1 (Alternate Function AF2) */
        GPIO_InitStruct.Pin       = GPIO_PIN_6;
        GPIO_InitStruct.Mode      = GPIO_MODE_AF_PP;
        GPIO_InitStruct.Pull      = GPIO_NOPULL;
        GPIO_InitStruct.Speed     = GPIO_SPEED_FREQ_LOW;
        GPIO_InitStruct.Alternate = GPIO_AF2_TIM3;
        HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);
    }
}
