/* ewc_head_subint8_weights.h — GÉNÉRÉ par export_weights_c.py --ewc-subint8 (S4803).
 * Schéma : mode=binary bits=1 granularité=per_channel symétrie=symmetric packé=1.
 * Build firmware associé : -DEWC_INT1 -DEWC_INTx_PACKED.
 * Poids pré-quantifiés (parité émulateur subint8). NE PAS ÉDITER À LA MAIN. */
#ifndef EWC_HEAD_SUBINT8_WEIGHTS_H
#define EWC_HEAD_SUBINT8_WEIGHTS_H
#include <stdint.h>

#define EWC_SUBINT8_WEIGHTS_PROVIDED 1
#define EWC_SUBINT8_NATIVE_DIM 5
#define EWC_SUBINT8_PACK_BITS 1
#define EWC_SUBINT8_PACKED 1

static const uint8_t EWC_SUB_W1[32][1] = {
    {3},
    {13},
    {29},
    {26},
    {8},
    {8},
    {25},
    {1},
    {21},
    {26},
    {8},
    {8},
    {1},
    {2},
    {15},
    {24},
    {24},
    {3},
    {13},
    {2},
    {6},
    {0},
    {7},
    {0},
    {0},
    {8},
    {1},
    {8},
    {5},
    {12},
    {21},
    {26}
};
static const float EWC_SUB_SCALE_W1[32] = {0.15192851f, 0.20923868f, 0.17693016f, 0.27003437f, 0.29225668f, 0.38873449f, 0.32496792f, 0.17314574f, 0.18298051f, 0.29294789f, 0.37988558f, 0.27758914f, 0.17319354f, 0.38110375f, 0.29759765f, 0.20712979f, 0.18199405f, 0.29380995f, 0.17996450f, 0.40280277f, 0.26749906f, 0.40105090f, 0.35720605f, 0.37143528f, 0.29686356f, 0.31225091f, 0.17349288f, 0.34065336f, 0.30927485f, 0.21994719f, 0.23863164f, 0.27138001f};
static const float EWC_SUB_B1[32] = {0.84393364f, 0.74324137f, -0.55193835f, -0.99258506f, 0.80327350f, 0.34073782f, -0.84898329f, 0.90092951f, -0.89774948f, -0.82295263f, -0.08904590f, 0.60814387f, 0.72101068f, 0.20504458f, -0.68287510f, -0.76783162f, -0.75353509f, 0.32707593f, 0.90062112f, -0.20659795f, 0.27995440f, 0.32864550f, 0.49010456f, 0.06502116f, 0.49633679f, 0.44805136f, 0.83104730f, 0.06006010f, 0.08019450f, 0.93961835f, -0.10609409f, -0.69005126f};

static const uint8_t EWC_SUB_W2[16][4] = {
    {179, 60, 254, 63},
    {179, 60, 254, 63},
    {76, 227, 161, 208},
    {179, 56, 254, 63},
    {76, 227, 161, 208},
    {179, 56, 254, 127},
    {179, 60, 254, 127},
    {204, 67, 10, 154},
    {1, 0, 240, 1},
    {76, 195, 129, 192},
    {179, 60, 254, 63},
    {179, 120, 86, 100},
    {210, 128, 64, 2},
    {76, 227, 161, 192},
    {179, 124, 254, 63},
    {179, 60, 254, 63}
};
static const float EWC_SUB_SCALE_W2[16] = {0.31392089f, 0.33266604f, 0.57589489f, 0.40531266f, 0.43223828f, 0.32448384f, 0.31608185f, 0.18737444f, 0.11706068f, 0.50090754f, 0.30418724f, 0.26723540f, 0.11807495f, 0.51279581f, 0.27059796f, 0.30601078f};
static const float EWC_SUB_B2[16] = {0.22713055f, 0.08053129f, 0.46538749f, 0.11555725f, 0.28676555f, 0.22084434f, 0.14414096f, 0.03204829f, 0.00265113f, 0.35588458f, 0.27791813f, 0.12935950f, -0.01419784f, 0.42888999f, 0.12636815f, 0.14583232f};

static const uint8_t EWC_SUB_W3[2][2] = {
    {107, 204},
    {148, 34}
};
static const float EWC_SUB_SCALE_W3[2] = {0.49248511f, 0.47791645f};
static const float EWC_SUB_B3[2] = {0.07137091f, -0.18078120f};

static const float EWC_SUB_SCALE_ACT_IN = 0.03031819f;
static const float EWC_SUB_SCALE_ACT_H1 = 0.02669838f;
static const float EWC_SUB_SCALE_ACT_H2 = 0.12971342f;

static const float EWC_SUB_ACT_MAX[3] = {3.85041070f, 3.39069438f, 16.47360420f};

#endif /* EWC_HEAD_SUBINT8_WEIGHTS_H */
