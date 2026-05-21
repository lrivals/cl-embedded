#!/bin/bash
# run_mahalanobis_sim.sh — Validation Mahalanobis pipeline via simulation Renode
#
# Lance stm32f4_blink.elf dans Renode (headless), envoie une trame UART de test,
# et verifie que le score Mahalanobis est visible dans la reponse du firmware.
#
# Prerequis :
#   - renode >= 1.14 installe et dans PATH
#   - arm-none-eabi-gcc installe (pour le build si ELF absent)
#   - Python 3.10+
#
# Usage :
#   bash firmware/renode/run_mahalanobis_sim.sh
#   bash firmware/renode/run_mahalanobis_sim.sh --skip-build

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
FIRMWARE_DIR="$REPO_ROOT/firmware/stm32f4_blink"
ELF="$FIRMWARE_DIR/build/stm32f4_blink.elf"
RESC="$REPO_ROOT/firmware/renode/nucleo_f439zi.resc"
SEND_FRAME="$REPO_ROOT/firmware/renode/send_test_frame.py"
RENODE_LOG=$(mktemp /tmp/renode_mahalanobis.XXXXXX)
SIM_OUT=$(mktemp /tmp/sim_output.XXXXXX)
SKIP_BUILD=0
RENODE_PID=""

for arg in "$@"; do
    case $arg in
        --skip-build) SKIP_BUILD=1 ;;
        *) echo "Option inconnue : $arg"; exit 1 ;;
    esac
done

cleanup() {
    if [ -n "$RENODE_PID" ] && kill -0 "$RENODE_PID" 2>/dev/null; then
        kill "$RENODE_PID" 2>/dev/null || true
    fi
    rm -f "$RENODE_LOG" "$SIM_OUT"
}
trap cleanup EXIT

# ── Verifier que Renode est installe ───────────────────────────────────────
if ! command -v renode &>/dev/null; then
    echo "ERROR: renode introuvable dans PATH."
    echo "  Installation : sudo dpkg -i renode_*.deb (voir docs/embedded_renode_guide.md)"
    exit 1
fi
RENODE_VER=$(renode --version 2>&1 | head -1)
echo "Renode : $RENODE_VER"

# ── Build firmware si absent ────────────────────────────────────────────────
if [ "$SKIP_BUILD" -eq 0 ] && [ ! -f "$ELF" ]; then
    echo "Build firmware ARM (make -C firmware/stm32f4_blink/ all)..."
    make -C "$FIRMWARE_DIR" all
fi

[ -f "$ELF" ] || { echo "ERROR: $ELF introuvable."; exit 1; }
echo "ELF : $ELF"

# ── Lancer Renode en mode headless ─────────────────────────────────────────
echo "Demarrage Renode (headless, socket USART3 sur port 3456)..."
timeout 60 renode --console --disable-xwt "$RESC" >"$RENODE_LOG" 2>&1 &
RENODE_PID=$!

# ── Attendre que le firmware soit pret ─────────────────────────────────────
# hw_info_print() + pipeline_init() ~ 300ms reel; Renode est ~10-100x plus lent.
# Le socket ServerSocketTerminal est ouvert avant que start soit execute.
echo "Attente initialisation firmware (5s)..."
sleep 5

if ! kill -0 "$RENODE_PID" 2>/dev/null; then
    echo "FAIL: Renode s'est termine prematurément."
    cat "$RENODE_LOG"
    exit 1
fi

# ── Envoyer une trame de test et lire la reponse firmware ──────────────────
echo "Envoi trame test UART (features=[0.1..0.5], label=0)..."
python3 "$SEND_FRAME" \
    --port 3456 \
    --features 0.1 0.2 0.3 0.4 0.5 \
    --label 0 \
    --connect-timeout 5.0 \
    --response-timeout 15.0 \
    | tee "$SIM_OUT" || true

# ── Assertions ─────────────────────────────────────────────────────────────
echo ""
echo "--- Assertions ---"
PASS=0

if grep -q "pred=" "$SIM_OUT"; then
    PRED_VAL=$(grep "Reponse binaire" "$SIM_OUT" | grep -oP "pred=\K[0-9]+")
    echo "PASS: reponse binaire recue (pred=$PRED_VAL)"
    PASS=$((PASS + 1))
else
    echo "FAIL: aucune reponse binaire du firmware (pred absent)"
fi

if grep -q "score=" "$SIM_OUT"; then
    SCORE_VAL=$(grep "DEBUG_PRINTF" "$SIM_OUT" | grep -oP "score=\K[0-9.]+")
    echo "PASS: score Mahalanobis visible (score=$SCORE_VAL)"
    PASS=$((PASS + 1))
else
    echo "FAIL: 'score=' absent — verifier DEBUG_PRINTF=1 dans le Makefile"
fi

echo ""
echo "Score: $PASS/2 assertions passees"

if [ "$PASS" -eq 2 ]; then
    echo "PASS: simulation Mahalanobis validee"
    exit 0
else
    echo "FAIL: simulation incomplete"
    echo ""
    echo "--- Logs Renode (dernières 20 lignes) ---"
    tail -20 "$RENODE_LOG"
    exit 1
fi
