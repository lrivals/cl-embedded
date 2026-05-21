"""
send_test_frame.py — Envoi d'une trame de test UART vers Renode via socket TCP.

Protocole binaire (little-endian), identique a sensor_sim.py :
    Envoi   : [MAGIC 0xABCD:2B][N:1B][features:f32*N][label:1B][CRC8:1B]
    Reponse : [pred:u8][confidence:f32][latency_us:u32] = 9 B
    Puis (DEBUG_PRINTF=1) : texte "score=X.XXXX pred=N lat=N us\\r\\n"

Le firmware envoie d'abord hw_info_print() au boot (texte avant d'entrer
dans la boucle pipeline). Ce script vide ce buffer avant d'envoyer la trame.

Usage :
    python3 firmware/renode/send_test_frame.py --port 3456
    python3 firmware/renode/send_test_frame.py --port 3456 --features 0.1 0.2 0.3 0.4 0.5 --label 0
"""

import argparse
import socket
import struct
import sys


def crc8(data: bytes) -> int:
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if (crc & 0x80) else (crc << 1) & 0xFF
    return crc


def build_frame(features: list[float], label: int) -> bytes:
    n = len(features)
    payload = struct.pack("<HB", 0xABCD, n)
    payload += struct.pack(f"<{n}f", *features)
    payload += struct.pack("<B", label)
    payload += struct.pack("<B", crc8(payload))
    return payload


def flush_boot_messages(s: socket.socket, drain_timeout: float = 0.5) -> bytes:
    """Vide le buffer socket des messages hw_info_print() emis au boot."""
    s.settimeout(drain_timeout)
    drained = b""
    try:
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            drained += chunk
    except (socket.timeout, BlockingIOError):
        pass
    return drained


def recv_exact(s: socket.socket, n: int, timeout: float = 5.0) -> bytes | None:
    """Lit exactement n octets avec timeout."""
    s.settimeout(timeout)
    data = b""
    try:
        while len(data) < n:
            chunk = s.recv(n - len(data))
            if not chunk:
                break
            data += chunk
    except (socket.timeout, OSError):
        pass
    return data if len(data) == n else None


def recv_line(s: socket.socket, timeout: float = 3.0) -> str:
    """Lit une ligne texte terminee par \\n (DEBUG_PRINTF output)."""
    s.settimeout(timeout)
    line = b""
    try:
        while b"\n" not in line:
            ch = s.recv(1)
            if not ch:
                break
            line += ch
    except (socket.timeout, OSError):
        pass
    return line.decode("ascii", errors="replace").strip()


def send_and_receive(
    host: str,
    port: int,
    frame: bytes,
    connect_timeout: float = 5.0,
    response_timeout: float = 10.0,
) -> tuple[int, float, int, str] | None:
    """
    Returns (pred, confidence, lat_us, debug_line) ou None si echec.
    debug_line contient le texte "score=X.XXXX pred=N lat=N us" (DEBUG_PRINTF=1).
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(connect_timeout)
        s.connect((host, port))
    except (ConnectionRefusedError, TimeoutError, OSError) as e:
        print(f"[WARN] Socket {host}:{port} — {e}", file=sys.stderr)
        return None

    with s:
        # Vider le buffer boot (hw_info_print texte emis avant pipeline_run)
        boot = flush_boot_messages(s)
        if boot:
            print(f"[INFO] Buffer boot vide ({len(boot)} octets) : {boot[:80]!r}...")

        # Envoyer la trame de test
        s.settimeout(response_timeout)
        s.sendall(frame)

        # Lire reponse binaire : [pred:u8][conf:f32][lat_us:u32] = 9 B
        raw = recv_exact(s, 9, timeout=response_timeout)
        if raw is None or len(raw) < 9:
            print(f"[WARN] Reponse binaire incomplete ({len(raw) if raw else 0}/9 B)", file=sys.stderr)
            return None

        pred, conf, lat_us = struct.unpack("<BfI", raw)

        # Lire la ligne DEBUG_PRINTF : "score=X.XXXX pred=N lat=N us\r\n"
        debug_line = recv_line(s, timeout=3.0)

    return pred, conf, lat_us, debug_line


def main() -> None:
    parser = argparse.ArgumentParser(description="Envoi trame test UART Renode")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=3456)
    parser.add_argument(
        "--features",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.4, 0.5],
        help="Valeurs FP32 des features (5 par defaut — dimension MAHA_DIM=5)",
    )
    parser.add_argument("--label", type=int, default=0)
    parser.add_argument("--connect-timeout", type=float, default=5.0)
    parser.add_argument("--response-timeout", type=float, default=10.0)
    args = parser.parse_args()

    frame = build_frame(args.features, args.label)
    print(f"Trame : {len(frame)} octets → {args.host}:{args.port}")
    print(f"  features={args.features} label={args.label}")

    result = send_and_receive(
        args.host, args.port, frame,
        connect_timeout=args.connect_timeout,
        response_timeout=args.response_timeout,
    )
    if result is None:
        print("FAIL: aucune reponse du firmware", file=sys.stderr)
        sys.exit(1)

    pred, conf, lat_us, debug_line = result
    print(f"Reponse binaire : pred={pred} conf={conf:.4f} lat_us={lat_us}")
    if debug_line:
        print(f"DEBUG_PRINTF    : {debug_line}")

    # Verifier la coherence
    if "score=" in debug_line and "pred=" in debug_line:
        print("PASS: score= et pred= visibles dans la sortie DEBUG_PRINTF")
    else:
        print("[WARN] DEBUG_PRINTF incomplet ou absent")


if __name__ == "__main__":
    main()
