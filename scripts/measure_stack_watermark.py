#!/usr/bin/env python3
"""measure_stack_watermark.py — Mesure du pic de RAM réel (stack high-water mark).

Le firmware NUCLEO-F439ZI rapporte `.bss` (= `_ebss - _sbss`) comme empreinte RAM,
mais ce chiffre **exclut la pile**. Le chemin HDC alloue p.ex. `float hv[HDC_DIM]`
= 4 Ko sur la pile : le pic de RAM réel vaut donc

    ram_peak = .data + .bss + pic_de_pile

Le startup (`startup_stm32f439xx.s`) peint la zone libre `[_ebss, _estack)` au boot
avec un canary POSITION-DÉPENDANT : chaque mot reçoit sa propre adresse (`str r2,[r2]`).
Après exécution d'une charge représentative (stream multi-modèle, pire cas HDC), ce
script lit la RAM via OpenOCD et trouve le mot le plus bas dont la valeur ≠ son adresse
→ profondeur de pile maximale réellement atteinte. Le motif position-dépendant élimine
le faux négatif d'un buffer de pile rempli d'une constante répétée (contrairement à une
sentinelle constante, qui pouvait être « imitée » par les données).

Procédure (zéro changement du protocole UART) :
  1. flasher + faire tourner le firmware (`make flash`) ;
  2. streamer une charge représentative (`sensor_stream.py`, idéalement les modes
     qui exercent le HDC) — laisser la carte tourner ;
  3. lancer ce script : il se connecte au serveur OpenOCD (Tcl RPC, port 6666),
     halt le cœur, scanne la sentinelle, calcule le pic, puis relance le cœur.

Les adresses `_sbss/_ebss/_sdata/_edata/_estack` sont lues depuis l'ELF (aucune
valeur en dur). Lance OpenOCD à part :

    openocd -f interface/stlink.cfg -f target/stm32f4x.cfg

Exemple :
    python scripts/measure_stack_watermark.py \
        --elf firmware/stm32f4_blink/build/stm32f4_blink.elf

Caveat : le seul faux négatif résiduel est un mot-frontière valant par hasard sa propre
adresse (~1/2³², négligeable) — le canary position-dépendant ayant déjà écarté le cas
d'un buffer rempli d'une constante répétée.
"""
from __future__ import annotations

import argparse
import socket
import subprocess
import sys
from pathlib import Path

RAM_TOTAL_BYTES = 256 * 1024  # NUCLEO-F439ZI : 192 Ko SRAM + 64 Ko CCM (budget Gap 2)
TCL_RPC_DEFAULT_PORT = 6666
TCL_TERMINATOR = b"\x1a"


# ── Lecture des symboles linker depuis l'ELF ──────────────────────────────────
def read_symbols(elf: Path, names: list[str]) -> dict[str, int]:
    """Retourne {nom: adresse} via `arm-none-eabi-nm`."""
    out = subprocess.check_output(
        ["arm-none-eabi-nm", str(elf)], text=True
    )
    table: dict[str, int] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[2] in names:
            table[parts[2]] = int(parts[0], 16)
    missing = [n for n in names if n not in table]
    if missing:
        raise SystemExit(f"Symboles introuvables dans {elf} : {missing}")
    return table


# ── Client OpenOCD Tcl RPC ────────────────────────────────────────────────────
class OpenOCD:
    """Client minimal pour le serveur Tcl RPC d'OpenOCD (port 6666)."""

    def __init__(self, host: str = "127.0.0.1", port: int = TCL_RPC_DEFAULT_PORT):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(10.0)
        self.sock.connect((host, port))

    def cmd(self, command: str) -> str:
        self.sock.sendall(command.encode("utf-8") + TCL_TERMINATOR)
        buf = bytearray()
        while True:
            chunk = self.sock.recv(4096)
            if not chunk:
                break
            buf += chunk
            if buf.endswith(TCL_TERMINATOR):
                buf = buf[:-1]
                break
        return buf.decode("utf-8", errors="replace")

    def read_words(self, addr: int, count: int) -> list[int]:
        """Lit `count` mots 32 bits à partir de `addr` via `read_memory`."""
        # read_memory <addr> <width> <count> renvoie une liste d'entiers Tcl
        out = self.cmd(f"read_memory 0x{addr:08x} 32 {count}")
        return [int(tok, 0) for tok in out.split()]

    def close(self) -> None:
        self.sock.close()


# ── Scan high-water ───────────────────────────────────────────────────────────
def scan_stack_peak(ocd: OpenOCD, ebss: int, estack: int,
                    chunk_words: int = 1024) -> int:
    """Scanne [ebss, estack) de bas en haut ; retourne le pic de pile en octets.

    La pile croît vers le bas depuis `estack`. Canary POSITION-DÉPENDANT : le
    startup a peint chaque mot avec sa propre adresse. On cherche le premier mot
    dont la valeur ≠ son adresse (en partant de `ebss`) ; tout ce qui est
    au-dessus a été touché par la pile.
    """
    total_words = (estack - ebss) // 4
    addr = ebss
    scanned = 0
    while scanned < total_words:
        n = min(chunk_words, total_words - scanned)
        words = ocd.read_words(addr, n)
        for i, w in enumerate(words):
            word_addr = addr + i * 4
            if (w & 0xFFFFFFFF) != (word_addr & 0xFFFFFFFF):
                return estack - word_addr
        addr += n * 4
        scanned += n
    return 0  # tout intact → pile jamais utilisée (improbable)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    default_elf = (Path(__file__).resolve().parent.parent
                   / "firmware/stm32f4_blink/build/stm32f4_blink.elf")
    ap.add_argument("--elf", type=Path, default=default_elf,
                    help="ELF du firmware (pour lire les symboles linker)")
    ap.add_argument("--host", default="127.0.0.1", help="hôte OpenOCD Tcl RPC")
    ap.add_argument("--port", type=int, default=TCL_RPC_DEFAULT_PORT,
                    help="port OpenOCD Tcl RPC (défaut 6666)")
    ap.add_argument("--no-resume", action="store_true",
                    help="ne pas relancer le cœur après la mesure")
    ap.add_argument("--json", type=Path, default=None,
                    help="écrire le résultat (JSON) dans ce fichier")
    ap.add_argument("--label", default=None,
                    help="étiquette du modèle/scénario (stockée dans le JSON)")
    args = ap.parse_args()

    if not args.elf.exists():
        raise SystemExit(f"ELF introuvable : {args.elf} (lancer `make all`)")

    syms = read_symbols(args.elf,
                        ["_sdata", "_edata", "_sbss", "_ebss", "_estack"])
    data_bytes = syms["_edata"] - syms["_sdata"]
    bss_bytes = syms["_ebss"] - syms["_sbss"]

    print(f"ELF        : {args.elf}")
    print(f"_sdata/_edata : 0x{syms['_sdata']:08x} / 0x{syms['_edata']:08x}  "
          f"(.data = {data_bytes} B)")
    print(f"_sbss/_ebss   : 0x{syms['_sbss']:08x} / 0x{syms['_ebss']:08x}  "
          f"(.bss  = {bss_bytes} B)")
    print(f"_estack       : 0x{syms['_estack']:08x}  "
          f"(zone pile libre = {syms['_estack'] - syms['_ebss']} B)")

    try:
        ocd = OpenOCD(args.host, args.port)
    except OSError as exc:
        raise SystemExit(
            f"Connexion OpenOCD échouée ({args.host}:{args.port}) : {exc}\n"
            "Lance d'abord : openocd -f interface/stlink.cfg -f target/stm32f4x.cfg"
        )

    try:
        ocd.cmd("halt")
        stack_peak = scan_stack_peak(ocd, syms["_ebss"], syms["_estack"])
    finally:
        if not args.no_resume:
            ocd.cmd("resume")
        ocd.close()

    ram_peak = data_bytes + bss_bytes + stack_peak
    print("\n── Pic de RAM mesuré ─────────────────────────────────────────")
    print(f"  .data        : {data_bytes:>8} B")
    print(f"  .bss         : {bss_bytes:>8} B  (chiffre habituellement rapporté)")
    print(f"  pic de pile  : {stack_peak:>8} B  (mesuré, high-water mark)")
    print(f"  ─────────────────────────────")
    print(f"  RAM pic total: {ram_peak:>8} B  "
          f"({100.0 * ram_peak / RAM_TOTAL_BYTES:.1f} % de 256 Ko)")
    if stack_peak == 0:
        print("  ⚠ pic de pile = 0 : la carte a-t-elle exécuté une charge avant "
              "la mesure ? (stream multi-modèle requis)")

    if args.json is not None:
        import json
        record = {
            "label": args.label,
            "elf": str(args.elf),
            "data_bytes": data_bytes,
            "bss_bytes": bss_bytes,
            "stack_peak_bytes": stack_peak,
            "ram_peak_bytes": ram_peak,
            "ram_total_bytes": RAM_TOTAL_BYTES,
            "ram_peak_pct": round(100.0 * ram_peak / RAM_TOTAL_BYTES, 2),
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(record, indent=2))
        print(f"\n→ résultat écrit : {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
