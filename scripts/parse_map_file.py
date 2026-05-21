"""
parse_map_file.py — Parse le fichier .map linker pour valider l'empreinte RAM.

Extrait les tailles des sections .bss, .data, .text depuis le fichier .map
généré par arm-none-eabi-ld avec -Wl,-Map,firmware.map.
Vérifie que SRAM totale < contrainte (64 Ko pour STM32N6, 192 Ko pour NUCLEO).

Usage :
    python scripts/parse_map_file.py firmware/stm32f4_blink/build/firmware.map
    python scripts/parse_map_file.py firmware.map --ram-limit 65536 --save results.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


# RAM limits
RAM_LIMIT_STM32N6_BYTES  = 64 * 1024   # 64 Ko — cible finale
RAM_LIMIT_NUCLEO_BYTES   = 192 * 1024  # 192 Ko — board intermédiaire


def parse_map_file(map_path: Path) -> dict:
    """
    Parse un fichier .map linker GNU et extrait les tailles de section.

    Returns
    -------
    dict avec les clés : text_bytes, data_bytes, bss_bytes, total_ram_bytes,
                         total_flash_bytes, symbols (liste de dicts)
    """
    content = map_path.read_text(errors="replace")

    sections = {}
    # Recherche des lignes de section memory map : ".bss   0x...  0x... size"
    section_re = re.compile(
        r"^(\.text|\.data|\.bss|\.rodata|\.ccmram)\s+"
        r"0x([0-9a-fA-F]+)\s+0x([0-9a-fA-F]+)",
        re.MULTILINE,
    )
    for m in section_re.finditer(content):
        name = m.group(1)
        size = int(m.group(3), 16)
        if name not in sections or sections[name] < size:
            sections[name] = size

    text_bytes  = sections.get(".text",   0) + sections.get(".rodata", 0)
    data_bytes  = sections.get(".data",   0)
    bss_bytes   = sections.get(".bss",    0)
    ccmram_bytes = sections.get(".ccmram", 0)

    # RAM = .data + .bss (+ .ccmram si mappé en SRAM principale)
    total_ram_bytes   = data_bytes + bss_bytes
    total_flash_bytes = text_bytes + data_bytes  # .data initialisé en Flash

    # Symboles notables (modèles + pipeline)
    notable = ["g_detector", "g_profiling", "g_ewc_head", "g_tinyol_enc"]
    symbol_sizes = {}
    sym_re = re.compile(
        r"0x[0-9a-fA-F]+\s+(0x[0-9a-fA-F]+)\s+(\S+)",
        re.MULTILINE,
    )
    for m in sym_re.finditer(content):
        size_str, name = m.group(1), m.group(2)
        if any(n in name for n in notable):
            symbol_sizes[name] = int(size_str, 16)

    return {
        "text_bytes":        text_bytes,
        "data_bytes":        data_bytes,
        "bss_bytes":         bss_bytes,
        "ccmram_bytes":      ccmram_bytes,
        "total_ram_bytes":   total_ram_bytes,
        "total_flash_bytes": total_flash_bytes,
        "symbols":           symbol_sizes,
    }


def check_constraints(sizes: dict, ram_limit: int) -> list[str]:
    violations = []
    ram = sizes["total_ram_bytes"]
    if ram > ram_limit:
        violations.append(
            f"RAM {ram} B > limite {ram_limit} B "
            f"({ram / 1024:.1f} Ko > {ram_limit / 1024:.0f} Ko)"
        )
    return violations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse fichier .map linker → validation contrainte RAM (Gap 2)")
    parser.add_argument("map_file", help="Chemin vers le fichier .map (ex: firmware.map)")
    parser.add_argument("--ram-limit", type=int, default=RAM_LIMIT_NUCLEO_BYTES,
                        help=f"Limite RAM en octets (défaut: {RAM_LIMIT_NUCLEO_BYTES} pour NUCLEO-F439ZI)")
    parser.add_argument("--stm32n6", action="store_true",
                        help="Applique la limite STM32N6 stricte (64 Ko)")
    parser.add_argument("--save", help="Chemin JSON pour sauvegarder les résultats")
    args = parser.parse_args()

    if args.stm32n6:
        ram_limit = RAM_LIMIT_STM32N6_BYTES
    else:
        ram_limit = args.ram_limit

    map_path = Path(args.map_file)
    if not map_path.exists():
        print(f"Erreur : fichier non trouvé : {map_path}")
        raise SystemExit(1)

    sizes = parse_map_file(map_path)
    violations = check_constraints(sizes, ram_limit)

    print(f"=== Analyse : {map_path.name} ===")
    print(f"  .text + .rodata (Flash) : {sizes['text_bytes']:>8} B  ({sizes['text_bytes']/1024:.1f} Ko)")
    print(f"  .data           (Flash) : {sizes['data_bytes']:>8} B")
    print(f"  .data + .bss    (RAM)   : {sizes['total_ram_bytes']:>8} B  ({sizes['total_ram_bytes']/1024:.1f} Ko)")
    if sizes["ccmram_bytes"]:
        print(f"  .ccmram         (CCM)   : {sizes['ccmram_bytes']:>8} B")
    print(f"  Limite RAM              : {ram_limit:>8} B  ({ram_limit/1024:.0f} Ko)")
    print(f"  Marge                   : {ram_limit - sizes['total_ram_bytes']:>8} B")

    if sizes["symbols"]:
        print("\n  Symboles notables :")
        for sym, sz in sizes["symbols"].items():
            print(f"    {sym}: {sz} B")

    if violations:
        print("\n  ❌ VIOLATIONS GAP 2 :")
        for v in violations:
            print(f"    - {v}")
    else:
        print(f"\n  ✅ Gap 2 RAM : {sizes['total_ram_bytes']} B < {ram_limit} B ({ram_limit/1024:.0f} Ko) — OK")

    if args.save:
        out = {**sizes, "ram_limit_bytes": ram_limit,
               "gap2_compliant": len(violations) == 0, "violations": violations}
        Path(args.save).write_text(json.dumps(out, indent=2))
        print(f"\nSauvegardé : {args.save}")


if __name__ == "__main__":
    main()
