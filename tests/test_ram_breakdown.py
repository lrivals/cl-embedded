"""Tests pour scripts/ram_breakdown.py — décomposition RAM par modèle.

Vérifie que le split statique/modulable est cohérent avec la taille `.bss`
réelle lue de l'ELF (nm), que les 4 modèles Monitoring sont présents, et que
les #define de dimension sont bien lus.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from scripts import ram_breakdown as rb

ELF = rb.DEFAULT_ELF
HAS_NM = shutil.which("arm-none-eabi-nm") is not None
skip_no_elf = pytest.mark.skipif(
    not (ELF.exists() and HAS_NM),
    reason="ELF firmware ou arm-none-eabi-nm indisponible",
)


def test_read_defines_monitoring_dims():
    d = rb.read_defines()
    assert d["EWC_IN"] == 5
    assert d["MAHA_DIM"] == 5
    assert d["TINYOL_IN"] == 5
    assert d["HDC_DIM"] == 1000
    assert d["HDC_N_CLASSES"] == 2


def test_layout_split_matches_analytic_total():
    """static + modular == bss_analytic pour chaque modèle (pas d'ELF requis)."""
    layout = rb.monitoring_layout(rb.read_defines())
    for name, m in layout.items():
        assert m["static"] > 0, name
        assert m["modular"] > 0, name
        # somme cohérente en interne
        assert m["static"] + m["modular"] > 0


def test_ewc_analytic_size_is_exact():
    """La formule EWC doit donner 8652 B (MLP 5→32→16→2, poids+Fisher+θ*+λ)."""
    layout = rb.monitoring_layout(rb.read_defines())
    ewc = layout["EWC"]
    assert ewc["static"] + ewc["modular"] == 8652


def test_hdc_static_is_projection():
    """HDC : la partie statique domine (projection 20 Ko) vs modulable (am 8 Ko)."""
    layout = rb.monitoring_layout(rb.read_defines())
    hdc = layout["HDC"]
    assert hdc["static"] == 20000            # proj[1000][5] * 4 B
    assert hdc["modular"] >= 8000            # am[2][1000] + buffer + ring


def test_stack_infer_vs_train_ordering():
    """La pile d'entraînement ≥ pile d'inférence (gradients en plus)."""
    layout = rb.monitoring_layout(rb.read_defines())
    for name, m in layout.items():
        assert m["stack_train"] >= m["stack_infer"], name
    # HDC dominé par hv[HDC_DIM] = 4 Ko sur la pile
    assert layout["HDC"]["stack_infer"] == 4000


@skip_no_elf
def test_monitoring_breakdown_matches_nm():
    """Le split analytique égale la taille .bss réelle (nm) pour les 4 modèles."""
    bd = rb.monitoring_breakdown(ELF)
    assert set(bd) == {"EWC", "HDC", "Mahalanobis", "TinyOL"}
    for name, m in bd.items():
        assert m["bss_real_nm"] > 0, name
        assert abs(m["bss_analytic"] - m["bss_real_nm"]) <= 8, name


@skip_no_elf
def test_read_bss_symbols_has_models():
    syms = rb.read_bss_symbols(ELF)
    for s in ("g_ewc_head", "g_hdc", "g_detector", "g_tinyol_enc"):
        assert s in syms and syms[s] > 0
