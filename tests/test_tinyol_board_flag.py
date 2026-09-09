"""Garde-fous du portage board TinyOL (S5201).

Contexte : ``sensor_stream.py`` ne posait aucun flag pour ``--model tinyol``
(``model_flags = 0``), ce qui est le chemin d'inférence PAR DÉFAUT du firmware,
c.-à-d. Mahalanobis (``pipeline.c``, branche ``else``). Toutes les cellules carte
« TinyOL » streamées ainsi ont donc mesuré Mahalanobis — les JSON S35 des deux
modèles étaient numériquement identiques.

En parallèle, les poids TinyOL n'étaient chargés qu'à ``TINYOL_IN == 5`` et
n'étaient jamais exportés par les drivers : même avec le bon flag, l'auto-encodeur
aurait tourné à poids nuls dès qu'une condition changeait la dimension.

Ces tests verrouillent les deux moitiés du correctif.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SENSOR_STREAM = ROOT / "scripts" / "sensor_stream.py"
EXPORT_TINYOL = ROOT / "scripts" / "export_weights_tinyol.py"
PIPELINE_C = ROOT / "firmware" / "stm32f4_blink" / "src" / "pipeline.c"
TINYOL_C = ROOT / "firmware" / "stm32f4_blink" / "src" / "tinyol.c"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── 1. Le flag UART ─────────────────────────────────────────────────────────

def _model_flag_branches() -> dict[str, str]:
    """Extrait la table ``args.model`` → constante de flag depuis le source.

    Analyse l'AST plutôt que d'exécuter ``main()`` (qui ouvre le port série) :
    la chaîne de ``elif`` est le seul endroit où le modèle est traduit en flag.
    """
    tree = ast.parse(SENSOR_STREAM.read_text(encoding="utf-8"))
    branches: dict[str, str] = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (isinstance(test, ast.Compare)
                and isinstance(test.ops[0], ast.Eq)
                and isinstance(test.left, ast.Attribute)
                and test.left.attr == "model"
                and isinstance(test.comparators[0], ast.Constant)):
            for stmt in node.body:
                if (isinstance(stmt, ast.Assign)
                        and isinstance(stmt.targets[0], ast.Name)
                        and stmt.targets[0].id == "model_flags"
                        and isinstance(stmt.value, ast.Name)):
                    branches[test.comparators[0].value] = stmt.value.id
    return branches


def test_tinyol_maps_to_dedicated_flag():
    """--model tinyol doit poser PROTO_FLAG_TINYOL_MODE (0x80)."""
    ss = _load(SENSOR_STREAM, "sensor_stream_flagtest")
    branches = _model_flag_branches()
    assert branches.get("tinyol") == "FRAME_FLAGS_TINYOL_MODE"
    assert ss.FRAME_FLAGS_TINYOL_MODE == 0x80


def test_mahalanobis_is_the_only_flagless_model():
    """Tout modèle sans flag exécute silencieusement Mahalanobis : un seul admis.

    C'est la garde anti-régression du bug S5201. Si un modèle est ajouté à
    ``--model`` sans branche de flag, ce test tombe.
    """
    src = SENSOR_STREAM.read_text(encoding="utf-8")
    choices = re.search(r'"--model",\s*choices=\[(.*?)\]', src, re.DOTALL)
    assert choices, "liste choices de --model introuvable"
    declared = set(re.findall(r'"([^"]+)"', choices.group(1)))
    flagged = set(_model_flag_branches())
    assert declared - flagged == {"mahalanobis"}


def test_no_flag_means_mahalanobis_route_in_firmware():
    """Le firmware confirme la sémantique : flags=0 → branche else = Mahalanobis."""
    src = PIPELINE_C.read_text(encoding="utf-8")
    assert "PROTO_FLAG_TINYOL_MODE" in src
    tinyol_at = src.index("g_recv_flags & PROTO_FLAG_TINYOL_MODE")
    maha_at = src.index("Chemin Mahalanobis (comportement historique)")
    assert tinyol_at < maha_at, "la route TinyOL doit précéder le fallback Mahalanobis"


# ── 2. L'export des poids à dim k ───────────────────────────────────────────

@pytest.fixture(scope="module")
def ewt():
    return _load(EXPORT_TINYOL, "export_weights_tinyol_test")


@pytest.mark.parametrize("k", [3, 4, 5, 9])
def test_export_section_matches_requested_dim(ewt, k):
    """Les tableaux générés et TINYOL_NATIVE_DIM suivent la dim du modèle."""
    model = ewt.TinyOLBoard(dim=k)
    section = ewt.build_tinyol_section(model, threshold=0.01)
    assert f"#define TINYOL_NATIVE_DIM {k}" in section
    assert f"TINYOL_W_ENC1[32][{k}]" in section
    assert f"TINYOL_W_DEC2[{k}][32]" in section
    assert f"TINYOL_B_DEC2[{k}]" in section
    # dims internes invariantes (architecture board tinyol.h)
    assert "TINYOL_W_ENC2[16][32]" in section
    assert "TINYOL_W_DEC1[32][16]" in section


def test_section_regex_replaces_the_define_too(ewt, tmp_path):
    """Un ré-export ne doit pas laisser deux TINYOL_NATIVE_DIM dans le header."""
    header = tmp_path / "model_weights.h"
    header.write_text("#pragma once\n", encoding="utf-8")
    for k in (5, 4, 7):
        ewt.update_model_weights_h(
            ewt.build_tinyol_section(ewt.TinyOLBoard(dim=k), threshold=0.01), header)
        content = header.read_text(encoding="utf-8")
        assert content.count("#define TINYOL_NATIVE_DIM") == 1
        assert f"#define TINYOL_NATIVE_DIM {k}" in content


def test_dim_of_state_dict_roundtrip(ewt):
    import torch

    model = ewt.TinyOLBoard(dim=7)
    assert ewt.dim_of_state_dict(model.state_dict()) == 7
    reloaded = ewt.TinyOLBoard(dim=ewt.dim_of_state_dict(model.state_dict()))
    reloaded.load_state_dict(model.state_dict())  # ne lève pas
    assert isinstance(reloaded, torch.nn.Module)


def test_fit_infers_dim_from_data(ewt):
    """fit_tinyol_board ne câble aucune dim : elle vient de X."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(64, 6)).astype(np.float32)
    model = ewt.fit_tinyol_board(X, epochs=2)
    assert model.dim == 6
    assert model._calibrated_threshold > 0


# ── 3. La garde firmware ────────────────────────────────────────────────────

def test_firmware_guards_use_tinyol_native_dim():
    """Les deux copies de poids sont gardées par TINYOL_NATIVE_DIM, pas par 5."""
    for path in (PIPELINE_C, TINYOL_C):
        src = path.read_text(encoding="utf-8")
        assert "#if (TINYOL_IN == TINYOL_NATIVE_DIM)" in src, path.name
        assert "#if (TINYOL_IN == WEIGHTS_NATIVE_DIM)" not in src, path.name


def test_tinyol_route_sized_on_tinyol_in():
    """La reconstruction est dimensionnée sur TINYOL_OUT, pas sur EWC_IN."""
    src = PIPELINE_C.read_text(encoding="utf-8")
    route = src[src.index("Chemin TinyOL autoencoder"):]
    route = route[:route.index("Chemin Mahalanobis")]
    assert "float recon[TINYOL_OUT];" in route
    assert "float recon[EWC_IN];" not in route
    assert "tinyol_reconstruction_error(raw, recon, EWC_IN)" not in route


@pytest.mark.skipif(
    subprocess.run(["which", "arm-none-eabi-gcc"], capture_output=True).returncode != 0,
    reason="toolchain ARM absente")
def test_test_reference_header_is_generated_and_consistent():
    """Le golden de parité C↔Python vit avec les poids (régénéré, jamais recopié)."""
    ref = ROOT / "firmware" / "stm32f4_blink" / "tests" / "tinyol_reference.h"
    weights = ROOT / "firmware" / "stm32f4_blink" / "inc" / "model_weights.h"
    assert ref.exists(), "tinyol_reference.h manquant — lancer --emit-test-reference"
    ref_dim = int(re.search(r"#define TINYOL_REF_DIM (\d+)", ref.read_text()).group(1))
    native = int(re.search(r"#define TINYOL_NATIVE_DIM (\d+)", weights.read_text()).group(1))
    assert ref_dim == native
