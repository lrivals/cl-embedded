"""test_s46_quant_moment.py — Tests structure + honnêteté du Sprint 46 (S4607).

Vérifie, sans jamais exiger de chiffres inventés :
  - le schéma JSON des expériences 3-way EWC/TinyOL (``exp_S46_{ewc,tinyol}/``) : 4 moments ;
  - la présence distincte de ``before``/``after``/``both`` (le maillon neuf S4602) ;
  - l'honnêteté N/A de HDC/Mahalanobis (``exp_S46_context/``) : ``moments_3way == "N/A"``,
    ``na_reason`` non vide, **aucune cellule 3-way artificielle** ;
  - le câblage du chemin ``both`` : les poids d'un modèle QAT sont **lus** (fc1/fc2/fc3) sans
    réentraînement, via ``_weights_from_model`` → ``EWCHeadWeights.from_state_dict`` ;
  - la règle « aucun chiffre inventé » : une métrique non calculée vaut ``null``, jamais ``0`` ;
  - le déterminisme (seed 42) du head QAT multiclasse board (S4608).

Skips honnêtes si un artefact manque (comme ``test_s39_quant.py``).

Références : S4602 (harnais), S4603/S4604/S4605 (expériences), S4608 (board QAT multiclasse).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
EWC_DIR = REPO / "experiments" / "exp_S46_ewc"
TINYOL_DIR = REPO / "experiments" / "exp_S46_tinyol"
CONTEXT_DIR = REPO / "experiments" / "exp_S46_context"

MOMENTS = {"fp32", "before", "after", "both"}
THREE_WAY = {"before", "after", "both"}

_needs_ewc = pytest.mark.skipif(
    not EWC_DIR.exists(), reason="exp_S46_ewc absent (lancer run_s46_quant_moment.py)"
)
_needs_tinyol = pytest.mark.skipif(
    not TINYOL_DIR.exists(), reason="exp_S46_tinyol absent (lancer run_s46_quant_moment.py)"
)
_needs_context = pytest.mark.skipif(
    not CONTEXT_DIR.exists(), reason="exp_S46_context absent (run_s46_quant_moment.py --moment context)"
)


def _threeway_jsons(d: Path) -> list[Path]:
    """JSON de résultats 3-way (exclut config_snapshot.yaml et tout non-.json)."""
    return sorted(d.glob("*.json"))


# ── Schéma JSON ──────────────────────────────────────────────────────────────


@_needs_ewc
def test_json_schema_ewc():
    files = _threeway_jsons(EWC_DIR)
    assert files, "aucun JSON EWC"
    for f in files:
        d = json.loads(f.read_text())
        assert d["model"] == "ewc", f"{f.name}: model inattendu"
        assert set(d["moments"]) == MOMENTS, f"{f.name}: moments {set(d['moments'])}"
        for key in ("delta_before_vs_fp32", "delta_after_vs_fp32", "delta_both_vs_fp32"):
            assert key in d, f"{f.name}: {key} manquant"


@_needs_tinyol
def test_json_schema_tinyol():
    files = _threeway_jsons(TINYOL_DIR)
    assert files, "aucun JSON TinyOL"
    for f in files:
        d = json.loads(f.read_text())
        assert d["model"] == "tinyol", f"{f.name}: model inattendu"
        assert set(d["moments"]) == MOMENTS, f"{f.name}: moments {set(d['moments'])}"


@_needs_ewc
def test_three_moments_present():
    """before / after / both présents et distincts (clés) pour EWC."""
    for f in _threeway_jsons(EWC_DIR):
        d = json.loads(f.read_text())
        for m in THREE_WAY:
            assert m in d["moments"], f"{f.name}: moment {m} absent"
        # `both` porte la note « fidèle au déploiement » ; `before` la note « borne haute ».
        assert d["moments"]["both"].get("note"), f"{f.name}: both sans note déploiement"
        assert d["moments"]["before"].get("note"), f"{f.name}: before sans note borne-haute"


# ── Honnêteté N/A (HDC / Mahalanobis) ────────────────────────────────────────


@_needs_context
def test_na_honest_hdc_maha():
    files = _threeway_jsons(CONTEXT_DIR)
    assert files, "aucun JSON contexte"
    seen = set()
    for f in files:
        d = json.loads(f.read_text())
        assert d["moments_3way"] == "N/A", f"{f.name}: moments_3way devrait être N/A"
        assert d.get("na_reason"), f"{f.name}: na_reason vide"
        # Aucune cellule 3-way artificielle pour ces modèles.
        for m in THREE_WAY:
            assert m not in d, f"{f.name}: cellule 3-way artificielle {m}"
        assert "moments" not in d, f"{f.name}: bloc moments 3-way présent (interdit HDC/Maha)"
        seen.add(d["model"])
    assert {"hdc", "mahalanobis"} <= seen, f"modèles contexte manquants : {seen}"


# ── Câblage du chemin `both` (maillon neuf) ──────────────────────────────────


def test_both_path_wiring():
    """`both` LIT les poids QAT (fc1/fc2/fc3) sans réentraîner — via _weights_from_model."""
    torch = pytest.importorskip("torch")
    from scripts.run_s46_quant_moment import _weights_from_model
    from src.models.ewc import EWCMlpInt8Classifier

    qat = EWCMlpInt8Classifier(input_dim=5, hidden_dims=[32, 16])
    # Empreinte des poids AVANT extraction : _weights_from_model ne doit pas les toucher.
    before = {n: p.detach().clone() for n, p in qat.named_parameters()}
    w = _weights_from_model(qat)

    # Les poids extraits correspondent aux couches fc1/fc2/fc3 (pas un réentraînement).
    assert w.w1.shape == (32, 5) and w.w2.shape == (16, 32)
    assert np.allclose(w.w1, qat.fc1.weight.detach().numpy(), atol=1e-6)
    # Le modèle QAT est intact (aucun step d'optimisation).
    for n, p in qat.named_parameters():
        assert torch.equal(p, before[n]), f"{n} modifié par l'extraction"


def test_both_path_multiclass_exportable():
    """Le head QAT multiclasse board (S4608) s'exporte comme un head FP32 (fc3 = n_classes)."""
    pytest.importorskip("torch")
    from src.models.ewc import EWCMlpMulticlassInt8
    from src.utils.int8_c_emulation import EWCHeadWeights

    m = EWCMlpMulticlassInt8(input_dim=5, n_classes=2, hidden_dims=[32, 16])
    w = EWCHeadWeights.from_state_dict(m.state_dict())
    assert w.w1.shape == (32, 5) and w.w2.shape == (16, 32) and w.w3.shape == (2, 16)


# ── Règle « aucun chiffre inventé » ──────────────────────────────────────────


@_needs_ewc
def test_no_invented_numbers():
    """Une métrique de moment est un float réel OU null — jamais 0.0 sentinelle."""
    for f in _threeway_jsons(EWC_DIR):
        d = json.loads(f.read_text())
        for m, cell in d["moments"].items():
            metric = cell.get("metric")
            assert metric is None or isinstance(metric, (int, float)), f"{f.name}:{m}"
            # 0.0 comme placeholder « non mesuré » est interdit (AUROC réel > 0).
            assert metric != 0, f"{f.name}:{m} métrique = 0 (placeholder interdit)"


def test_no_invented_numbers_template():
    """Le squelette du harnais met `null` (pas 0) pour un moment non calculé."""
    pytest.importorskip("torch")
    from scripts.run_s46_quant_moment import _assemble

    out = _assemble("ewc", "monitoring", "cfg.yaml", "auroc", 42, {}, cells={})
    for m in MOMENTS:
        assert out["moments"][m]["metric"] is None, f"{m} devrait être null sans run"
    assert out["delta_both_vs_fp32"] is None
    assert out["gap3_metric_ok_both"] is None


# ── Déterminisme (head QAT multiclasse board, S4608) ─────────────────────────


def test_deterministic_seed_qat_multiclass():
    """Deux entraînements QAT seed 42 (mêmes données) → poids identiques."""
    torch = pytest.importorskip("torch")
    from src.models.ewc import EWCMlpMulticlassInt8
    from src.utils.reproducibility import set_seed

    def _train() -> dict:
        set_seed(42)
        X = torch.randn(256, 5)
        y = (X[:, 0] > 0).long()
        m = EWCMlpMulticlassInt8(input_dim=5, n_classes=2, hidden_dims=[32, 16])
        opt = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9)
        crit = torch.nn.CrossEntropyLoss()
        m.train()
        for _ in range(3):
            opt.zero_grad()
            (crit(m(X), y) + m.ewc_penalty()).backward()
            opt.step()
        return {n: p.detach().clone() for n, p in m.named_parameters()}

    a, b = _train(), _train()
    for n in a:
        assert torch.allclose(a[n], b[n], atol=1e-6), f"{n} non déterministe"
