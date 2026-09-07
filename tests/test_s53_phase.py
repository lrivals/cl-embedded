"""test_s53_phase.py — Profil temporel par phase et µJ par intégration (S5305).

Ce que ces tests protègent, et qui ne se voit PAS sur le banc :

1. **La voie PA8 reste intacte.** `derive_phase_windows` et sa signature sont figées ici :
   la voie A est une fonction AJOUTÉE À CÔTÉ (spec S5305), jamais une réécriture. Les 16
   tests de `test_energy_capture.py` doivent rester verts tels quels.
2. **Le seuil ne se saisit pas à la main.** Il se déduit de la trace (médiane + k·MAD) ;
   un repos bruité ne doit produire aucun créneau.
3. **Le refus honnête.** Si le nombre de créneaux détectés s'écarte de plus de 5 % de
   `cadence × durée`, aucune énergie n'est publiée : `"à mesurer"` + raison chiffrée,
   jamais un zéro ni une division par un dénominateur faux.
4. **La limite 1 bit.** `startup` et `acquisition` ne valent JAMAIS zéro : la voie A
   sépare l'actif de l'inactif, pas les trois sous-phases actives.
5. **Les trois estimateurs ne fusionnent pas.** Le libellé de méthode de l'intégration
   doit rester distinct de celui de la régression de cadence (S5304).

Tout est vérifié sur des traces SYNTHÉTIQUES construites dans le test : aucun chiffre de
mesure n'est inventé, et les tests de schéma des cellules `skip` tant que le banc n'a pas
tourné.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp_S53_phase_profile"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ec = _load("energy_capture", ROOT / "scripts" / "energy_capture.py")
rr = _load("rate_regression", ROOT / "src" / "evaluation" / "rate_regression.py")

FS_HZ = 100_000.0            # mode dynamique de la sonde
DT_S = 1.0 / FS_HZ
I_IDLE_A = 0.045             # niveaux ARBITRAIRES de construction des traces de test,
I_ACTIVE_A = 0.052           # pas des mesures : ils n'apparaissent dans aucun JSON.
VOLTAGE_V = 3.3


def make_trace(n_bursts: int, duration_s: float, burst_us: float,
               noise_a: float = 0.0, seed: int = 42) -> dict[str, np.ndarray]:
    """Trace synthétique : `n_bursts` créneaux régulièrement espacés sur `duration_s`."""
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * FS_HZ))
    current = np.full(n, I_IDLE_A, dtype=np.float64)
    if noise_a:
        current += rng.normal(0.0, noise_a, size=n)
    largeur = max(1, int(round(burst_us * 1e-6 * FS_HZ)))
    for k in range(n_bursts):
        start = int(round((k + 0.25) * n / max(1, n_bursts)))
        current[start:start + largeur] += I_ACTIVE_A - I_IDLE_A
    return {
        "time_s": np.arange(n, dtype=np.float64) * DT_S,
        "current_a": current,
        "voltage_v": np.full(n, VOLTAGE_V, dtype=np.float64),
        "sync": None,
    }


# ── 1. Non-régression : la voie PA8 est intacte ──────────────────────────────

def test_voie_pa8_inchangee():
    """La voie A est une fonction ajoutée, pas une réécriture de la voie PA8."""
    signature = inspect.signature(ec.derive_phase_windows)
    assert list(signature.parameters) == ["trace", "threshold"]
    assert signature.parameters["threshold"].default == 0.5

    trace = {"time_s": np.array([0.0, 1.0, 2.0, 3.0]),
             "sync": np.array([0.0, 1.0, 1.0, 0.0])}
    assert ec.derive_phase_windows(trace) == [
        ("idle", 0.0, 1.0), ("inference", 1.0, 3.0), ("idle", 3.0, 3.0)]

    with pytest.raises(ValueError):
        ec.derive_phase_windows({"time_s": np.array([0.0]), "sync": None})


# ── 2. Détection des créneaux ────────────────────────────────────────────────

def test_creneaux_detectes_bornes_exactes():
    """Créneau parfait : n bursts imposés ⇒ n fenêtres `inference`, bornes au pas près."""
    courant = np.full(1000, I_IDLE_A)
    courant[200:300] = I_ACTIVE_A
    courant[600:650] = I_ACTIVE_A

    fenetres = ec.derive_phase_windows_from_current(courant, DT_S)
    actives = [(t0, t1) for name, t0, t1 in fenetres if name == "inference"]

    assert ec.count_bursts(fenetres) == 2
    assert actives[0] == pytest.approx((200 * DT_S, 300 * DT_S))
    assert actives[1] == pytest.approx((600 * DT_S, 650 * DT_S))
    # Les plateaux bas encadrent les créneaux : la trace est entièrement couverte.
    assert [n for n, _, _ in fenetres] == [
        "idle", "inference", "idle", "inference", "idle"]


def test_seuil_deduit_de_la_trace_pas_saisi():
    """Le seuil vient de la trace (médiane + k·MAD) : il ne se passe pas en argument.

    Critère d'acceptation S5305 — un seuil saisi à la main ne serait traçable à rien.
    """
    trace = make_trace(n_bursts=100, duration_s=1.0, burst_us=2095.0)
    seuil = ec.robust_current_threshold(trace["current_a"])
    assert I_IDLE_A < seuil < I_ACTIVE_A
    # Sans seuil explicite, le découpage est le même qu'avec le seuil déduit.
    assert (ec.derive_phase_windows_from_current(trace["current_a"], DT_S)
            == ec.derive_phase_windows_from_current(trace["current_a"], DT_S, seuil))


def test_repos_bruite_ne_produit_aucun_creneau():
    """Un repos sans inférence ne doit rien déclencher — sinon la méthode fabriquerait
    des créneaux à partir du bruit d'acquisition."""
    rng = np.random.default_rng(7)
    repos = I_IDLE_A + rng.normal(0.0, 1e-4, size=100_000)
    fenetres = ec.derive_phase_windows_from_current(repos, DT_S)
    assert ec.count_bursts(fenetres) == 0


# ── 3. Contrôle du dénominateur : le refus honnête ───────────────────────────

def test_ecart_dans_la_tolerance_accepte():
    ok, ecart, raison = ec.validate_burst_count(98, rate_hz=10.0, duration_s=10.0)
    assert ok and raison is None
    assert ecart == pytest.approx(-2.0)


def test_ecart_hors_tolerance_refuse_avec_raison_chiffree():
    ok, ecart, raison = ec.validate_burst_count(90, rate_hz=10.0, duration_s=10.0)
    assert not ok
    assert ecart == pytest.approx(-10.0)
    assert "90" in raison and "100" in raison
    assert "%" in raison


def test_cadence_nulle_refusee():
    """Sans cadence imposée, le dénominateur d'une énergie par inférence n'existe pas."""
    ok, _, raison = ec.validate_burst_count(50, rate_hz=0.0, duration_s=10.0)
    assert not ok and raison


# ── 4. Profil complet : publication et N/A ───────────────────────────────────

def test_profil_publie_energie_marginale_et_brute():
    """Trace conforme : l'énergie sort chiffrée, marginale ET brute, sans confusion."""
    trace = make_trace(n_bursts=100, duration_s=1.0, burst_us=2095.0)
    bloc = ec.profile_from_current(trace, rate_hz=100.0)

    assert bloc["segmentation"] == "courant (voie A)"
    assert bloc["n_bursts_detected"] == 100
    assert abs(bloc["detection_error_pct"]) <= ec.BURST_TOLERANCE * 100
    assert "energy_na_reason" not in bloc

    # Marginale = brute moins le repos intégré sur la durée du créneau : strictement
    # inférieure, et c'est elle qui est comparable au delta et à la régression.
    assert 0 < bloc["energy_uj_per_inference"] < bloc["energy_uj_per_burst_gross"]
    # Contrôle arithmétique : surcoût × V × durée du créneau, mesuré sur la trace.
    attendu = (I_ACTIVE_A - I_IDLE_A) * VOLTAGE_V * 2095e-6 * 1e6
    assert bloc["energy_uj_per_inference"] == pytest.approx(attendu, rel=0.02)
    assert bloc["total_uj"] == pytest.approx(
        bloc["phases_uj"]["inference"] + bloc["phases_uj"]["idle"])


def test_profil_refuse_publie_a_mesurer_jamais_zero():
    """Segmentation hors tolérance ⇒ « à mesurer » + raison, jamais 0."""
    trace = make_trace(n_bursts=90, duration_s=1.0, burst_us=2095.0)
    bloc = ec.profile_from_current(trace, rate_hz=100.0)

    assert bloc["n_bursts_detected"] == 90
    assert bloc["energy_uj_per_inference"] == ec.A_MESURER
    assert bloc["total_uj"] == ec.A_MESURER
    assert all(v == ec.A_MESURER for v in bloc["phases_uj"].values())
    assert "90" in bloc["energy_na_reason"]


def test_profil_sans_tension_refuse():
    """Fabriquer une tension d'alimentation est interdit : la cellule sort en N/A."""
    trace = make_trace(n_bursts=100, duration_s=1.0, burst_us=2095.0)
    trace["voltage_v"] = None
    bloc = ec.profile_from_current(trace, rate_hz=100.0)
    assert bloc["energy_uj_per_inference"] == ec.A_MESURER
    assert "tension" in bloc["energy_na_reason"]


def test_limite_1bit_conservee():
    """`startup` et `acquisition` restent « à mesurer » — elles ne valent pas zéro."""
    trace = make_trace(n_bursts=100, duration_s=1.0, burst_us=2095.0)
    bloc = ec.profile_from_current(trace, rate_hz=100.0)
    assert bloc["phases_uj"]["startup"] == ec.A_MESURER
    assert bloc["phases_uj"]["acquisition"] == ec.A_MESURER
    assert isinstance(bloc["phases_uj"]["inference"], float)
    assert "1 bit" in bloc["phases_na_reason"]


def test_echantillons_par_creneau_expose_la_limite():
    """EWC (~50 µs ⇒ 5 échantillons à 100 kSPS) est à la limite du segmentable : le JSON
    doit le rendre visible au lieu de le laisser deviner."""
    lent = ec.profile_from_current(
        make_trace(100, 1.0, burst_us=2095.0), rate_hz=100.0)
    rapide = ec.profile_from_current(
        make_trace(100, 1.0, burst_us=50.0), rate_hz=100.0)
    assert lent["samples_per_burst_median"] == pytest.approx(2095e-6 * FS_HZ, rel=0.01)
    assert rapide["samples_per_burst_median"] == pytest.approx(5.0, abs=1.0)


def test_voies_a_et_b_publient_la_meme_grandeur():
    """Sur une trace portant les DEUX informations, les deux voies doivent s'accorder.

    C'est la garde contre l'asymétrie la plus coûteuse du sprint : si la voie PA8
    publiait une consommation brute et la voie A un surcoût sous le même nom de champ,
    la comparaison des trois estimateurs (S5309) comparerait des grandeurs différentes.
    """
    rs = _load("run_s53_phase_profile", ROOT / "scripts" / "run_s53_phase_profile.py")
    trace = make_trace(n_bursts=100, duration_s=1.0, burst_us=2095.0)
    # Colonne de synchronisation dérivée de la même vérité, comme le ferait PA8.
    trace_pa8 = dict(trace)
    trace_pa8["sync"] = (trace["current_a"] > (I_IDLE_A + I_ACTIVE_A) / 2).astype(float)

    voie_a = rs.profile_trace(trace, 100.0, None, ec.BURST_TOLERANCE)
    voie_b = rs.profile_trace(trace_pa8, 100.0, None, ec.BURST_TOLERANCE)

    assert voie_a["segmentation"] == "courant (voie A)"
    assert voie_b["segmentation"] == "pa8_d7 (voie B)"
    assert voie_b["energy_uj_per_inference"] == pytest.approx(
        voie_a["energy_uj_per_inference"], rel=0.02)
    assert voie_b["energy_uj_per_inference"] < voie_b["energy_uj_per_burst_gross"]


# ── 5. Les trois estimateurs ne fusionnent jamais ────────────────────────────

def test_methode_distincte_des_autres_estimateurs():
    assert ec.METHOD_INTEGRATION != rr.METHOD
    assert "intégration" in ec.METHOD_INTEGRATION
    bloc = ec.profile_from_current(
        make_trace(100, 1.0, burst_us=2095.0), rate_hz=100.0)
    assert bloc["method"] == ec.METHOD_INTEGRATION


# ── 6. Schéma des cellules — skip tant que le banc n'a pas tourné ────────────

CELLS = sorted(p for p in EXP_DIR.glob("*.json")
               if p.name != "summary.json") if EXP_DIR.is_dir() else []


@pytest.mark.skipif(not CELLS, reason="aucune cellule mesurée (carte + sonde requises)")
@pytest.mark.parametrize("path", CELLS, ids=lambda p: p.stem)
def test_schema_cellule(path):
    cell = json.loads(path.read_text(encoding="utf-8"))
    for champ in ("model", "sysclk_mhz", "acqmode", "fs_hz", "method",
                  "phases_uj", "total_uj", "energy_uj_per_inference"):
        assert champ in cell, f"{path.name} : champ {champ} manquant"
    assert cell["method"] == ec.METHOD_INTEGRATION
    energie = cell["energy_uj_per_inference"]
    if energie == ec.A_MESURER:
        assert cell.get("energy_na_reason"), "un N/A DOIT porter sa raison mesurée"
    else:
        assert energie > 0
        assert abs(cell["detection_error_pct"]) <= ec.BURST_TOLERANCE * 100
