"""Tests du pilote LPM01A (`scripts/lpm01a_probe.py`) — décodage, dialogue série, CSV.

Aucun test n'exige la sonde : le décodage est vérifié contre les exemples
canoniques de l'UM2243 et contre un flux binaire reconstruit avec la structure
de blocs réellement observée sur le banc (en-tête `F0F3 … FFFF` toutes les
1000 valeurs). Le contrat CSV est vérifié en le relisant avec le chargeur de
`energy_capture.py`, qui est le vrai consommateur.

Le **dialogue série** est exercé contre une fausse liaison (`FakeSerial`) qui rejoue des
réponses relevées sur le banc. Ce n'est pas une commodité : les quatre défauts de la séance
du 2026-09-07 étaient tous dans ces fonctions-là (`voltage_v`, `take_control`, `hold_run`,
`power_on`), aucune n'était couverte, et l'un des correctifs est passé au vert sans son
`import re` — l'erreur n'aurait explosé qu'au banc, au milieu d'une campagne.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lp = _load("lpm01a_probe", ROOT / "scripts" / "lpm01a_probe.py")
ec = _load("energy_capture", ROOT / "scripts" / "energy_capture.py")


def _build_block(values: list[int], timestamp_ms: int) -> bytes:
    """Reconstruit un bloc du flux binaire : en-tête 9 octets + mots 16 bits."""
    header = b"\xf0\xf3" + timestamp_ms.to_bytes(4, "big") + b"\x00" + b"\xff\xff"
    return header + b"".join(v.to_bytes(2, "big") for v in values)


class TestDecodage:
    def test_ascii_exemple_um2243(self):
        # 6409-07 <=> 6409 x 10^-7 = 640,9 µA
        assert lp.decode_ascii_sample("6409-07") == pytest.approx(640.9e-6)

    def test_binaire_exemple_um2243(self):
        # 52A0 <=> (2A0)16 x 16^-5 = 640,9 µA
        assert lp.decode_bin_word(0x52A0) == pytest.approx(640.9e-6, rel=1e-3)

    def test_exposant_positif_ascii(self):
        assert lp.decode_ascii_sample("2972-03") == pytest.approx(2.972)

    def test_jeton_illisible_leve(self):
        with pytest.raises(ValueError):
            lp.decode_ascii_sample("pas-un-echantillon-valide-xx")

    def test_entetes_de_bloc_ignores(self):
        """Les en-têtes intermédiaires ne doivent produire aucun échantillon."""
        vals = [0x52A0] * 1000
        stream = _build_block(vals, 0) + _build_block(vals, 10) + b"\xf0\xf4\xff\xff"
        out = lp.decode_bin_stream(stream)
        assert out.size == 2000, "chaque en-tête de bloc doit être sauté, pas décodé"
        assert np.allclose(out, 640.9e-6, rtol=1e-3)

    def test_marqueur_de_fin_borne_le_flux(self):
        stream = _build_block([0x52A0] * 10, 0) + b"\xf0\xf4\xff\xff" + b"\xde\xad"
        assert lp.decode_bin_stream(stream).size == 10

    def test_flux_sans_entete_leve(self):
        with pytest.raises(lp.LPM01AError):
            lp.decode_bin_stream(b"\x52\xa0" * 8)

    def test_ascii_ignore_les_lignes_de_service(self):
        text = (
            "\r\nPowerShield > ack start\r\n"
            "TimeStamp: 000s 000ms, buff 00%\r\n"
            "6409-07\r\n0000-10\r\nend\r\n"
        )
        out = lp.decode_ascii_stream(text)
        assert out.size == 2 and out[0] == pytest.approx(640.9e-6)


class TestFrequence:
    @pytest.mark.parametrize("consigne,attendu", [("100k", 1e5), ("10k", 1e4), ("500", 500.0)])
    def test_parse_freq(self, consigne, attendu):
        assert lp.parse_freq_hz(consigne) == attendu

    def test_ascii_limite_a_10khz(self):
        """Au-delà de 10 kHz le format ASCII ne suit plus (contrainte firmware)."""
        assert lp.ASCII_MAX_HZ == 10_000


class TestContratCSV:
    def test_csv_relu_par_energy_capture(self, tmp_path):
        samples = np.array([1e-6, 2e-6, 3e-6, 4e-6])
        out = tmp_path / "trace.csv"
        lp.write_csv(out, samples, voltage_v=3.266, fs_hz=1000.0, metadata={"source": "test"})

        trace = ec._load_csv(out)
        assert trace["current_a"].size == 4
        assert trace["voltage_v"] is not None, "la colonne tension est requise par segment_by_phase"
        assert trace["time_s"][1] == pytest.approx(1e-3), "pas de temps = 1/fs"
        # Pas de voie de synchronisation côté sonde : la limite doit rester visible.
        assert trace["sync"] is None

    def test_energie_integrable(self, tmp_path):
        """La trace produite doit traverser la chaîne S33 sans adaptation."""
        out = tmp_path / "trace.csv"
        lp.write_csv(out, np.full(1000, 1e-3), voltage_v=3.3, fs_hz=1000.0, metadata={})
        uj = ec.segment_by_phase(out, [("idle", 0.0, 1.0)])
        # 1 mA x 3,3 V pendant ~1 s = ~3300 µJ (valeur analytique, pas une mesure).
        assert uj["idle"] == pytest.approx(3300.0, rel=0.01)

    def test_entete_trace_le_controle_dintegrite(self, tmp_path):
        out = tmp_path / "trace.csv"
        lp.write_csv(
            out,
            np.zeros(5),
            voltage_v=3.3,
            fs_hz=1000.0,
            metadata={"n_samples_decoded": "5", "n_samples_expected": "10"},
        )
        head = out.read_text(encoding="utf-8")
        assert "n_samples_decoded" in head and "n_samples_expected" in head


# ── Dialogue série : fausse liaison, réponses relevées sur le banc ────────────

class FakeSerial:
    """Liaison série simulée : journalise les commandes, rejoue des réponses scriptées.

    `responses` associe une commande (ou son préfixe) à la réponse que la sonde renverrait.
    Toute commande non listée reçoit la réponse nominale « ack <commande> », de sorte qu'un
    test ne déclare que ce qui l'intéresse.
    """

    def __init__(self, responses: dict[str, str] | None = None,
                 stream: bytes | None = None) -> None:
        self.responses = responses or {}
        self.commands: list[str] = []
        self.closed = False
        self._pending = b""
        self._stream = stream

    # -- écriture -------------------------------------------------------------
    def write(self, data: bytes) -> int:
        cmd = data.decode().strip()
        self.commands.append(cmd)
        if cmd == "start" and self._stream is not None:
            self._pending = self._stream
        else:
            self._pending = self._reply_for(cmd).encode()
        return len(data)

    def _reply_for(self, cmd: str) -> str:
        for key, value in self.responses.items():
            if cmd == key or cmd.startswith(key + " ") or cmd.startswith(key):
                return value
        return f"\r\nack {cmd}\r\n{lp.PROMPT} "

    # -- lecture --------------------------------------------------------------
    def read(self, size: int = 1) -> bytes:
        out, self._pending = self._pending[:size], self._pending[size:]
        return out

    def flush(self) -> None:
        pass

    def reset_input_buffer(self) -> None:
        self._pending = b""

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _sans_attente(monkeypatch):
    """Neutralise les temporisations du dialogue série (elles cadencent la SONDE, pas le test).

    Les `wait_s` de `command`, l'établissement de `hold_run` et l'auto-calibration de `volt`
    sont des délais matériels : contre une fausse liaison ils n'attendent rien et coûtaient
    68 s de suite. Le contrat vérifié ici est la SÉQUENCE des commandes, pas leur cadence.
    """
    monkeypatch.setattr(lp.time, "sleep", lambda _s: None)


def make_probe(responses: dict[str, str] | None = None,
               stream: bytes | None = None) -> tuple[object, FakeSerial]:
    """Session PowerShield sans ouvrir de port réel (le constructeur ouvre un vrai port)."""
    probe = object.__new__(lp.PowerShield)
    ser = FakeSerial(responses, stream)
    probe.port = "/dev/null"
    probe.ser = ser
    return probe, ser


class TestCommande:
    def test_reponse_en_erreur_leve(self):
        probe, _ = make_probe({"freq": "\r\nerror: unknown parameter\r\n"})
        with pytest.raises(lp.LPM01AError):
            probe.command("freq 42k", wait_s=0.0)

    def test_reponse_acquittee_ne_leve_pas(self):
        probe, _ = make_probe({"freq": "\r\nack freq 100k\r\n"})
        assert "ack" in probe.command("freq 100k", wait_s=0.0)


class TestTension:
    def test_reponse_nominale(self):
        probe, _ = make_probe({"volt get": "\r\nack volt get 3300-03\r\n"})
        assert probe.voltage_v() == pytest.approx(3.3)

    def test_ligne_de_service_intercalee(self):
        """Cas MESURÉ le 2026-09-07 : le dernier jeton valait « handled », pas la tension.

        La cellule entière sortait alors en N/A sur ce seul aléa série. La tension doit se
        lire au FORMAT attendu, pas en position finale.
        """
        probe, _ = make_probe({
            "volt get": "\r\nack volt get 3300-03\r\ncalibration request handled\r\n"
        })
        assert probe.voltage_v() == pytest.approx(3.3)

    def test_aucune_valeur_lisible_leve(self):
        """Aucune tension inventée : sans jeton au format, on lève."""
        probe, _ = make_probe({"volt get": "\r\nack volt get handled\r\n"})
        with pytest.raises(lp.LPM01AError):
            probe.voltage_v()


class TestPriseDeControle:
    def test_reprise_unique_apres_un_refus(self):
        """Un premier `htc` refusé sur un aléa de tampon ne doit pas couler la campagne."""
        etat = {"n": 0}

        class Serie(FakeSerial):
            def _reply_for(self, cmd):
                if cmd == "htc":
                    etat["n"] += 1
                    if etat["n"] == 1:
                        return "\r\nerror: Command unknown\r\n"
                return f"\r\nack {cmd}\r\n"

        probe = object.__new__(lp.PowerShield)
        probe.port, probe.ser = "/dev/null", Serie()
        probe.take_control()
        assert etat["n"] == 2, "la reprise doit émettre un second htc"

    def test_deux_refus_sont_une_vraie_panne(self):
        probe, _ = make_probe({"htc": "\r\nerror: Command unknown\r\n"})
        with pytest.raises(lp.LPM01AError):
            probe.take_control()


class TestMaintienAlimentation:
    """`hold_run` / `power_on` : la sortie doit rester sous tension pendant la commande.

    Hors acquisition, la carte alimentée par la sonde s'effondre et redémarre en boucle
    (constat banc 2026-08-04) : la séquence de commandes EST le contrat.
    """

    def test_hold_run_maintient_et_arrete(self):
        probe, ser = make_probe()
        rc = lp.hold_run(probe, 3300, "true")
        assert rc == 0
        assert "acqmode stat" in ser.commands, "jamais dyn : plafond ~59 mA dépassé"
        assert "pwrend on" in ser.commands
        assert "acqtime inf" in ser.commands
        assert ser.commands.index("start") < ser.commands.index("stop")

    def test_hold_run_arrete_meme_si_la_commande_echoue(self):
        probe, ser = make_probe()
        rc = lp.hold_run(probe, 3300, "false")
        assert rc != 0, "le code de retour est propagé, jamais absorbé"
        assert "stop" in ser.commands, "l'acquisition est arrêtée par le finally"

    def test_power_on_borne_le_maintien(self):
        probe, ser = make_probe({"volt get": "\r\nack volt get 3300-03\r\n"})
        assert "3.300" in lp.power_on(probe, 3300, hold_s=0.01)
        assert "acqmode stat" in ser.commands
        # Sous le plafond firmware, la durée est passée telle quelle.
        assert any(c.startswith("acqtime 0.01") for c in ser.commands)

    def test_power_on_passe_en_duree_infinie_au_dela_du_plafond(self):
        probe, ser = make_probe({"volt get": "\r\nack volt get 3300-03\r\n"})
        lp.power_on(probe, 3300, hold_s=lp.ACQTIME_MAX_S + 0.001)
        assert "acqtime inf" in ser.commands


def _flux_ascii(courants_a: list[float]) -> bytes:
    """Flux ASCII décimal plausible, terminé par le résumé attendu par `capture`."""
    lignes = [f"{int(round(i * 1e7))}-07" for i in courants_a]
    return ("\r\n".join(lignes) + "\r\nsummary beg\r\nsummary end\r\n").encode()


class TestCapture:
    def test_sequence_et_format_ascii_sous_10khz(self):
        probe, ser = make_probe({"volt get": "\r\nack volt get 3300-03\r\n"},
                                stream=_flux_ascii([0.046] * 8))
        samples, voltage_v, _summary = lp.capture(probe, "1k", 0.01, 3300, acqmode="stat")
        assert samples.size == 8 and voltage_v == pytest.approx(3.3)
        ordre = [c.split()[0] for c in ser.commands]
        attendu = ["acqmode", "funcmode", "pwrend", "output", "format",
                   "freq", "acqtime", "trigsrc", "volt"]
        assert ordre[:len(attendu)] == attendu
        assert "format ascii_dec" in ser.commands

    def test_format_binaire_au_dela_de_10khz(self):
        probe, ser = make_probe({"volt get": "\r\nack volt get 3300-03\r\n"},
                                stream=_build_block([0x3BB8] * 4, 0)
                                + b"\xf0\xf4\xff\xff" + b"summary beg summary end")
        lp.capture(probe, "100k", 0.01, 3300)
        assert "format bin_hexa" in ser.commands

    def test_flux_vide_leve(self):
        probe, _ = make_probe({"volt get": "\r\nack volt get 3300-03\r\n"},
                              stream=b"summary beg\r\nsummary end\r\n")
        with pytest.raises(lp.LPM01AError):
            lp.capture(probe, "1k", 0.01, 3300, acqmode="stat")


class TestPlausibiliteDuCourant:
    """Garde-fou A2 : trois exécutions de S5305 ont profilé une carte NON alimentée."""

    def test_carte_non_alimentee_refusee(self):
        probe, _ = make_probe({"volt get": "\r\nack volt get 3300-03\r\n"},
                              stream=_flux_ascii([1e-6] * 8))
        with pytest.raises(lp.LPM01AError, match="implausible"):
            lp.capture(probe, "1k", 0.01, 3300, acqmode="stat")

    def test_courant_reel_de_la_carte_accepte(self):
        """46 mA : le régime le plus bas mesuré sous flux (Mahalanobis, S5008)."""
        probe, _ = make_probe({"volt get": "\r\nack volt get 3300-03\r\n"},
                              stream=_flux_ascii([0.0464] * 8))
        samples, _v, _s = lp.capture(probe, "1k", 0.01, 3300, acqmode="stat")
        assert samples.mean() == pytest.approx(0.0464, rel=1e-3)

    def test_echappatoire_explicite(self):
        """`min_current_a=None` reste possible — mais jamais par défaut."""
        probe, _ = make_probe({"volt get": "\r\nack volt get 3300-03\r\n"},
                              stream=_flux_ascii([1e-6] * 8))
        samples, _v, _s = lp.capture(probe, "1k", 0.01, 3300, acqmode="stat",
                                     min_current_a=None)
        assert samples.size == 8

    def test_seuil_tres_en_dessous_des_regimes_mesures(self):
        """Le seuil ne doit discriminer AUCUN régime réel : 18,2 mA est le plus bas relevé."""
        assert lp.MIN_PLAUSIBLE_CURRENT_A < 0.0182 / 10
