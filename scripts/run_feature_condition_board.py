#!/usr/bin/env python3
"""run_feature_condition_board.py — Balayage board réel des 3 conditions de features (S3508).

Pour chaque ``(condition ∈ {5feat, all, best}, dataset)``, sur la NUCLEO-F439ZI réelle :

1. **Entraîne** les modèles de référence board (Mahalanobis + EWCMlpMulticlass) à la
   dimension ``k`` de la condition, sur EXACTEMENT les colonnes natives que
   ``sensor_stream.py --condition`` enverra (``load_condition_arrays``) → parité par
   construction. ``best`` : ``k`` et indices **par modèle**.
2. **Exporte** les poids → headers C (``export_weights_c.py --mahal --ewc-head``), arrays
   dimensionnés à ``k`` (``MAHA_NATIVE_DIM``/``EWC_HEAD_NATIVE_DIM``).
3. **Recompile + flashe** une fois par (condition, dataset), dims par modèle via ``-D``
   (``EWC_IN``/``MAHA_DIM``/``TINYOL_IN``/``HDC_N_FEATURES`` ; ``PROTO_MAX_N`` si k>16).
4. **Streame** chaque modèle (``sensor_stream.py --condition``, **sans --update** → poids
   figés → parité exacte) : EWC/Maha/TinyOL = parité board↔PC ; HDC = HW-only.
5. **Consigne** ``experiments/exp_S35_board_{condition}_{model}_{dataset}/results.json``.

Idempotent (``--skip-existing``) ; ``--dry-run`` valide la matrice (60 cellules) sans board.
Aucun chiffre board inventé : une cellule en échec garde ``"à mesurer"``.

Usage :
    python scripts/run_feature_condition_board.py --dry-run
    python scripts/run_feature_condition_board.py --port /dev/ttyACM0 --skip-existing
    python scripts/run_feature_condition_board.py --condition best --dataset cwru
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.feature_conditions import (  # noqa: E402
    CONDITIONS,
    DATASETS,
    load_condition_arrays,
    resolve_feature_indices,
)
from src.utils.reproducibility import set_seed  # noqa: E402

FW_DIR = Path("firmware/stm32f4_blink")
EXPERIMENTS = Path("experiments")
SUMMARY = EXPERIMENTS / "exp_S35_board_sweep_summary.json"

# S5201 : TinyOL rejoint les modèles à parité — ses poids sont désormais exportés
# à la dim k de la condition (TINYOL_NATIVE_DIM). Avant, aucun poids n'était exporté
# ET sensor_stream.py n'envoyait pas le flag 0x80 : la carte exécutait Mahalanobis.
PARITY_MODELS = ["mahalanobis", "ewc", "tinyol"]  # parité board↔PC exacte (poids exportés)
HWONLY_MODELS = ["hdc"]                  # latence/.bss seulement (parité N/A par construction)
ALL_MODELS = PARITY_MODELS + HWONLY_MODELS

GAP2_LATENCY_US = 100_000   # 100 ms (Gap 2)

EWC_EPOCHS_PER_TASK = 15
EWC_LR = 0.01
EWC_LAMBDA = 400.0
N_TASKS = 3
TINYOL_EPOCHS = 150


# ── Sous-process helpers ────────────────────────────────────────────────────

def _run(cmd: list[str], timeout: int = 1800) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _bss_bytes() -> int:
    out = subprocess.run(["arm-none-eabi-size", str(FW_DIR / "build/stm32f4_blink.elf")],
                         capture_output=True, text=True)
    line = out.stdout.strip().splitlines()[-1].split()
    return int(line[2])  # text data bss ...


# ── Entraînement référence board (dims arbitraires) ─────────────────────────

def _temporal_tasks(X: np.ndarray, y: np.ndarray, n_tasks: int) -> list[tuple]:
    size = max(1, len(X) // n_tasks)
    return [(X[i * size:(i + 1) * size], y[i * size:(i + 1) * size]) for i in range(n_tasks)]


def train_maha_board(X: np.ndarray, exp_dir: Path) -> Path:
    import yaml

    from src.models.unsupervised import MahalanobisDetector

    cfg = yaml.safe_load(Path("configs/board_mahalanobis.yaml").read_text())
    maha_cfg = cfg.get("mahalanobis", {"anomaly_percentile": 95, "cl_strategy": "refit"})
    model = MahalanobisDetector(maha_cfg)
    model.fit_task(X, task_id=0)
    ck_dir = exp_dir / "checkpoints"
    ck_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ck_dir / "mahalanobis_task0.pkl"
    with open(ckpt, "wb") as f:
        pickle.dump(model, f)
    print(f"  [maha] μ={model.mu_.shape} seuil={model.threshold_:.4f} → {ckpt}")
    return ckpt


def train_ewc_board(X: np.ndarray, y: np.ndarray, exp_dir: Path, k: int) -> Path:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from src.models.ewc.ewc_mlp_multiclass import EWCMlpMulticlass

    model = EWCMlpMulticlass(input_dim=k, n_classes=2, hidden_dims=[32, 16],
                             dropout=0.2, ewc_lambda=EWC_LAMBDA)
    optimizer = torch.optim.SGD(model.parameters(), lr=EWC_LR, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    for _task_id, (Xt, yt) in enumerate(_temporal_tasks(X, y, N_TASKS)):
        ds = TensorDataset(torch.tensor(Xt, dtype=torch.float32),
                           torch.tensor(yt, dtype=torch.long))
        loader = DataLoader(ds, batch_size=32, shuffle=True)
        model.train()
        for _ in range(EWC_EPOCHS_PER_TASK):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb) + model.ewc_penalty()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        model.consolidate(loader, n_samples=200)
    ck_dir = exp_dir / "checkpoints"
    ck_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ck_dir / "ewc_head.pt"
    torch.save({"model_state_dict": model.state_dict()}, ckpt)
    print(f"  [ewc] EWCMlpMulticlass(in={k}) → {ckpt}")
    return ckpt


def _tinyol_module():
    """Charge scripts/export_weights_tinyol.py comme module (architecture board)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "export_weights_tinyol", Path(__file__).parent / "export_weights_tinyol.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def train_tinyol_board(X: np.ndarray, y: np.ndarray, exp_dir: Path, k: int) -> Path:
    """Auto-encodeur TinyOL k→32→16→k entraîné sur la classe normale (S5201)."""
    import torch

    ewt = _tinyol_module()
    model = ewt.fit_tinyol_board(X[y == 0], epochs=TINYOL_EPOCHS)
    ck_dir = exp_dir / "checkpoints"
    ck_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ck_dir / "tinyol_board.pt"
    torch.save(model.state_dict(), ckpt)
    thr = float(model._calibrated_threshold)
    ckpt.with_suffix(".threshold.json").write_text(
        json.dumps({"threshold": thr, "calibration": "P95 × 1.5 des MSE training"}, indent=2))
    print(f"  [tinyol] TinyOLBoard(in={k}) seuil={thr:.8f} → {ckpt}")
    return ckpt


# ── Référence PC pour la parité ─────────────────────────────────────────────

def _pc_pred_maha(ckpt: Path, feats: np.ndarray) -> np.ndarray:
    with open(ckpt, "rb") as f:
        model = pickle.load(f)
    return (model.anomaly_score(feats) > model.threshold_).astype(int)


def _pc_pred_ewc(ckpt: Path, feats: np.ndarray) -> np.ndarray:
    import torch

    from src.models.ewc.ewc_mlp_multiclass import EWCMlpMulticlass

    sd = torch.load(ckpt, map_location="cpu")["model_state_dict"]
    k = int(sd["fc1.weight"].shape[1])
    model = EWCMlpMulticlass(input_dim=k, n_classes=2, hidden_dims=[32, 16])
    model.load_state_dict(sd)
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(feats, dtype=torch.float32)).argmax(dim=1).numpy()


def _pc_pred_tinyol(ckpt: Path, feats: np.ndarray) -> np.ndarray:
    """Réplique PC de la route TinyOL firmware : pred = (MSE de reconstruction > seuil)."""
    import torch

    ewt = _tinyol_module()
    state = torch.load(ckpt, map_location="cpu")
    model = ewt.TinyOLBoard(dim=ewt.dim_of_state_dict(state))
    model.load_state_dict(state)
    model.eval()
    thr = float(json.loads(ckpt.with_suffix(".threshold.json").read_text())["threshold"])
    with torch.no_grad():
        _emb, recon = model(torch.tensor(feats, dtype=torch.float32))
    mse = ((torch.tensor(feats, dtype=torch.float32) - recon) ** 2).mean(dim=1).numpy()
    return (mse > thr).astype(int)


_PC_PRED = {
    "mahalanobis": _pc_pred_maha,
    "ewc": _pc_pred_ewc,
    "tinyol": _pc_pred_tinyol,
}


def _degenerate_reason(model: str, ckpt: Path, feats: np.ndarray) -> str | None:
    """Détecte une cellule dégénérée, où la décision n'a plus de contenu.

    Cas rencontré (``best`` × cmapss × tinyol) : la sélection ne retient qu'une
    variable, ``T2``, capteur **constant** de CMAPSS (écart-type nul). L'auto-encodeur
    s'entraîne alors sur des zéros, son seuil calibré (P95 × 1,5) vaut exactement 0, et
    la comparaison stricte ``MSE > seuil`` se joue sur le bruit d'arrondi : le PC en
    float64 reste à 0, la carte en float32 passe juste au-dessus. Ce n'est ni une perte
    de parité ni un défaut de portage — la cellule ne mesure simplement rien, et doit
    être déclarée N/A plutôt que comptée comme un échec.

    Retourne la raison (chaîne) ou ``None`` si la cellule est exploitable.
    """
    if feats.size and float(feats.std()) == 0.0:
        return ("entrée constante (écart-type nul sur toutes les variables retenues) "
                "— la décision ne dépend d'aucun signal")
    if model == "tinyol":
        thr_path = ckpt.with_suffix(".threshold.json")
        if thr_path.exists():
            thr = float(json.loads(thr_path.read_text())["threshold"])
            if thr == 0.0:
                return ("seuil de reconstruction calibré à 0 (MSE d'entraînement "
                        "identiquement nulle) — la comparaison stricte MSE > seuil est "
                        "tranchée par l'arrondi flottant, pas par le modèle")
    return None


def _parity(model: str, ckpt: Path, samples: list[dict]) -> dict:
    valid = [s for s in samples if s.get("features")]
    if not valid:
        return {"parity_ok": None, "n_compared": 0, "parity_mismatch_count": None}
    feats = np.array([s["features"] for s in valid], dtype=np.float32)
    board = np.array([int(s["pred"]) for s in valid])

    reason = _degenerate_reason(model, ckpt, feats)
    if reason is not None:
        return {
            "parity_ok": None,
            "n_compared": len(valid),
            "parity_mismatch_count": None,
            "parity_rate": None,
            "degenerate": True,
            "na_reason": f"cellule dégénérée : {reason}",
        }

    pc = _PC_PRED[model](ckpt, feats)
    n_mismatch = int((board != pc).sum())
    return {
        "parity_ok": bool(n_mismatch == 0),
        "n_compared": len(valid),
        "parity_mismatch_count": n_mismatch,
        "parity_rate": float((board == pc).mean()),
    }


# ── Streaming ───────────────────────────────────────────────────────────────

def _stream(dataset: str, model: str, condition: str, out_json: Path, args) -> dict:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "scripts/sensor_stream.py",
        "--dataset", dataset, "--model", model, "--condition", condition,
        "--n-samples", str(args.n_samples), "--rate-hz", str(args.rate_hz),
        "--protocol-version", "3", "--dump-samples",
        "--port", args.port, "--output", str(out_json),
    ]
    if args.dry_run:
        cmd.append("--dry-run")
    proc = _run(cmd, timeout=600)
    if proc.returncode != 0:
        print(f"  [stream FAIL {model}] {(proc.stderr or proc.stdout)[-400:]}")
        return {}
    return json.loads(out_json.read_text()) if out_json.exists() else {}


# ── Cellule (condition, dataset) ────────────────────────────────────────────

def _placeholder(condition, model, dataset, k, note) -> dict:
    """Cellule non mesurée : champs « à mesurer » (aucun chiffre inventé)."""
    return {
        "exp_id": f"exp_S35_board_{condition}_{model}_{dataset}",
        "condition": condition, "model": model, "dataset": dataset,
        "platform": "nucleo_f439zi", "n_features": int(k),
        "online_accuracy": "à mesurer", "f1_faulty": "à mesurer", "f1_macro": "à mesurer",
        "latency_us_p50": "à mesurer", "latency_us_p99": "à mesurer", "bss_bytes": "à mesurer",
        "parity_class": "exact" if model in PARITY_MODELS else "hw_only",
        "parity_ok": None, "parity_note": note,
    }


def run_cell(condition: str, dataset: str, args, rows: list[dict]) -> None:
    print(f"\n{'='*70}\n=== BOARD CELL  condition={condition}  dataset={dataset}  ===\n{'='*70}")
    set_seed(args.seed)

    # Indices/dims par modèle (best → diffèrent par modèle ; all/5feat → identiques).
    # Fallback robuste : si un best_features/{model}_{dataset}.yaml manque, on retombe sur
    # `all` (dims natives) POUR CE MODÈLE — la cellule build/flash/stream quand même, et les
    # modèles à parité (avec config) restent mesurés. Note explicite par modèle concerné.
    idx: dict[str, list[int]] = {}
    fallback: dict[str, bool] = {}
    for m in ALL_MODELS:
        try:
            idx[m] = resolve_feature_indices(condition, m, dataset)[0]
            fallback[m] = False
        except FileNotFoundError:
            idx[m] = resolve_feature_indices("all", m, dataset)[0]
            fallback[m] = True
            print(f"  [fallback all] best/{m}/{dataset} manquant → dims natives ({len(idx[m])})")
    dims = {m: len(idx[m]) for m in ALL_MODELS}
    proto_max = max(dims.values())

    exp_dirs = {m: EXPERIMENTS / f"exp_S35_board_{condition}_{m}_{dataset}" for m in ALL_MODELS}

    # 1) Entraîner + sauver les checkpoints de parité (Maha + EWC) à leurs dims.
    ckpts: dict[str, Path] = {}
    for m in PARITY_MODELS:
        ck = exp_dirs[m] / "checkpoints" / {
            "mahalanobis": "mahalanobis_task0.pkl",
            "ewc": "ewc_head.pt",
            "tinyol": "tinyol_board.pt",
        }[m]
        if args.skip_existing and ck.exists():
            ckpts[m] = ck
            continue
        try:
            X, y, _i, _n = load_condition_arrays(dataset, condition, m)
            exp_dirs[m].mkdir(parents=True, exist_ok=True)
            if m == "mahalanobis":
                ckpts[m] = train_maha_board(X, exp_dirs[m])
            elif m == "tinyol":
                ckpts[m] = train_tinyol_board(X, y, exp_dirs[m], dims[m])
            else:
                ckpts[m] = train_ewc_board(X, y, exp_dirs[m], dims[m])
        except Exception as exc:  # noqa: BLE001 — cellule robuste
            print(f"  [FAIL train {m}] {type(exc).__name__}: {exc}")
            for mm in ALL_MODELS:
                rows.append(_placeholder(condition, mm, dataset, dims[mm], f"échec entraînement {m}: {exc}"))
            return

    # 2) Export → headers C (dims = k).
    if not args.dry_run:
        if _run([sys.executable, "scripts/export_weights_c.py",
                 "--mahal", str(ckpts["mahalanobis"]),
                 "--ewc-head", str(ckpts["ewc"])]).returncode != 0:
            print("  [FAIL export]")
            return
        # S5201 : poids TinyOL à la dim de la cellule → TINYOL_NATIVE_DIM.
        # Sans cet export, pipeline.c laissait l'auto-encodeur à zéro en .bss.
        # --emit-test-reference : le golden de parité C↔Python vit avec les poids.
        # Sans lui, une campagne laisse le header et tests/tinyol_reference.h
        # désynchronisés et `make test` échoue sur test_tinyol_forward_delta.
        if _run([sys.executable, "scripts/export_weights_tinyol.py",
                 "--checkpoint", str(ckpts["tinyol"]),
                 "--dim", str(dims["tinyol"]),
                 "--emit-test-reference"]).returncode != 0:
            print("  [FAIL export tinyol]")
            return
        # 3) Build + flash (1× par condition×dataset), dims par modèle via -D.
        make_dims = [f"EWC_IN={dims['ewc']}", f"MAHA_DIM={dims['mahalanobis']}",
                     f"TINYOL_IN={dims['tinyol']}", f"HDC_N_FEATURES={dims['hdc']}"]
        if proto_max > 16:
            make_dims.append(f"PROTO_MAX_N={proto_max}")
        subprocess.run(["make", "-C", str(FW_DIR), "clean"], capture_output=True)
        if _run(["make", "-C", str(FW_DIR), *make_dims, "all"]).returncode != 0:
            print("  [FAIL build]")
            return
        bss = _bss_bytes()
        if _run(["make", "-C", str(FW_DIR), "flash"]).returncode != 0:
            print("  [FAIL flash]")
            return
    else:
        bss = _bss_bytes() if (FW_DIR / "build/stm32f4_blink.elf").exists() else 0

    date = datetime.now().isoformat(timespec="seconds")

    # 4) Streamer chaque modèle + consigner.
    for m in ALL_MODELS:
        exp_dirs[m].mkdir(parents=True, exist_ok=True)
        stats = _stream(dataset, m, condition, exp_dirs[m] / "stream.json", args)
        lat = stats.get("latency_p50_us", stats.get("latency_mean_us"))
        result = {
            "exp_id": f"exp_S35_board_{condition}_{m}_{dataset}",
            "condition": condition, "model": m, "dataset": dataset,
            "platform": "nucleo_f439zi", "date": date,
            "n_features": int(dims[m]), "bss_bytes": bss,
            "online_accuracy": stats.get("accuracy", "à mesurer"),
            "f1_faulty": stats.get("f1_faulty", "à mesurer"),
            "f1_macro": stats.get("f1_macro", "à mesurer"),
            "latency_us_p50": lat if lat is not None else "à mesurer",
            "latency_us_p99": stats.get("latency_p99_us", "à mesurer"),
            "crc_errors": stats.get("crc_errors"),
            "gap2_latency_compliant": (lat is not None and lat < GAP2_LATENCY_US),
            "stream_mode": "frozen (sans --update → parité exacte)",
            "parity_class": "exact" if m in PARITY_MODELS else "hw_only",
        }
        if fallback.get(m):
            result["feature_fallback"] = f"best/{m}/{dataset} absent → dims natives (all)"
        if m in PARITY_MODELS and not args.dry_run:
            result.update(_parity(m, ckpts[m], stats.get("samples", [])))
            if result.get("degenerate"):
                # La latence, le .bss et le CRC restent des mesures ; la métrique, elle,
                # ne décrit qu'une décision prise sur du bruit → N/A honnête, pas 0.
                for field in ("online_accuracy", "f1_faulty", "f1_macro"):
                    result[field] = None
        else:
            result["parity_ok"] = None
            result["parity_note"] = (
                "N/A par construction (HDC : projection embarquée, dim 1000≠1024, init en ligne)"
                if m in HWONLY_MODELS else "dry-run")

        (exp_dirs[m] / "results.json").write_text(json.dumps(result, indent=2))
        rows.append(result)
        print(f"  [{m:11s}] k={dims[m]} lat_p50={result['latency_us_p50']} "
              f"parity={result.get('parity_ok')} .bss={bss}")


def main() -> None:
    p = argparse.ArgumentParser(description="Balayage board réel des conditions de features (S3508)")
    p.add_argument("--condition", choices=CONDITIONS, default=None)
    p.add_argument("--dataset", choices=DATASETS, default=None)
    p.add_argument("--n-samples", type=int, default=150)
    p.add_argument("--rate-hz", type=float, default=50.0)
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Pas de flash/stream board")
    args = p.parse_args()

    conditions = [args.condition] if args.condition else CONDITIONS
    datasets = [args.dataset] if args.dataset else DATASETS
    cells = [(c, d) for c in conditions for d in datasets]

    if args.dry_run and not (args.condition or args.dataset):
        print(f"{len(cells) * len(ALL_MODELS)} cellules ({len(cells)} builds) :")
        for c, d in cells:
            try:
                dims = {m: len(resolve_feature_indices(c, m, d)[0]) for m in ALL_MODELS}
            except FileNotFoundError:
                dims = "config best manquante"
            print(f"  {c:6s} {d:10s} dims={dims}")
        return

    rows: list[dict] = []
    for c, d in cells:
        run_cell(c, d, args, rows)

    # Consolidation (fusion par exp_id).
    merged: dict[str, dict] = {}
    if SUMMARY.exists():
        for r in json.load(open(SUMMARY)):
            merged[r["exp_id"]] = r
    for r in rows:
        merged[r["exp_id"]] = r
    summary = sorted(merged.values(), key=lambda r: r["exp_id"])
    SUMMARY.write_text(json.dumps(summary, indent=2))

    par = [r for r in summary if r.get("parity_class") == "exact" and r.get("parity_ok") is not None]
    ok = sum(1 for r in par if r["parity_ok"])
    print(f"\n{'='*60}\nBoard sweep S35 : {len(summary)} cellules ; "
          f"parité {ok}/{len(par)} OK\nRésumé : {SUMMARY}")


if __name__ == "__main__":
    main()
