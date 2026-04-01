"""
memory_profiler.py — Profiling RAM pour les modèles CL embarqués.

Objectif : produire les chiffres de RAM mesurés nécessaires pour valider
le Gap 2 (opération sub-100 Ko avec chiffres précis).

Note méthodologique :
    tracemalloc mesure l'allocateur Python/PyTorch, pas la RAM C native.
    Les chiffres produits ici sont des PROXIES PC.
    Les mesures MCU réelles seront effectuées en Phase 2 (portage STM32N6).
    Les deux mesures doivent être reportées distinctement dans le manuscrit.
"""

from __future__ import annotations

import tracemalloc
import time
from typing import Callable, Any

import numpy as np
import torch
import torch.nn as nn


def profile_forward_pass(
    model: nn.Module,
    input_shape: tuple,
    n_runs: int = 100,
    device: str = "cpu",
) -> dict:
    """
    Profile la mémoire et la latence d'un forward pass.

    Parameters
    ----------
    model : nn.Module
        Modèle à profiler.
    input_shape : tuple
        Forme du tenseur d'entrée (ex. (1, 6) pour un seul exemple de 6 features).
    n_runs : int
        Nombre de runs pour la moyenne de latence.
    device : str

    Returns
    -------
    dict :
        ram_peak_bytes        : pic RAM Python (tracemalloc)
        ram_current_bytes     : RAM courante après le forward
        inference_latency_ms  : latence moyenne sur n_runs (ms)
        inference_latency_std_ms : écart-type latence
        n_params              : nombre de paramètres
        params_fp32_bytes     : estimation poids FP32
        params_int8_bytes     : estimation poids INT8
        within_budget_64ko    : bool — RAM < 65 536 B
    """
    model.eval()
    dummy = torch.zeros(input_shape, device=device)

    # --- Mesure mémoire ---
    tracemalloc.start()
    with torch.no_grad():
        _ = model(dummy)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # --- Mesure latence ---
    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(dummy)
            latencies.append((time.perf_counter() - t0) * 1000)  # ms

    # --- Paramètres ---
    n_params = sum(p.numel() for p in model.parameters())

    return {
        "ram_peak_bytes": peak,
        "ram_current_bytes": current,
        "inference_latency_ms": float(np.mean(latencies)),
        "inference_latency_std_ms": float(np.std(latencies)),
        "n_params": n_params,
        "params_fp32_bytes": n_params * 4,
        "params_int8_bytes": n_params * 1,
        "within_budget_64ko": peak < 65_536,
    }


def profile_cl_update(
    update_fn: Callable[[torch.Tensor, torch.Tensor], float],
    input_shape: tuple,
    label_shape: tuple = (1, 1),
    n_runs: int = 50,
    device: str = "cpu",
) -> dict:
    """
    Profile la mémoire lors d'une mise à jour CL (forward + backward + step).

    Parameters
    ----------
    update_fn : Callable
        Fonction de mise à jour qui prend (x, y) et retourne la loss.
        Doit encapsuler l'optimizer.step().
    input_shape : tuple
    label_shape : tuple
    n_runs : int
    device : str

    Returns
    -------
    dict :
        ram_peak_bytes_update : pic RAM pendant la mise à jour
        update_latency_ms     : latence moyenne d'une mise à jour
    """
    dummy_x = torch.randn(input_shape, device=device)
    dummy_y = torch.zeros(label_shape, device=device)

    tracemalloc.start()
    for _ in range(3):  # warm-up
        update_fn(dummy_x, dummy_y)
    current, peak_warmup = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Mesure propre
    tracemalloc.start()
    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        update_fn(dummy_x, dummy_y)
        latencies.append((time.perf_counter() - t0) * 1000)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "ram_peak_bytes_update": peak,
        "update_latency_ms": float(np.mean(latencies)),
        "update_latency_std_ms": float(np.std(latencies)),
        "within_budget_64ko_update": peak < 65_536,
    }


def full_memory_report(
    model: nn.Module,
    input_shape: tuple,
    update_fn: Callable | None = None,
    model_name: str = "Model",
) -> dict:
    """
    Génère un rapport mémoire complet : forward + update (si fourni).

    Parameters
    ----------
    model : nn.Module
    input_shape : tuple
    update_fn : Callable, optional
        Si fourni, profile aussi la phase de mise à jour CL.
    model_name : str

    Returns
    -------
    dict : rapport complet formaté pour JSON
    """
    print(f"\n🔍 Profiling mémoire : {model_name}")
    print(f"   Input shape : {input_shape}")

    fwd = profile_forward_pass(model, input_shape)

    report = {
        "model": model_name,
        "input_shape": list(input_shape),
        "forward": fwd,
    }

    if update_fn is not None:
        label_shape = (input_shape[0], 1)
        upd = profile_cl_update(update_fn, input_shape, label_shape)
        report["update"] = upd

    # Affichage
    budget = 65_536
    fwd_peak = fwd["ram_peak_bytes"]
    fwd_pct = fwd_peak / budget * 100

    print(f"\n   ┌─ Résultats ─────────────────────────────────────┐")
    print(f"   │  Paramètres    : {fwd['n_params']:>10,} params")
    print(f"   │  RAM FP32      : {fwd['params_fp32_bytes']:>10,} B ({fwd['params_fp32_bytes']/1024:.1f} Ko)")
    print(f"   │  RAM INT8 est. : {fwd['params_int8_bytes']:>10,} B ({fwd['params_int8_bytes']/1024:.1f} Ko)")
    print(f"   │  RAM peak fwd  : {fwd_peak:>10,} B ({fwd_peak/1024:.1f} Ko) — {fwd_pct:.1f}% budget")
    print(f"   │  Latence fwd   : {fwd['inference_latency_ms']:>10.3f} ms (± {fwd['inference_latency_std_ms']:.3f})")

    status = "✅ DANS LE BUDGET" if fwd["within_budget_64ko"] else "❌ DÉPASSE LE BUDGET"
    print(f"   │  STM32N6 64Ko  : {status}")

    if update_fn is not None:
        upd = report["update"]
        upd_peak = upd["ram_peak_bytes_update"]
        print(f"   │  RAM peak upd  : {upd_peak:>10,} B ({upd_peak/1024:.1f} Ko)")
        print(f"   │  Latence upd   : {upd['update_latency_ms']:>10.3f} ms")

    print(f"   └────────────────────────────────────────────────┘")

    return report


def compare_models_memory(reports: list[dict]) -> str:
    """
    Génère un tableau comparatif de la mémoire pour plusieurs modèles.

    Parameters
    ----------
    reports : list[dict]
        Liste de rapports générés par full_memory_report().

    Returns
    -------
    str : tableau formaté.
    """
    header = f"{'Modèle':<20} {'Params':>10} {'RAM FP32':>10} {'RAM peak':>10} {'Latence':>12} {'Budget OK':>10}"
    separator = "-" * len(header)
    rows = [header, separator]

    for r in reports:
        name = r["model"]
        fwd = r["forward"]
        row = (
            f"{name:<20} "
            f"{fwd['n_params']:>10,} "
            f"{fwd['params_fp32_bytes']/1024:>9.1f}Ko "
            f"{fwd['ram_peak_bytes']/1024:>9.1f}Ko "
            f"{fwd['inference_latency_ms']:>10.3f}ms "
            f"{'✅' if fwd['within_budget_64ko'] else '❌':>10}"
        )
        rows.append(row)

    return "\n".join(rows)
