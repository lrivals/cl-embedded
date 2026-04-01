"""
reproducibility.py — Contrôle de la reproductibilité des expériences.

Usage :
    from src.utils.reproducibility import set_seed
    set_seed(42)  # à appeler en début de chaque script/notebook
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Fixe tous les seeds pour la reproductibilité complète.

    Couvre : Python random, NumPy, PyTorch CPU et GPU.

    Parameters
    ----------
    seed : int
        Valeur du seed. Default : 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Déterminisme CUDA (peut ralentir les opérations)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"✅ Seed fixé à {seed}")
