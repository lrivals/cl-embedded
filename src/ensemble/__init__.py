"""Briques d'ensemble (paires de modèles + méta-arbitrage) — Sprints 30–31."""

from src.ensemble.meta_learner import MetaLearner, build_meta_features
from src.ensemble.model_pair import ModelPair, native_to_fault

__all__ = ["MetaLearner", "ModelPair", "build_meta_features", "native_to_fault"]
