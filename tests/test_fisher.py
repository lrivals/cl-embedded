import torch
from torch.utils.data import DataLoader, TensorDataset
from src.models.ewc import EWCMlpClassifier
from src.models.ewc.fisher import compute_fisher_diagonal, update_fisher_online


def _make_loader(n: int = 64, input_dim: int = 6) -> DataLoader:
    x = torch.randn(n, input_dim)
    y = torch.randint(0, 2, (n, 1)).float()
    return DataLoader(TensorDataset(x, y), batch_size=32)


def test_fisher_shape():
    """La Fisher doit avoir la même structure que les paramètres du modèle."""
    model = EWCMlpClassifier()
    loader = _make_loader()
    fisher = compute_fisher_diagonal(model, loader, torch.device("cpu"))
    for name, param in model.named_parameters():
        assert name in fisher
        assert fisher[name].shape == param.shape


def test_fisher_non_negative():
    """La Fisher est constituée de carrés de gradients → valeurs ≥ 0."""
    model = EWCMlpClassifier()
    loader = _make_loader()
    fisher = compute_fisher_diagonal(model, loader, torch.device("cpu"))
    for name, f in fisher.items():
        assert (f >= 0).all(), f"Fisher négative sur {name}"


def test_fisher_non_zero():
    """La Fisher ne doit pas être entièrement nulle sur un modèle entraînable."""
    model = EWCMlpClassifier()
    loader = _make_loader()
    fisher = compute_fisher_diagonal(model, loader, torch.device("cpu"))
    total = sum(f.sum().item() for f in fisher.values())
    assert total > 0, "Fisher entièrement nulle — gradients bloqués ?"


def test_update_fisher_online_none():
    """Avec fisher_old=None, update_fisher_online doit retourner une copie de fisher_new."""
    model = EWCMlpClassifier()
    loader = _make_loader()
    fisher_new = compute_fisher_diagonal(model, loader, torch.device("cpu"))
    fisher_updated = update_fisher_online(None, fisher_new, gamma=0.9)
    for name in fisher_new:
        assert torch.allclose(fisher_updated[name], fisher_new[name])


def test_update_fisher_online_accumulates():
    """Après deux tâches, la Fisher Online doit être plus grande qu'après une seule."""
    model = EWCMlpClassifier()
    loader = _make_loader()
    fisher_t1 = compute_fisher_diagonal(model, loader, torch.device("cpu"))
    fisher_t2 = compute_fisher_diagonal(model, loader, torch.device("cpu"))
    fisher_online = update_fisher_online(fisher_t1, fisher_t2, gamma=0.9)
    # γ·F_t1 + F_t2 > F_t2 si F_t1 > 0
    for name in fisher_t2:
        assert (fisher_online[name] >= fisher_t2[name]).all()


def test_n_fisher_samples_limit():
    """Avec n_samples petit, seule une partie du loader doit être utilisée."""
    model = EWCMlpClassifier()
    loader = _make_loader(n=256)
    # Ne pas planter avec n_samples < taille du loader
    fisher = compute_fisher_diagonal(model, loader, torch.device("cpu"), n_samples=32)
    assert fisher is not None
