"""
tests/test_ewc.py — Tests unitaires complémentaires pour EWCMlpClassifier.

Exécution :
    pytest tests/test_ewc.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.ewc.ewc_mlp import EWCMlpClassifier


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def model():
    return EWCMlpClassifier(input_dim=6, hidden_dims=[32, 16])


@pytest.fixture
def dummy_batch():
    x = torch.randn(32, 6)
    y = torch.randint(0, 2, (32, 1)).float()
    return x, y


# ------------------------------------------------------------------
# Tests d'architecture
# ------------------------------------------------------------------


def test_forward_shape(model, dummy_batch):
    """Le forward produit une sortie de forme [batch, 1]."""
    x, _ = dummy_batch
    out = model(x)
    assert out.shape == (32, 1), f"Forme attendue (32, 1), obtenu {out.shape}"


def test_forward_range(model, dummy_batch):
    """La sortie sigmoid est dans [0, 1]."""
    x, _ = dummy_batch
    out = model(x).detach().numpy()
    assert np.all(out >= 0) and np.all(out <= 1), "Sortie hors de [0, 1]"


def test_count_parameters(model):
    """Le nombre de paramètres est dans la plage attendue (620–900)."""
    n = model.count_parameters()
    assert 620 <= n <= 900, f"Nombre de paramètres inattendu : {n}"


# ------------------------------------------------------------------
# Tests de contraintes MCU
# ------------------------------------------------------------------


def test_ram_within_budget(model):
    """La RAM estimée des poids est < 64 Ko (cible STM32N6)."""
    ram = model.estimate_ram_bytes("fp32")
    assert ram < 65_536, f"RAM estimée ({ram} B) > 64 Ko"


def test_ram_with_ewc_overhead(model):
    """Poids + Fisher + snapshot théorique (×3) reste < 64 Ko."""
    ram_model = model.estimate_ram_bytes("fp32")
    ram_total = ram_model * 3  # poids + Fisher diag + snapshot θ*
    assert ram_total < 65_536, f"RAM EWC ({ram_total} B) > 64 Ko"


# ------------------------------------------------------------------
# Tests EWC
# ------------------------------------------------------------------


def test_ewc_loss_task1_is_bce(model, dummy_batch):
    """Sur Task 1 (fisher=None), la perte EWC est identique à la perte BCE."""
    x, y = dummy_batch
    model.eval()  # désactive Dropout pour comparer deux forward passes identiques
    loss_ewc = model.ewc_loss(x, y, fisher=None, theta_star=None, ewc_lambda=1000.0)
    y_hat = model(x)
    loss_bce = torch.nn.functional.binary_cross_entropy(y_hat, y)
    assert abs(loss_ewc.item() - loss_bce.item()) < 1e-5


def test_ewc_loss_backprop(model, dummy_batch):
    """La perte EWC est rétropropageable."""
    x, y = dummy_batch
    fisher = {n: torch.ones_like(p) for n, p in model.named_parameters()}
    theta_star = model.get_theta_star()
    loss = model.ewc_loss(x, y, fisher=fisher, theta_star=theta_star, ewc_lambda=1000.0)
    loss.backward()
    for param in model.parameters():
        assert param.grad is not None, "Gradient manquant après backward()"


def test_no_gradient_leak_after_freezing(model, dummy_batch):
    """
    Simule le comportement MCU : vérifier que torch.no_grad() bloque les gradients.
    (Applicable lors d'une inférence pure sans mise à jour.)
    """
    x, y = dummy_batch
    x.requires_grad_(True)

    with torch.no_grad():
        out = model(x)

    assert out.grad_fn is None, "grad_fn doit être None dans torch.no_grad()"


# ------------------------------------------------------------------
# Tests sauvegarde/chargement
# ------------------------------------------------------------------


def test_save_load_state(model, dummy_batch, tmp_path):
    """Le modèle sauvegardé et rechargé produit les mêmes prédictions."""
    save_path = str(tmp_path / "ewc_model.pt")
    model.save_state(save_path)

    model2 = EWCMlpClassifier(input_dim=6, hidden_dims=[32, 16])
    model2.load_state(save_path)

    x, _ = dummy_batch
    model.eval()
    model2.eval()
    with torch.no_grad():
        out1 = model(x)
        out2 = model2(x)

    assert torch.allclose(out1, out2, atol=1e-6), "Prédictions divergent après chargement"


# ------------------------------------------------------------------
# Tests get_theta_star
# ------------------------------------------------------------------


def test_theta_star_values(model):
    """get_theta_star() retourne des valeurs identiques aux poids courants."""
    theta_star = model.get_theta_star()
    for name, param in model.named_parameters():
        assert torch.allclose(theta_star[name], param.detach()), (
            f"Divergence dans {name}"
        )


def test_theta_star_independence(model):
    """Modifier les poids après get_theta_star() ne doit pas altérer le snapshot."""
    theta_star = model.get_theta_star()
    original = {n: t.clone() for n, t in theta_star.items()}

    with torch.no_grad():
        for param in model.parameters():
            param.add_(1.0)

    for name in theta_star:
        assert torch.allclose(theta_star[name], original[name]), (
            f"Le snapshot {name} a été modifié après changement des poids"
        )


# ------------------------------------------------------------------
# S1-10 — Classes de tests (critères d'acceptation sprint)
# ------------------------------------------------------------------


class TestEWCMlpClassifier:
    """Tests conformes aux critères d'acceptation S1-10."""

    def test_forward_shape(self, model):
        """Le forward pass doit retourner [batch, 1] avec valeurs ∈ [0, 1]."""
        x = torch.randn(32, 6)
        out = model(x)
        assert out.shape == (32, 1), f"Shape attendue (32,1), obtenue {out.shape}"
        assert (out >= 0).all() and (out <= 1).all(), "Sigmoid doit borner [0,1]"

    def test_n_params(self, model):
        """769 paramètres attendus (conforme à ewc_mlp_spec.md §2)."""
        n = sum(p.numel() for p in model.parameters())
        assert n == 769, f"Attendu 769 params, obtenu {n}"

    def test_ewc_loss_task1_is_bce(self, model):
        """Sur Task 1 (fisher=None), la perte doit être pure BCE (positive et différentiable)."""
        x = torch.randn(16, 6)
        y = torch.randint(0, 2, (16, 1)).float()
        loss = model.ewc_loss(x, y, fisher=None, theta_star=None, ewc_lambda=1000.0)
        assert loss.item() > 0, "BCE doit être positive"
        assert loss.requires_grad, "La perte doit être différentiable"

    def test_ewc_loss_increases_with_lambda(self, model):
        """La régularisation EWC doit croître avec λ."""
        x = torch.randn(16, 6)
        y = torch.randint(0, 2, (16, 1)).float()
        fisher = {n: torch.ones_like(p) for n, p in model.named_parameters()}
        theta_star = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        loss_low = model.ewc_loss(x, y, fisher, theta_star, ewc_lambda=1.0)
        loss_high = model.ewc_loss(x, y, fisher, theta_star, ewc_lambda=10_000.0)
        assert loss_high.item() > loss_low.item(), "Perte EWC doit croître avec λ"

    def test_theta_star_detached(self, model):
        """Le snapshot θ* doit être détaché du graphe de calcul."""
        theta_star = model.get_theta_star()
        for name, tensor in theta_star.items():
            assert not tensor.requires_grad, f"{name} ne devrait pas avoir requires_grad"

    def test_backprop(self, model):
        """Un backward pass complet doit mettre à jour les gradients."""
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        x = torch.randn(8, 6)
        y = torch.randint(0, 2, (8, 1)).float()
        optimizer.zero_grad()
        loss = model.ewc_loss(x, y, None, None, 0.0)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "Les gradients doivent être calculés après backward()"


class TestFisherDiagonal:
    """Tests sur le calcul et la mise à jour de la Fisher diagonale."""

    def test_fisher_shape(self, model, synthetic_loader):
        """Fisher doit avoir la même structure de clés que model.named_parameters()."""
        from src.models.ewc.fisher import compute_fisher_diagonal
        loader = synthetic_loader()
        fisher = compute_fisher_diagonal(model, loader, device="cpu", n_samples=32)
        param_names = {n for n, _ in model.named_parameters()}
        assert set(fisher.keys()) == param_names

    def test_fisher_non_negative(self, model, synthetic_loader):
        """La Fisher diagonale doit être non-négative (carrés de gradients)."""
        from src.models.ewc.fisher import compute_fisher_diagonal
        loader = synthetic_loader()
        fisher = compute_fisher_diagonal(model, loader, device="cpu", n_samples=32)
        for name, f in fisher.items():
            assert (f >= 0).all(), f"Fisher[{name}] contient des valeurs négatives"

    def test_online_fisher_decay(self, model, synthetic_loader):
        """L'update Online doit interpoler : F_online = γ·F_old + F_new."""
        from src.models.ewc.fisher import compute_fisher_diagonal, update_fisher_online
        loader = synthetic_loader()
        f1 = compute_fisher_diagonal(model, loader, device="cpu", n_samples=32)
        f2 = compute_fisher_diagonal(model, loader, device="cpu", n_samples=32)
        f_online = update_fisher_online(f1, f2, gamma=0.9)
        for name in f1:
            expected = 0.9 * f1[name] + f2[name]
            assert torch.allclose(f_online[name], expected, atol=1e-5), (
                f"Decay incorrect pour {name}"
            )


class TestCLMetrics:
    """Tests sur compute_cl_metrics avec une matrice aux valeurs connues."""

    def test_aa_known_matrix(self, known_acc_matrix):
        """AA = moyenne de la dernière ligne (toutes tâches vues)."""
        from src.evaluation.metrics import compute_cl_metrics
        metrics = compute_cl_metrics(known_acc_matrix)
        expected_aa = (0.86 + 0.83 + 0.89) / 3
        assert abs(metrics["aa"] - expected_aa) < 1e-4, (
            f"AA attendu {expected_aa:.4f}, obtenu {metrics['aa']:.4f}"
        )

    def test_af_positive_for_forgetting(self, known_acc_matrix):
        """AF doit être positif (oubli) pour cette matrice."""
        from src.evaluation.metrics import compute_cl_metrics
        metrics = compute_cl_metrics(known_acc_matrix)
        assert metrics["af"] > 0, f"AF doit être > 0, obtenu {metrics['af']}"

    def test_bwt_negative_for_forgetting(self, known_acc_matrix):
        """BWT doit être négatif pour une matrice présentant de l'oubli."""
        from src.evaluation.metrics import compute_cl_metrics
        metrics = compute_cl_metrics(known_acc_matrix)
        assert metrics["bwt"] < 0, f"BWT doit être < 0, obtenu {metrics['bwt']}"

    def test_metrics_keys_present(self, known_acc_matrix):
        """Toutes les clés obligatoires doivent être présentes dans le dict résultat."""
        from src.evaluation.metrics import compute_cl_metrics
        metrics = compute_cl_metrics(known_acc_matrix)
        for key in ["aa", "af", "bwt", "fwt", "n_tasks", "acc_matrix"]:
            assert key in metrics, f"Clé manquante : {key}"

    def test_save_metrics_creates_file(self, tmp_path, known_acc_matrix):
        """save_metrics() doit créer le répertoire parent et le fichier JSON."""
        from src.evaluation.metrics import compute_cl_metrics, save_metrics
        metrics = compute_cl_metrics(known_acc_matrix)
        output = tmp_path / "results" / "metrics.json"
        save_metrics(metrics, str(output))
        assert output.exists(), "Le fichier JSON doit être créé"


class TestMemoryProfiler:
    """Smoke tests — RAM mesurée > 0 et budget STM32N6 respecté."""

    def test_forward_profile_non_zero(self, model):
        """Le pic RAM mesuré doit être > 0."""
        from src.evaluation.memory_profiler import profile_forward_pass
        result = profile_forward_pass(model, input_shape=(1, 6), n_runs=5)
        assert result["ram_peak_bytes"] > 0, "ram_peak_bytes doit être > 0"

    def test_ewc_within_budget(self, model):
        """EWCMlpClassifier doit respecter le budget 64 Ko (STM32N6)."""
        from src.evaluation.memory_profiler import profile_forward_pass
        result = profile_forward_pass(model, input_shape=(1, 6), n_runs=5)
        assert result["within_budget_64ko"], (
            f"RAM peak {result['ram_peak_bytes']} B dépasse 65 536 B"
        )

    def test_n_params_matches_model(self, model):
        """n_params dans le rapport doit correspondre au modèle réel."""
        from src.evaluation.memory_profiler import profile_forward_pass
        result = profile_forward_pass(model, input_shape=(1, 6), n_runs=5)
        expected = sum(p.numel() for p in model.parameters())
        assert result["n_params"] == expected, (
            f"n_params attendu {expected}, obtenu {result['n_params']}"
        )
