"""Tests unitaires — modèles non supervisés (S5-08).

Couvre KMeansDetector, KNNDetector, PCABaseline.
Fixtures partagées : unsupervised_data, kmeans_config, knn_config, pca_config (conftest.py).

Critère : pytest tests/test_unsupervised.py -v → ≥ 12 tests, 0 skip, < 5 s.
"""

from __future__ import annotations

import numpy as np

from src.models.unsupervised import KMeansDetector, KNNDetector, PCABaseline

# ===========================================================================
# KMeansDetector — 10 tests
# ===========================================================================


class TestKMeansDetector:

    def test_init_from_config(self, kmeans_config: dict) -> None:
        """KMeansDetector s'instancie correctement depuis la config."""
        model = KMeansDetector(kmeans_config["kmeans"])
        assert model is not None

    def test_fit_task_and_predict_binary(
        self, kmeans_config: dict, unsupervised_data: dict
    ) -> None:
        """fit_task puis predict retourne des prédictions binaires {0, 1}."""
        model = KMeansDetector(kmeans_config["kmeans"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        preds = model.predict(unsupervised_data["X_val"])
        assert preds.shape == (len(unsupervised_data["X_val"]),)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_anomaly_score_shape(self, kmeans_config: dict, unsupervised_data: dict) -> None:
        """anomaly_score retourne un vecteur de la bonne taille, valeurs ≥ 0."""
        model = KMeansDetector(kmeans_config["kmeans"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        scores = model.anomaly_score(unsupervised_data["X_val"])
        assert scores.shape == (len(unsupervised_data["X_val"]),)
        assert np.all(scores >= 0), "Score d'anomalie doit être non-négatif"

    def test_score_between_0_and_1(self, kmeans_config: dict, unsupervised_data: dict) -> None:
        """score() retourne une accuracy dans [0, 1]."""
        model = KMeansDetector(kmeans_config["kmeans"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        acc = model.score(unsupervised_data["X_val"], unsupervised_data["y_val"])
        assert 0.0 <= acc <= 1.0

    def test_anomalies_have_higher_score(
        self, kmeans_config: dict, unsupervised_data: dict
    ) -> None:
        """Les anomalies N(5,1) ont un score moyen supérieur aux normaux N(0,1)."""
        model = KMeansDetector(kmeans_config["kmeans"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        scores = model.anomaly_score(unsupervised_data["X_val"])
        y = unsupervised_data["y_val"]
        assert scores[y == 1].mean() > scores[y == 0].mean()

    def test_sequential_fit_two_tasks(self, kmeans_config: dict, unsupervised_data: dict) -> None:
        """fit_task séquentiel sur 2 tâches ne lève pas d'erreur."""
        model = KMeansDetector(kmeans_config["kmeans"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        model.fit_task(unsupervised_data["X_train"], task_id=1)
        preds = model.predict(unsupervised_data["X_val"])
        assert preds.shape == (len(unsupervised_data["X_val"]),)

    def test_summary_returns_string(self, kmeans_config: dict, unsupervised_data: dict) -> None:
        """summary() retourne une chaîne non vide après fit_task."""
        model = KMeansDetector(kmeans_config["kmeans"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        s = model.summary()
        assert isinstance(s, str) and len(s) > 0

    def test_count_parameters_positive(self, kmeans_config: dict, unsupervised_data: dict) -> None:
        """count_parameters() retourne un entier > 0 après fit_task."""
        model = KMeansDetector(kmeans_config["kmeans"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        n = model.count_parameters()
        assert isinstance(n, int) and n > 0

    def test_save_and_load(self, kmeans_config: dict, unsupervised_data: dict, tmp_path) -> None:
        """save() puis load() (classmethod) préserve les prédictions."""
        model = KMeansDetector(kmeans_config["kmeans"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        preds_before = model.predict(unsupervised_data["X_val"])

        checkpoint = tmp_path / "kmeans_test.pkl"
        model.save(checkpoint)

        model2 = KMeansDetector.load(checkpoint)
        preds_after = model2.predict(unsupervised_data["X_val"])

        np.testing.assert_array_equal(preds_before, preds_after)

    def test_ram_budget(self, kmeans_config: dict, unsupervised_data: dict) -> None:
        """count_parameters() × 4 bytes ≤ 64 Ko (contrainte STM32N6)."""
        model = KMeansDetector(kmeans_config["kmeans"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        ram_fp32 = model.count_parameters() * 4
        assert ram_fp32 <= 65536, f"KMeans dépasse 64 Ko : {ram_fp32} B"


# ===========================================================================
# KNNDetector — 5 tests
# ===========================================================================


class TestKNNDetector:

    def test_init_from_config(self, knn_config: dict) -> None:
        """KNNDetector s'instancie correctement depuis la config."""
        model = KNNDetector(knn_config["knn"])
        assert model is not None

    def test_fit_and_predict(self, knn_config: dict, unsupervised_data: dict) -> None:
        """fit_task puis predict retourne des prédictions binaires {0, 1}."""
        model = KNNDetector(knn_config["knn"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        preds = model.predict(unsupervised_data["X_val"])
        assert preds.shape == (len(unsupervised_data["X_val"]),)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_anomaly_score_monotone(self, knn_config: dict, unsupervised_data: dict) -> None:
        """Les anomalies N(5,1) ont un score de distance moyen supérieur aux normaux."""
        model = KNNDetector(knn_config["knn"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        scores = model.anomaly_score(unsupervised_data["X_val"])
        y = unsupervised_data["y_val"]
        assert scores[y == 1].mean() > scores[y == 0].mean()

    def test_accumulate_strategy_grows_ref(self, knn_config: dict, unsupervised_data: dict) -> None:
        """Avec cl_strategy=accumulate, X_ref_ grandit après chaque tâche."""
        model = KNNDetector(knn_config["knn"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        n_ref_0 = len(model.X_ref_)
        model.fit_task(unsupervised_data["X_train"], task_id=1)
        n_ref_1 = len(model.X_ref_)
        assert n_ref_1 >= n_ref_0, "cl_strategy=accumulate doit conserver les données passées"

    def test_save_load(self, knn_config: dict, unsupervised_data: dict, tmp_path) -> None:
        """save() puis load() (classmethod) préserve les prédictions."""
        model = KNNDetector(knn_config["knn"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        preds_before = model.predict(unsupervised_data["X_val"])

        checkpoint = tmp_path / "knn_test.pkl"
        model.save(checkpoint)

        model2 = KNNDetector.load(checkpoint)
        np.testing.assert_array_equal(preds_before, model2.predict(unsupervised_data["X_val"]))


# ===========================================================================
# PCABaseline — 6 tests
# ===========================================================================


class TestPCABaseline:

    def test_init_from_config(self, pca_config: dict) -> None:
        """PCABaseline s'instancie correctement depuis la config."""
        model = PCABaseline(pca_config["pca"])
        assert model is not None

    def test_fit_and_predict(self, pca_config: dict, unsupervised_data: dict) -> None:
        """fit_task puis predict retourne des prédictions binaires {0, 1}."""
        model = PCABaseline(pca_config["pca"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        preds = model.predict(unsupervised_data["X_val"])
        assert preds.shape == (len(unsupervised_data["X_val"]),)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_reconstruction_error_higher_for_anomalies(
        self, pca_config: dict, unsupervised_data: dict
    ) -> None:
        """PCA entraîné sur normaux uniquement : erreur de reconstruction plus haute pour anomalies.

        Usage standard de PCA pour anomaly detection : fit sur données normales seulement.
        Entraîner sur données mixtes ferait capturer la direction N(0)→N(5) par PCA,
        réduisant l'erreur de reconstruction des anomalies.
        """
        model = PCABaseline(pca_config["pca"])
        x_normal = unsupervised_data["X_train"][unsupervised_data["y_train"] == 0]
        model.fit_task(x_normal, task_id=0)
        scores = model.anomaly_score(unsupervised_data["X_val"])
        y = unsupervised_data["y_val"]
        assert scores[y == 1].mean() > scores[y == 0].mean()

    def test_n_components_respected(self, pca_config: dict, unsupervised_data: dict) -> None:
        """Le modèle PCA utilise exactement n_components composantes."""
        model = PCABaseline(pca_config["pca"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        assert model.pca_.n_components_ == pca_config["pca"]["n_components"]

    def test_save_load(self, pca_config: dict, unsupervised_data: dict, tmp_path) -> None:
        """save() puis load() (classmethod) préserve les prédictions."""
        model = PCABaseline(pca_config["pca"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        preds_before = model.predict(unsupervised_data["X_val"])

        checkpoint = tmp_path / "pca_test.pkl"
        model.save(checkpoint)

        model2 = PCABaseline.load(checkpoint)
        np.testing.assert_array_equal(preds_before, model2.predict(unsupervised_data["X_val"]))

    def test_ram_budget(self, pca_config: dict, unsupervised_data: dict) -> None:
        """count_parameters() × 4 bytes ≤ 64 Ko (contrainte STM32N6)."""
        model = PCABaseline(pca_config["pca"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        ram_fp32 = model.count_parameters() * 4
        assert ram_fp32 <= 65536, f"PCA dépasse 64 Ko : {ram_fp32} B"
