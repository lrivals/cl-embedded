"""Tests unitaires — modèles non supervisés (S5-08).

Couvre KMeansDetector, KNNDetector, PCABaseline.
Fixtures partagées : unsupervised_data, kmeans_config, knn_config, pca_config (conftest.py).

Critère : pytest tests/test_unsupervised.py -v → ≥ 12 tests, 0 skip, < 5 s.
"""

from __future__ import annotations

import numpy as np

from src.models.unsupervised import DBSCANDetector, KMeansDetector, KNNDetector, MahalanobisDetector, PCABaseline

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


# ===========================================================================
# MahalanobisDetector — 10 tests
# ===========================================================================


class TestMahalanobisDetector:

    def test_init_from_config(self, mahalanobis_config: dict) -> None:
        """MahalanobisDetector s'instancie correctement depuis la config."""
        model = MahalanobisDetector(mahalanobis_config["mahalanobis"])
        assert model is not None

    def test_fit_task_and_predict_binary(
        self, mahalanobis_config: dict, unsupervised_data: dict
    ) -> None:
        """fit_task puis predict retourne des prédictions binaires {0, 1}."""
        model = MahalanobisDetector(mahalanobis_config["mahalanobis"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        preds = model.predict(unsupervised_data["X_val"])
        assert preds.shape == (len(unsupervised_data["X_val"]),)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_anomaly_score_shape(
        self, mahalanobis_config: dict, unsupervised_data: dict
    ) -> None:
        """anomaly_score retourne un vecteur float32 de la bonne taille, valeurs ≥ 0."""
        model = MahalanobisDetector(mahalanobis_config["mahalanobis"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        scores = model.anomaly_score(unsupervised_data["X_val"])
        assert scores.shape == (len(unsupervised_data["X_val"]),)
        assert scores.dtype == np.float32
        assert np.all(scores >= 0), "Distance de Mahalanobis doit être non-négative"

    def test_score_between_0_and_1(
        self, mahalanobis_config: dict, unsupervised_data: dict
    ) -> None:
        """score() retourne une accuracy dans [0, 1]."""
        model = MahalanobisDetector(mahalanobis_config["mahalanobis"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        acc = model.score(unsupervised_data["X_val"], unsupervised_data["y_val"])
        assert 0.0 <= acc <= 1.0

    def test_anomalies_have_higher_score(
        self, mahalanobis_config: dict, unsupervised_data: dict
    ) -> None:
        """Les anomalies N(5,1) ont un score moyen supérieur aux normaux N(0,1)."""
        model = MahalanobisDetector(mahalanobis_config["mahalanobis"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        scores = model.anomaly_score(unsupervised_data["X_val"])
        y = unsupervised_data["y_val"]
        assert scores[y == 1].mean() > scores[y == 0].mean()

    def test_sequential_fit_two_tasks(
        self, mahalanobis_config: dict, unsupervised_data: dict
    ) -> None:
        """fit_task séquentiel sur 2 tâches — threshold inchangé après task_id=1."""
        model = MahalanobisDetector(mahalanobis_config["mahalanobis"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        threshold_after_task0 = model.threshold_
        model.fit_task(unsupervised_data["X_train"], task_id=1)
        assert model.threshold_ == threshold_after_task0, (
            "Le seuil ne doit pas être recalculé après Task 0"
        )
        preds = model.predict(unsupervised_data["X_val"])
        assert preds.shape == (len(unsupervised_data["X_val"]),)

    def test_summary_returns_string(
        self, mahalanobis_config: dict, unsupervised_data: dict
    ) -> None:
        """summary() retourne une chaîne non vide après fit_task."""
        model = MahalanobisDetector(mahalanobis_config["mahalanobis"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        s = model.summary()
        assert isinstance(s, str) and len(s) > 0

    def test_count_parameters_equals_d_plus_d2(
        self, mahalanobis_config: dict, unsupervised_data: dict
    ) -> None:
        """count_parameters() retourne d + d² (pour d=4 → 20)."""
        model = MahalanobisDetector(mahalanobis_config["mahalanobis"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        d = unsupervised_data["X_train"].shape[1]  # 4
        assert model.count_parameters() == d + d * d

    def test_save_and_load(
        self, mahalanobis_config: dict, unsupervised_data: dict, tmp_path
    ) -> None:
        """save() puis load() (classmethod) préserve les prédictions."""
        model = MahalanobisDetector(mahalanobis_config["mahalanobis"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        preds_before = model.predict(unsupervised_data["X_val"])

        checkpoint = tmp_path / "mahalanobis_test.pkl"
        model.save(checkpoint)

        model2 = MahalanobisDetector.load(checkpoint)
        np.testing.assert_array_equal(preds_before, model2.predict(unsupervised_data["X_val"]))

    def test_ram_bytes_d4(
        self, mahalanobis_config: dict, unsupervised_data: dict
    ) -> None:
        """_estimate_ram_bytes() == 80 pour d=4 (= (4 + 16) × 4 octets @ FP32)."""
        model = MahalanobisDetector(mahalanobis_config["mahalanobis"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        assert model._estimate_ram_bytes() == 80, (
            f"Attendu 80 B pour d=4, obtenu {model._estimate_ram_bytes()} B"
        )


# ===========================================================================
# DBSCANDetector — 10 tests
# ===========================================================================


class TestDBSCANDetector:

    def test_init_from_config(self, dbscan_config: dict) -> None:
        """DBSCANDetector s'instancie correctement depuis la config."""
        model = DBSCANDetector(dbscan_config["dbscan"])
        assert model is not None
        assert model.eps == dbscan_config["dbscan"]["EPSILON"]
        assert model.min_samples == dbscan_config["dbscan"]["MIN_SAMPLES"]

    def test_fit_task_and_predict_binary(
        self, dbscan_config: dict, unsupervised_data: dict
    ) -> None:
        """fit_task puis predict retourne des prédictions binaires {0, 1}."""
        model = DBSCANDetector(dbscan_config["dbscan"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        preds = model.predict(unsupervised_data["X_val"])
        assert preds.shape == (len(unsupervised_data["X_val"]),)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_anomaly_score_shape(self, dbscan_config: dict, unsupervised_data: dict) -> None:
        """anomaly_score retourne un vecteur float32 de la bonne taille, valeurs ≥ 0."""
        model = DBSCANDetector(dbscan_config["dbscan"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        scores = model.anomaly_score(unsupervised_data["X_val"])
        assert scores.shape == (len(unsupervised_data["X_val"]),)
        assert scores.dtype == np.float32
        assert np.all(scores >= 0), "Score DBSCAN doit être non-négatif"

    def test_score_between_0_and_1(self, dbscan_config: dict, unsupervised_data: dict) -> None:
        """score() retourne une accuracy dans [0, 1]."""
        model = DBSCANDetector(dbscan_config["dbscan"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        acc = model.score(unsupervised_data["X_val"], unsupervised_data["y_val"])
        assert 0.0 <= acc <= 1.0

    def test_anomalies_have_higher_score(
        self, dbscan_config: dict, unsupervised_data: dict
    ) -> None:
        """Les anomalies N(5,1) ont un score moyen supérieur aux normaux N(0,1).

        DBSCAN entraîné sur normaux N(0,1) → core points proches de 0.
        Anomalies N(5,1) sont distantes des core points → score élevé.
        """
        cfg = dict(dbscan_config["dbscan"])
        model = DBSCANDetector(cfg)
        x_normal = unsupervised_data["X_train"][unsupervised_data["y_train"] == 0]
        model.fit_task(x_normal, task_id=0)
        scores = model.anomaly_score(unsupervised_data["X_val"])
        y = unsupervised_data["y_val"]
        assert scores[y == 1].mean() > scores[y == 0].mean()

    def test_sequential_fit_two_tasks(self, dbscan_config: dict, unsupervised_data: dict) -> None:
        """fit_task séquentiel sur 2 tâches — threshold inchangé après task_id=1."""
        model = DBSCANDetector(dbscan_config["dbscan"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        threshold_after_task0 = model.threshold_
        model.fit_task(unsupervised_data["X_train"], task_id=1)
        assert model.threshold_ == threshold_after_task0, (
            "Le seuil ne doit pas être recalculé après Task 0"
        )
        preds = model.predict(unsupervised_data["X_val"])
        assert preds.shape == (len(unsupervised_data["X_val"]),)

    def test_summary_returns_string(self, dbscan_config: dict, unsupervised_data: dict) -> None:
        """summary() retourne une chaîne non vide après fit_task."""
        model = DBSCANDetector(dbscan_config["dbscan"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        s = model.summary()
        assert isinstance(s, str) and len(s) > 0

    def test_count_parameters_positive(self, dbscan_config: dict, unsupervised_data: dict) -> None:
        """count_parameters() retourne un entier > 0 après fit_task."""
        model = DBSCANDetector(dbscan_config["dbscan"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        n = model.count_parameters()
        assert isinstance(n, int) and n > 0

    def test_save_and_load(self, dbscan_config: dict, unsupervised_data: dict, tmp_path) -> None:
        """save() puis load() (classmethod) préserve les prédictions."""
        model = DBSCANDetector(dbscan_config["dbscan"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        preds_before = model.predict(unsupervised_data["X_val"])

        checkpoint = tmp_path / "dbscan_test.pkl"
        model.save(checkpoint)

        model2 = DBSCANDetector.load(checkpoint)
        np.testing.assert_array_equal(preds_before, model2.predict(unsupervised_data["X_val"]))

    def test_ram_budget(self, dbscan_config: dict, unsupervised_data: dict) -> None:
        """count_parameters() × 4 bytes ≤ 64 Ko pour cl_strategy=refit (contrainte STM32N6)."""
        model = DBSCANDetector(dbscan_config["dbscan"])
        model.fit_task(unsupervised_data["X_train"], task_id=0)
        ram_fp32 = model.count_parameters() * 4
        assert ram_fp32 <= 65536, f"DBSCAN (refit) dépasse 64 Ko : {ram_fp32} B"
