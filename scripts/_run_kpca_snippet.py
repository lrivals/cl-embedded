"""Script temporaire pour exécuter le snippet KPCA RBF."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.feature_space_plots import fit_projection, plot_feature_space_2d

CSV = Path("data/raw/equipment_monitoring/Industrial_Equipment_Monitoring_Dataset/equipment_anomaly_data.csv")
df = pd.read_csv(CSV)

FEATURES = ["temperature", "pressure", "vibration", "humidity"]
X = df[FEATURES].values.astype(np.float32)
y = df["faulty"].values.astype(int)

# Normalisation Z-score simple
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

fig, ax = plt.subplots(figsize=(8, 6))

model, X_proj, xlabel, ylabel = fit_projection(X, method="kpca_rbf", gamma=0.5)
plot_feature_space_2d(X_proj, y, "Kernel PCA RBF", ax, xlabel=xlabel, ylabel=ylabel)

out = Path("notebooks/figures/model_viz/kpca_rbf_snippet.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(out, dpi=150)
print(f"Saved → {out}")
