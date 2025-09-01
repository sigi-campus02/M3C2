"""Visualize distance distributions including various outlier filtering methods.

The script reads distance files for different processing variants and produces
boxplots comparing all distances with several inlier-selection techniques.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

# Base directory containing distance files for the selected scan pair.
base = "data/0342-0349/"

# Distances are evaluated for each variant, typically a reference processing
# chain (e.g., classical reference versus AI-assisted reference).
variants = [
    ("ref", "python_ref_m3c2_distances.txt"),
    ("ref_ai", "python_ref_ai_m3c2_distances.txt"),
]
# Different strategies for computing inliers that exclude outliers.
inlier_suffixes = [
    ("std", "Inlier _STD"),
    ("rmse", "Inlier _RMSE"),
    ("nmad", "Inlier _NMAD"),
    ("iqr", "Inlier _IQR"),
]

for variant, dist_file in variants:
    data_list = []
    labels = []

    # Load the complete distance distribution for the current variant.
    file_path = base + dist_file
    try:
        data = np.loadtxt(file_path)
        if data.ndim == 0 or data.size == 0:
            data = np.array([])
        else:
            # Remove potential NaN values before plotting.
            data = data[~np.isnan(data)]
        if data.size > 0:
            data_list.append(data)
            labels.append("All Distances")
    except Exception:
        # Failure to load the file is silently ignored to allow partial plots.
        pass

    # Load and append each variant of the inlier-filtered distance set.
    for suffix, label in inlier_suffixes:
        file_path = f"{base}python_{variant}_m3c2_distances_coordinates_inlier_{suffix}.txt"
        try:
            arr = np.loadtxt(file_path, skiprows=1)
            data = arr[:, -1]
            data = data[~np.isnan(data)]
            if data.size > 0:
                data_list.append(data)
                labels.append(label)
        except Exception:
            # Ignore missing files for robustness.
            pass

    # Render a boxplot comparing all collected distributions.
    plt.figure(figsize=(8, 6))
    plt.boxplot(data_list, labels=labels)
    plt.ylabel("Distanz")
    plt.title(f"Vergleich der Distanzverteilungen (0342-0349) {variant}")
    plt.grid(True)
    plt.tight_layout()

    outdir = os.path.join("outputs", "MARS_output", "Plots_MARS_Outlier")
    os.makedirs(outdir, exist_ok=True)
    basename = base.strip("/").split("/")[-1]  # yields "0342-0349"
    plt.savefig(os.path.join(outdir, f"{basename}_OutlierComparison_{variant}.png"))
    plt.close()

