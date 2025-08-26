import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

def exclude_outliers(data_folder, ref_variant, outlier_rmse_multiplicator=3):
	arr = np.loadtxt(f"{data_folder}/python_{ref_variant}_m3c2_distances_coordinates.txt", skiprows=1)
	# arr.shape: (N, 4), Spalten: x, y, z, distance

	# Nur gültige Werte (ohne NaN)
	mask_valid = ~np.isnan(arr[:, 3])
	arr_valid = arr[mask_valid]
	distances_valid = arr_valid[:, 3]

	rms = np.sqrt(np.mean(distances_valid ** 2))
	outlier_mask = np.abs(distances_valid) > (outlier_rmse_multiplicator * rms)
	
	arr_excl_outlier = arr_valid[~outlier_mask]

	# Exportiere die Inlier (ohne Outlier) als TXT
	out_path_inlier = os.path.join(data_folder, f"python_{ref_variant}_m3c2_distances_coordinates_inlier.txt")
	header = "x y z distance"
	np.savetxt(out_path_inlier, arr_excl_outlier, fmt="%.6f", header=header)

	# Exportiere nur die Outlier (arr_valid[outlier_mask])
	out_path_outlier = os.path.join(data_folder, f"python_{ref_variant}_m3c2_distances_coordinates_outlier.txt")
	np.savetxt(out_path_outlier, arr_valid[outlier_mask], fmt="%.6f", header=header)

	logger.info(f"[Exclude Outliers] Gesamt: {arr.shape[0]}")
	logger.info(f"[Exclude Outliers] NaN: {(np.isnan(arr[:, 3])).sum()}")
	logger.info(f"[Exclude Outliers] Valid (ohne NaN): {arr_valid.shape[0]}")
	logger.info(f"[Exclude Outliers] Outlier: {arr_valid[outlier_mask].shape[0]}")
	logger.info(f"[Exclude Outliers] Inlier: {arr_excl_outlier.shape[0]}")
	logger.info(f"[Exclude Outliers] Inlier (ohne Outlier) gespeichert: {out_path_inlier}")
	logger.info(f"[Exclude Outliers] Outlier gespeichert: {out_path_outlier}")

