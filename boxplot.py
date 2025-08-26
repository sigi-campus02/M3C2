import numpy as np
import matplotlib.pyplot as plt

# 1. Einspaltige Datei laden und NaNs entfernen
data1 = np.loadtxt("data/0342-0349/python_ref_m3c2_distances.txt")
if data1.ndim == 0 or data1.size == 0:
    data1 = np.array([])
else:
    data1 = data1[~np.isnan(data1)]

# 2. Mehrspaltige Dateien laden, Distanz ist letzte Spalte
arr2 = np.loadtxt("data/0342-0349/python_ref_m3c2_distances_coordinates_inlier_3sigma.txt", skiprows=1)
data2 = arr2[:, -1]
data2 = data2[~np.isnan(data2)]

arr3 = np.loadtxt("data/0342-0349/python_ref_m3c2_distances_coordinates_inlier_1rms.txt", skiprows=1)
data3 = arr3[:, -1]
data3 = data3[~np.isnan(data3)]

arr4 = np.loadtxt("data/0342-0349/python_ref_m3c2_distances_coordinates_inlier_2rms.txt", skiprows=1)
data4 = arr4[:, -1]
data4 = data4[~np.isnan(data4)]

arr5 = np.loadtxt("data/0342-0349/python_ref_m3c2_distances_coordinates_inlier_3rms.txt", skiprows=1)
data5 = arr5[:, -1]
data5 = data5[~np.isnan(data5)]


# Nur Boxplots für vorhandene Daten erzeugen
data_list = []
labels = []
if data1.size > 0:
    data_list.append(data1)
    labels.append("All Distances")
if data2.size > 0:
    data_list.append(data2)
    labels.append("Inlier _3sigma")
if data3.size > 0:
    data_list.append(data3)
    labels.append("Inlier _1rms")
if data4.size > 0:
    data_list.append(data4)
    labels.append("Inlier _2rms")
if data5.size > 0:
    data_list.append(data5)
    labels.append("Inlier _3rms")

plt.figure(figsize=(8, 6))
plt.boxplot(data_list, labels=labels)
plt.ylabel("Distanz")
plt.title("Vergleich der Distanzverteilungen (0342-0349) ref")
plt.grid(True)
plt.tight_layout()
plt.show()



# REF AI ---------------------


# 1. Einspaltige Datei laden und NaNs entfernen
data1 = np.loadtxt("data/0342-0349/python_ref_ai_m3c2_distances.txt")
if data1.ndim == 0 or data1.size == 0:
    data1 = np.array([])
else:
    data1 = data1[~np.isnan(data1)]

# 2. Mehrspaltige Dateien laden, Distanz ist letzte Spalte
arr2 = np.loadtxt("data/0342-0349/python_ref_ai_m3c2_distances_coordinates_inlier_3sigma.txt", skiprows=1)
data2 = arr2[:, -1]
data2 = data2[~np.isnan(data2)]

arr3 = np.loadtxt("data/0342-0349/python_ref_ai_m3c2_distances_coordinates_inlier_1rms.txt", skiprows=1)
data3 = arr3[:, -1]
data3 = data3[~np.isnan(data3)]



# Nur Boxplots für vorhandene Daten erzeugen
data_list = []
labels = []
if data1.size > 0:
    data_list.append(data1)
    labels.append("All Distances")
if data2.size > 0:
    data_list.append(data2)
    labels.append("Inlier _3sigma")
if data3.size > 0:
    data_list.append(data3)
    labels.append("Inlier _1rms")


plt.figure(figsize=(8, 6))
plt.boxplot(data_list, labels=labels)
plt.ylabel("Distanz")
plt.title("Vergleich der Distanzverteilungen (0342-0349) ref_ai")
plt.grid(True)
plt.tight_layout()
plt.show()