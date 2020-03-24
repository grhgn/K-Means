import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

#meangambil data dari csv dan mengubahnya menjadi numpy array
data = pd.read_csv("Dataset.csv")
x_array = np.array(data)

#normalisasi data dengan menggunakan MinMax
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x_array)

#set Kmenas dengan jumlah cluster 8 karena dilihat dari datasat, dataset berkemungkinan memiliki 8 cluster
kmeans = KMeans(n_clusters = 8, random_state=123)
kmeans.fit(x_scaled)
#memberikan setiap data labelnya masing-masing
data["kluster"] = kmeans.labels_

#memvisualisasikan hasilnya
output = plt.scatter(x_scaled[:,0], x_scaled[:,1], s = 100, c = data.kluster, marker = "o", alpha = 1, )
centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c="red", s=200, alpha=1 , marker="o")
plt.title("Hasil Klustering K-Means")
plt.colorbar (output)
plt.show()
