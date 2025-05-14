import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np

# https://www.ncei.noaa.gov/products/natural-hazards/tsunamis-earthquakes-volcanoes/earthquakes/intensity-database-1638-1985 
df = pd.read_csv("eqint_tsqp.csv")[["LONGITUDE", "LATITUDE"]]
df = df.dropna()

X=df.iloc[:, 1:].to_numpy()
X = df.values

#compute K-means
inertia = np.zeros(14)
for i in range(1, 15):
    kmeans = KMeans(n_clusters=i).fit(X)
    inertia[i-1] = kmeans.inertia_

#find the Elbow
plt.plot(np.arange(1, 15), inertia)
plt.show()
plt.style.use('dark_background')

#Performs K means clustering on the data
kmeans = KMeans(n_clusters=5).fit(X)

fig, ax = plt.subplots()

#creates the Voronor edges
vor = Voronoi(kmeans.cluster_centers_)
voronoi_plot_2d(vor, show_vertices=False, show_points=False, line_colors='white', line_width=1.0, ax=ax)

#plots clusters and data points
plt.scatter(X[:,0], X[:,1], c=kmeans.labels_,s=10)
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],c='r')
plt.show()

