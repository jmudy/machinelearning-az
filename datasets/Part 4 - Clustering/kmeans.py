# -*- coding: utf-8 -*-
"""
Created on Mon May 16 19:46:25 2022

@author: mudar
"""

# K-Means

# Importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Cargamos los datos con pandas
dataset = pd.read_csv('data/Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# Metodo del codo para avergiuar el numero optimo de clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Metodo del codo')
plt.xlabel('umero de Clusters')
plt.ylabel('WCSS(k)')
plt.show()

# Aplicar el metodo de k-means para segmentar el dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)

# Visualizacion de los clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, color = 'red', label = 'Cautos')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, color = 'blue', label = 'Estandar')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, color = 'green', label = 'Objetivo')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, color = 'cyan', label = 'Descuidados')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, color = 'magenta', label = 'Conservadores')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, color = 'yellow', label = 'Baricentros')
plt.title('Cluster de clientes')
plt.xlabel('Ingresos anuales (en miles de $)')
plt.ylabel('Puntuacion de los Gastos (1-100)')
plt.legend()
plt.show()
