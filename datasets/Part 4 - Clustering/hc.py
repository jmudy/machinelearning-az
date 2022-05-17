# -*- coding: utf-8 -*-
"""
Created on Tue May 17 10:58:24 2022

@author: mudar
"""

# Clustering Jerarquico

# Importar las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Cargamos los datos del centro comercial con pandas
dataset = pd.read_csv('data/Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# Utilizar el dendrograma para encontrar el numero optimo de clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrograma')
plt.xlabel('Clientes')
plt.ylabel('Distancia Euclidea')
plt.show()

# Ajustar el clustering jerarquico a nuestro conjunto de datos
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

# Visualizacion de los clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, color = 'red', label = 'Cautos')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, color = 'blue', label = 'Estandar')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, color = 'green', label = 'Objetivo')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, color = 'cyan', label = 'Descuidados')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, color = 'magenta', label = 'Conservadores')
plt.title('Cluster de clientes')
plt.xlabel('Ingresos anuales (en miles de $)')
plt.ylabel('Puntuacion de los Gastos (1-100)')
plt.legend()
plt.show()
