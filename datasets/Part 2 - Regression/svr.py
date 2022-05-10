# -*- coding: utf-8 -*-
"""
Created on Sun May  8 00:28:32 2022

@author: mudar
"""

# SVR

# Como importar las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
dataset = pd.read_csv('data/Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values # para que no sea un vector y sea una matriz ponemos 1:2
y = dataset.iloc[:, 2].values


# Dividir el dataset en conjunto de entrenamiento y conjunto de testing
# 80% conjunto entrenamiento y 20% conjunto de testing
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
"""


# Escalado de variables
# Para que no haya tanta diferencia en los rangos de
# valores en los datos de las distintas columnas
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1, 1))


# Ajustar la Regresion con todo el dataset
from sklearn.svm import SVR
regression = SVR(kernel = 'rbf')
regression.fit(X, y)


# Prediccion de nuestros modelos con SVR
y_pred = sc_y.inverse_transform(regression.predict(sc_X.transform(np.array([[6.5]]))).reshape(1, -1))


# Visualizacion de los resultados del SVR
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regression.predict(X_grid), color = 'blue')
plt.title('Modelo de Regresion (SVR)')
plt.xlabel('Posicion del empleado')
plt.ylabel('Sueldo (en $)')
plt.show()

# Ejercicio propuesto reescalado de los ejes
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_X.inverse_transform(X_grid), sc_y.inverse_transform(regression.predict(X_grid).reshape(-1, 1)), color = 'blue')
plt.title('Modelo de Regresion (SVR)')
plt.xlabel('Posicion del empleado')
plt.ylabel('Sueldo (en $)')
plt.show()
