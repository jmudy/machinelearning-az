# -*- coding: utf-8 -*-
"""
Created on Sun May  8 12:54:08 2022

@author: mudar
"""

# Regresion con Arboles de Decision

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
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""


# Ajustar la Regresion con todo el dataset
from sklearn.tree import DecisionTreeRegressor
regression = DecisionTreeRegressor(random_state = 0)
regression.fit(X, y)


# Prediccion de nuestro modelo de regresion con Arboles de Decision
y_pred = regression.predict([[6.5]])


# Visualizacion de los resultados del Modelo
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X, regression.predict(X), color = 'blue')
plt.title('Modelo de Regresion con Arboles de Decision')
plt.xlabel('Posicion del empleado')
plt.ylabel('Sueldo (en $)')
plt.show()
