# -*- coding: utf-8 -*-
"""
Created on Thu May  5 11:00:48 2022

@author: mudar
"""

# Regresion Polinomica

# Como importar las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
dataset = pd.read_csv('data/Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values # para que no sea un vector y sea una matriz ponemos 1:2
y = dataset.iloc[:, 2].values


# En este caso no vamos a dividir el conjunto de entrenamiento en test train y
# y test dada la escasa cantidad de datos

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


# Ajustar la Regresion Lineal con todo el dataset para comparar
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Ajustar la Regresion Polinomica con todo el dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)


# Visualizacion de los resultados del Modelo Lineal
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Modelo de Regresion Lineal')
plt.xlabel('Posicion del empleado')
plt.ylabel('Sueldo (en $)')
plt.show()

# Visualizacion de los resultados del Modelo Polinomico
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Modelo de Regresion Polinomica')
plt.xlabel('Posicion del empleado')
plt.ylabel('Sueldo (en $)')
plt.show()


# Prediccion de nuestros modelos
# En el siguiente caso con el modelo lineal sale un valor irreal
lin_reg.predict([[6.5]])

# Con el modelo polinomico si que se ajusta mas a los datos
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))

