# -*- coding: utf-8 -*-
"""
Created on Sun May  1 11:41:11 2022

@author: mudar
"""

# Regresion Lineal Simple

# Como importar las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
dataset = pd.read_csv('data/Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


# Dividir el dataset en conjunto de entrenamiento y conjunto de testing
# En este caso se ha cogido como conjunto de test 1/3 lo cual no es lo habitual
# y recomendable pero para este caso tan simple funciona correctamente
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)


# En caso de regresion lineal simple no hace falta hacer escalado de variables
# porque la libreria de python que se utiliza ya lo implementa

# Escalado de variables
# Para que no haya tanta diferencia en los rangos de
# valores en los datos de las distintas columnas
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""


# Crear modelo de Regresion Lineal Simple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

# Predecir el conjunto de test
y_pred = regression.predict(X_test)

# Visualizar los resultados de entrenamiento
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regression.predict(X_train), color = 'blue')
plt.title('Sueldo vs Años de Experiencia (Conjunto de Entrenamiento)')
plt.xlabel('Años de experiencia')
plt.ylabel('Sueldo (en $)')
plt.show()

# Visualizar los resultados de test
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regression.predict(X_train), color = 'blue')
plt.title('Sueldo vs Años de Experiencia (Conjunto de Testing)')
plt.xlabel('Años de experiencia')
plt.ylabel('Sueldo (en $)')
plt.show()






