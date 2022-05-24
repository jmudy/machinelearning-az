# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 12:44:10 2022

@author: mudar
"""

# Plantilla de Pre Procesado - Datos faltantes

# Como importar las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
dataset = pd.read_csv('data/Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Tratamiento de los NAs
from sklearn.impute import SimpleImputer

# Sustituir los valores nan por la media de la columna
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
