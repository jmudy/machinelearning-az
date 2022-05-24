# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 12:44:07 2022

@author: mudar
"""

# Plantilla de Pre Procesado - Datos categoricos

# Como importar las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
dataset = pd.read_csv('data/Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Codificar datos categóricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

# Como hay mas de dos datos categoricos en X hay que usar un one hot encoder
# y crear variables dummy
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],   
    remainder='passthrough'                        
)

X = np.array(ct.fit_transform(X), dtype=np.float)

# En y los unicos datos categoricos son 'yes' y 'no' por lo que no hace falta
# usar un one hoy encoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
