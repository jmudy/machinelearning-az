# -*- coding: utf-8 -*-
"""
Created on Fri May 27 00:01:38 2022

@author: mudar
"""

# Redes Neuronales Convolucionales

# Instalar Theano, Keras y Tensorflow
#pip install Theano
#conda install -c conda-forge keras
#pip install tensorflow


# Parte 1 - Preprocesado de datos

# Como importar las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
dataset = pd.read_csv('data/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Codificar datos categóricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

# Como hay mas de dos datos categoricos en en la columna de los paises hay que
# usar un one hot encoder y crear variables dummy
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [1])],   
    remainder='passthrough'                        
)

X = np.array(ct.fit_transform(X), dtype=np.float)

# Evitar la trampa de las variables dummy, hay que eliminar una columna
X = X[:, 1:]

# Dividir el dataset en conjunto de entrenamiento y conjunto de testing
# 80% conjunto entrenamiento y 20% conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Escalado de variables
# Para que no haya tanta diferencia en los rangos de
# valores en los datos de las distintas columnas
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# Parte 2 - Construir la RNA

# Importar Keras y librerías adicionales
import keras
from keras.models import Sequential
from keras.layers import Dense

# Inicializar la RNA
classifier = Sequential()

# Añadir las capas de entrada y primera oculta
# Para elegir los nodos de la capa oculta se suele coger la media de los nodos de la capa
# de entrada y de la capa de salida, en este caso hay 11 nodos/caracteristicas de entrada,
# y un nodo en la capa de salida, por tanto elegimos 6 nodos en la capa oculta
classifier.add(Dense(units = 6, kernel_initializer = 'uniform',
                     activation = 'relu', input_dim = 11))

# A partir de aqui no hace falta indicar el input_dim porque los siguientes nodos ya
# se conectaran automaticamente con la capa anterior

# Añadir la segunda capa oculta
classifier.add(Dense(units = 6, kernel_initializer = 'uniform',
                     activation = 'relu'))

# Añadir la capa de salida
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compilar la RNA
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Ajustar la RNA al Conjunto de Entrenamiento
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)


# Parte 3 - Evaluar el modelo y calcular predicciones finales

# Predicción de los resultados con el Conjunto de Testing
y_pred  = classifier.predict(X_test)
y_pred = y_pred > 0.5

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
