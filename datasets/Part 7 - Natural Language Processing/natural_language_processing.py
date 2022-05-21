# -*- coding: utf-8 -*-
"""
Created on Sat May 21 18:50:24 2022

@author: mudar
"""

# Natural Language Processing

# Importar las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
dataset = pd.read_csv('data/Restaurant_Reviews.tsv',
                      delimiter = '\t',
                      quoting = 3) # Ignora las comillas dobles

# Limpieza de texto
import re
import nltk
nltk.download('stopwords') # Eliminar palabras irrelevantes
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []

for i in range(0, 1000):
    # Sustituir por un espacio todos los caracteres excepto
    # con las letras de a la a a la z minusculas y mayusculas
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower() # Pasar a minusculas
    review = review.split() # Pasar a lista la review
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Crear el Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Dividir el dataset en conjunto de entrenamiento y conjunto de testing
# 80% conjunto entrenamiento y 20% conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Ajustar el clasificador en el Conjunto de Entrenamiento
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Prediccion de los resultados con el Conjunto de Testing
y_pred = classifier.predict(X_test)

# Elaborar una matriz de confusion
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)
(55+91)/200
