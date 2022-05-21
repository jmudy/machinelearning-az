# -*- coding: utf-8 -*-
"""
Created on Sat May 21 23:39:16 2022

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


# Predicción de los resultados con el Conjunto de Testing con Naive Bayes

# Ajustar el clasificador en el Conjunto de Entrenamiento
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Prediccion de los resultados con el Conjunto de Testing
y_pred = classifier.predict(X_test)

# Elaborar una matriz de confusion
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

TP = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TN = cm[1][1]

accuracy = (TP+TN)/(TP+TN+FP+FN)
precision = TP/(TP+FP)
recall = TP/(TP+FN)

print('Accuracy = {:.2f}'.format(accuracy))
print('Precision = {:.2f}'.format(precision))
print('Recall = {:.2f}'.format(recall))
print('F1 Score = {:.2f}'.format(2*precision*recall/(precision+recall)))

# Accuracy = 0.73
# Precision = 0.57
# Recall = 0.82
# F1 Score = 0.67


# Predicción de los resultados con el Conjunto de Testing con Logistic Regression

# Ajustar el modelo de Regresion Logistica en el Conjunto de Entrenamiento
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Prediccion de los resultados con el Conjunto de Testing
y_pred = classifier.predict(X_test)

# Elaborar una matriz de confusion
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

TP = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TN = cm[1][1]

accuracy = (TP+TN)/(TP+TN+FP+FN)
precision = TP/(TP+FP)
recall = TP/(TP+FN)

print('Accuracy = {:.2f}'.format(accuracy))
print('Precision = {:.2f}'.format(precision))
print('Recall = {:.2f}'.format(recall))
print('F1 Score = {:.2f}'.format(2*precision*recall/(precision+recall)))

# Accuracy = 0.71
# Precision = 0.78
# Recall = 0.67
# F1 Score = 0.72


# Predicción de los resultados con el Conjunto de Testing con K-NN

# Ajustar el K-NN en el Conjunto de Entrenamiento
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Prediccion de los resultados con el Conjunto de Testing
y_pred = classifier.predict(X_test)

# Elaborar una matriz de confusion
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

TP = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TN = cm[1][1]

accuracy = (TP+TN)/(TP+TN+FP+FN)
precision = TP/(TP+FP)
recall = TP/(TP+FN)

print('Accuracy = {:.2f}'.format(accuracy))
print('Precision = {:.2f}'.format(precision))
print('Recall = {:.2f}'.format(recall))
print('F1 Score = {:.2f}'.format(2*precision*recall/(precision+recall)))

# Accuracy = 0.67
# Precision = 0.82
# Recall = 0.62
# F1 Score = 0.70


# Predicción de los resultados con el Conjunto de Testing con SVM

# Ajustar el SVM en el Conjunto de Entrenamiento
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Prediccion de los resultados con el Conjunto de Testing
y_pred = classifier.predict(X_test)

# Elaborar una matriz de confusion
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

TP = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TN = cm[1][1]

accuracy = (TP+TN)/(TP+TN+FP+FN)
precision = TP/(TP+FP)
recall = TP/(TP+FN)

print('Accuracy = {:.2f}'.format(accuracy))
print('Precision = {:.2f}'.format(precision))
print('Recall = {:.2f}'.format(recall))
print('F1 Score = {:.2f}'.format(2*precision*recall/(precision+recall)))

# Accuracy = 0.72
# Precision = 0.76
# Recall = 0.69
# F1 Score = 0.73


# Predicción de los resultados con el Conjunto de Testing con Kernel SVM

# Ajustar el clasificador en el Conjunto de Entrenamiento
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Prediccion de los resultados con el Conjunto de Testing
y_pred = classifier.predict(X_test)

# Elaborar una matriz de confusion
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

TP = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TN = cm[1][1]

accuracy = (TP+TN)/(TP+TN+FP+FN)
precision = TP/(TP+FP)
recall = TP/(TP+FN)

print('Accuracy = {:.2f}'.format(accuracy))
print('Precision = {:.2f}'.format(precision))
print('Recall = {:.2f}'.format(recall))
print('F1 Score = {:.2f}'.format(2*precision*recall/(precision+recall)))

# Accuracy = 0.73
# Precision = 0.93
# Recall = 0.66
# F1 Score = 0.77


# Predicción de los resultados con el Conjunto de Testing con Decission Tree

# Ajustar el clasificador de Arbol de Decision en el Conjunto de Entrenamiento
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Prediccion de los resultados con el Conjunto de Testing
y_pred = classifier.predict(X_test)

# Elaborar una matriz de confusion
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

TP = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TN = cm[1][1]

accuracy = (TP+TN)/(TP+TN+FP+FN)
precision = TP/(TP+FP)
recall = TP/(TP+FN)

print('Accuracy = {:.2f}'.format(accuracy))
print('Precision = {:.2f}'.format(precision))
print('Recall = {:.2f}'.format(recall))
print('F1 Score = {:.2f}'.format(2*precision*recall/(precision+recall)))

# Accuracy = 0.71
# Precision = 0.76
# Recall = 0.68
# F1 Score = 0.72


# Predicción de los resultados con el Conjunto de Testing con Random Forest

# Ajustar el clasificador de Random Forest en el Conjunto de Entrenamiento
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Prediccion de los resultados con el Conjunto de Testing
y_pred = classifier.predict(X_test)

# Elaborar una matriz de confusion
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

TP = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TN = cm[1][1]

accuracy = (TP+TN)/(TP+TN+FP+FN)
precision = TP/(TP+FP)
recall = TP/(TP+FN)

print('Accuracy = {:.2f}'.format(accuracy))
print('Precision = {:.2f}'.format(precision))
print('Recall = {:.2f}'.format(recall))
print('F1 Score = {:.2f}'.format(2*precision*recall/(precision+recall)))

# Accuracy = 0.72
# Precision = 0.90
# Recall = 0.65
# F1 Score = 0.76


# Tabla de resultados

"""
|                     | Accuracy | Precission | Recall | F1 Score |
|---------------------|:--------:|:----------:|:------:|:--------:|
| Naïve Bayes         |   0.73   |    0.57    |  0.82  |   0.67   |
| Logistic Regression |   0.71   |    0.78    |  0.67  |   0.72   |
| K-NN                |   0.67   |    0.82    |  0.62  |   0.70   |
| SVM                 |   0.72   |    0.76    |  0.69  |   0.73   |
| Kernel SVM          |   0.73   |    0.93    |  0.66  |   0.77   |
| Decission Tree      |   0.71   |    0.76    |  0.68  |   0.72   |
| Random Forest       |   0.72   |    0.90    |  0.65  |   0.76   |

Mejor resultado obtenido: SVM con radial basis function kernel
"""
