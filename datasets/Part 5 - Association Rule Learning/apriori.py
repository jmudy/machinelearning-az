# -*- coding: utf-8 -*-
"""
Created on Wed May 18 11:35:45 2022

@author: mudar
"""

# Apriori

# Como importar las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
dataset = pd.read_csv('data/Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])
    
# Entrenar el algoritmo de Apriori
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2,
                min_lift = 3, min_length = 2)

# Visualizacion de los resultados
results = list(rules)
results[4]
