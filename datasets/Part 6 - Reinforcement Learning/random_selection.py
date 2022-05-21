# -*- coding: utf-8 -*-
"""
Created on Fri May 20 12:25:02 2022

@author: mudar
"""

# Seleccion Aleatoria

# Importar las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
dataset = pd.read_csv('data/Ads_CTR_Optimisation.csv')

# Implementrar una Selección Aleatoria
import random
N = 10000
d = 10
ads_selected = []
total_reward = 0
for n in range(0, N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    total_reward = total_reward + reward

# Visualizar los resultados - Histograma
plt.hist(ads_selected)
plt.title('Histograma de selección de anuncios')
plt.xlabel('Anuncio')
plt.ylabel('Numero de veces que ha sido visualizado')
plt.show()
