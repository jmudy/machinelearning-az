
# Plantilla para el Pre Procesado de Datos

setwd("~/repos/machinelearning-az/datasets/Part 1 - Data Preprocessing")

# Importar el dataset
dataset = read.csv('data/Data.csv')
#dataset = dataset[, 2:3]

# Dividir los datos en conjunto de entrenamiento y conjunto de test
library(caTools) # Instalar con install.packages('caTools')
set.seed(123) # Para que salgan los mismos resultados que en el curso
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)

# Escalado de valores
# Hay que indicar las columnas donde hacer el escalado
# No se hacen sobre los valores dummy de las columnas Country y Purchased porque
# no puede operar con factores que convirtio previamente a tipo string
#training_set[, 2:3] = scale(training_set[, 2:3])
#testing_set[, 2:3] = scale(testing_set[, 2:3])
