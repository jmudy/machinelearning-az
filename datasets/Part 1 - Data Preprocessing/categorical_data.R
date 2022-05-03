
# Plantilla para el Pre Procesado de Datos - Datos categoricos

setwd("~/repos/machinelearning-az/datasets/Part 1 - Data Preprocessing")

# Importar el dataset
dataset = read.csv('data/Data.csv')


# Codificar las variables categoricas
dataset$Country = factor(dataset$Country,
                         levels = c('France', 'Spain', 'Germany'),
                         labels = c(1, 2, 3))

dataset$Purchased = factor(dataset$Purchased,
                           levels = c('No', 'Yes'),
                           labels = c(0, 1))