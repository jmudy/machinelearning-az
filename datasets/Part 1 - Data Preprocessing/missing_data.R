
# Plantilla para el Pre Procesado de Datos - Datos faltantes

setwd("~/repos/machinelearning-az/datasets/Part 1 - Data Preprocessing")

# Importar el dataset
dataset = read.csv('data/Data.csv')


# Tratamiento de los valores NA
dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age)

dataset$Salary = ifelse(is.na(dataset$Salary),
                        ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                        dataset$Salary)
                        