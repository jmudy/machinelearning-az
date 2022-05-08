
# Arbol de Decision para Regresion

setwd("~/repos/machinelearning-az/datasets/Part 2 - Regression")

# Importar el dataset
dataset = read.csv('data/Position_Salaries.csv')
dataset = dataset[, 2:3]

# En este caso no se va a dividir en conjunto de entrenamiento y test
# debido a la escasa cantidad de datos

# Dividir los datos en conjunto de entrenamiento y conjunto de test
# library(caTools) # Instalar con install.packages('caTools')
# set.seed(123) # Para que salgan los mismos resultados que en el curso
# split = sample.split(dataset$Purchased, SplitRatio = 0.8)
# training_set = subset(dataset, split == TRUE)
# testing_set = subset(dataset, split == FALSE)


# Escalado de valores
#training_set[, 2:3] = scale(training_set[, 2:3])
#testing_set[, 2:3] = scale(testing_set[, 2:3])


# Ajustar Modelo de Regresion con el Conjunto de Datos
library(rpart) # Instalar con install.packages('rpart')
regression = rpart(formula = Salary ~ .,
                   data = dataset,
                   control = rpart.control(minsplit = 1))

# Prediccion de nuevos resultados con Arbol de Regresion
y_pred = predict(regression, newdata = data.frame(Level = 6.5))

# Visualizacion del Modelo de Arbol de Regresion
library(ggplot2) # Instalar con install.packages('ggplot2')
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             color = 'red') +
  geom_line(aes(x = x_grid, y = predict(regression,
                                        newdata = data.frame(Level = x_grid))),
            color = 'blue') +
  ggtitle('Prediccion con Arbol de Decision (Modelo de Regresion)') +
  xlab('Nivel del empleado') +
  ylab('Sueldo (en $)')
