
# Regresion Polinomica

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


# Ajustar Modelo de Regresion Lineal con el Conjunto de Datos
lin_reg = lm(formula = Salary ~ .,
             data = dataset)

# Ajustar Modelo de Regresion Polinomica con el Conjunto de Datos
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
poly_reg = lm(formula = Salary ~ .,
              data = dataset)

# Visualizacion del Modelo Lineal
library(ggplot2) # Instalar con install.packages('ggplot2')
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             color = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)),
            color = 'blue') +
  ggtitle('Prediccion lineal del sueldo en funcion del nivel del empleado') +
  xlab('Nivel del empleado') +
  ylab('Sueldo (en $)')

# Visualizacion del Modelo Polinomico
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             color = 'red') +
  geom_line(aes(x = x_grid, y = predict(poly_reg, newdata = data.frame(Level = x_grid,
                                                                       Level2 = x_grid^2,
                                                                       Level3 = x_grid^3,
                                                                       Level4 = x_grid^4))),
            color = 'blue') +
  ggtitle('Prediccion polinomica del sueldo en funcion del nivel del empleado') +
  xlab('Nivel del empleado') +
  ylab('Sueldo (en $)')

# Subiendo el grado del polinomio la curva de prediccion se adapta mejor a los puntos

# Prediccion de nuevos resultados con Regresion Lineal
y_pred = predict(lin_reg, newdata = data.frame(Level = 6.5))

# Prediccion de nuevos resultados con Regresion Polinomica
y_pred_poly = predict(poly_reg, newdata = data.frame(Level = 6.5,
                                                     Level2 = 6.5^2,
                                                     Level3 = 6.5^3,
                                                     Level4 = 6.5^4))
