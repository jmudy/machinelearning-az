
# Regresion Lineal Multiple

setwd("~/repos/machinelearning-az/datasets/Part 2 - Regression")

# Importar el dataset
dataset = read.csv('data/50_Startups.csv')
#dataset = dataset[, 2:3]

# Codificar las variables categoricas
dataset$State = factor(dataset$State,
                       levels = c('New York', 'California', 'Florida'),
                       labels = c(1, 2, 3))

# Dividir los datos en conjunto de entrenamiento y conjunto de test
library(caTools) # Instalar con install.packages('caTools')
set.seed(123) # Para que salgan los mismos resultados que en el curso
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)

# Escalado de valores
#training_set[, 2:3] = scale(training_set[, 2:3])
#testing_set[, 2:3] = scale(testing_set[, 2:3])

# Ajustar el modelo de Regresion Lineal Multiple con el Conjunto de Entrenamiento
regression = lm(formula = Profit ~ ., # El punto indica que el resto de variables son las independientes
                data = training_set)

# Al hacer summary(regression) se puede ver que la funcion lm() ha creado eliminado del
# calculo de coeficientes una variable dummy para evitar problemas de colinealidad, cosa
# que en Python tendremos que hacer a mano

# Predecir los resultados con el conjunto de testing
y_pred = predict(regression, newdata = testing_set)

# Construir un modelo optimo con la Eliminacion hacia atras
SL = 0.05
regression = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend, data = dataset)
summary(regression)

# Ejercicio eliminar variables que presenten un p-valor > SL

regression = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend, data = dataset)
summary(regression)

regression = lm(formula = Profit ~ R.D.Spend + Marketing.Spend, data = dataset)
summary(regression)

regression = lm(formula = Profit ~ R.D.Spend, data = dataset)
summary(regression)


#============================================================================
# Automatizacion eliminacion hacia atras utilizando p-valores
backwardElimination <- function(x, sl) {
  numVars = length(x)
  for (i in c(1:numVars)){
    regressor = lm(formula = Profit ~ ., data = x)
    maxVar = max(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"])
    if (maxVar > sl){
      j = which(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"] == maxVar)
      x = x[, -j]
    }
    numVars = numVars - 1
  }
  return(summary(regressor))
}

SL = 0.05
dataset = dataset[, c(1,2,3,4,5)]
backwardElimination(training_set, SL)
