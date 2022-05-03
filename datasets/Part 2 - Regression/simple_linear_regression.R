
# Regresion Lineal Simple

setwd("~/repos/machinelearning-az/datasets/Part 2 - Regression")

# Importar el dataset
dataset = read.csv('data/Salary_Data.csv')
#dataset = dataset[, 2:3]

# Dividir los datos en conjunto de entrenamiento y conjunto de test
library(caTools) # Instalar con install.packages('caTools')
set.seed(123) # Para que salgan los mismos resultados que en el curso
# 20 individuos para entrenar y 10 individuos para validar
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)


# Escalado de valores
# Hay que indicar las columnas donde hacer el escalado
# No se hacen sobre los valores dummy de las columnas Country y Purchased porque
# no puede operar con factores que convirtio previamente a tipo string
#training_set[, 2:3] = scale(training_set[, 2:3])
#testing_set[, 2:3] = scale(testing_set[, 2:3])

# Ajustar el modelo de regresion lineal simple con el conjunto de entrenamiento
regressor = lm(formula = Salary ~ YearsExperience,
               data = training_set)

# Predecir resultados con el conjunto de test
y_pred = predict(regressor, newdata = testing_set)

# Visualizacion de los resultados en el conjunto de entrenamiento
library(ggplot2) # Instalar con install.packages('ggplot2')
ggplot() +
  geom_point(aes(x = training_set$YearsExperience,
                 y = training_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience,
                y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Sueldo vs Años de Experiencia (Conjunto de Entrenamiento)') +
  xlab('Años de Experiencia') +
  ylab('Suelo (en $)')

# Visualizacion de los resultados en el conjunto de testing
ggplot() +
  geom_point(aes(x = testing_set$YearsExperience,
                 y = testing_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience,
                y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Sueldo vs Años de Experiencia (Conjunto de Testing)') +
  xlab('Años de Experiencia') +
  ylab('Suelo (en $)')
