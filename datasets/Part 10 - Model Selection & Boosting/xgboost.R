
# XGBoost

setwd("~/repos/machinelearning-az/datasets/Part 10 - Model Selection & Boosting")

# Importar el dataset
dataset = read.csv('data/Churn_Modelling.csv')
dataset = dataset[, 4:14]

# Codificar las variables Geography y Gender como factor
dataset$Geography = as.numeric(factor(dataset$Geography,
                                      levels = c('France', 'Spain', 'Germany'),
                                      labels = c(1, 2, 3)))

dataset$Gender = as.numeric(factor(dataset$Gender,
                                   levels = c('Female', 'Male'),
                                   labels = c(1, 2)))

# Dividir los datos en conjunto de entrenamiento y conjunto de test
library(caTools) # Instalar con install.packages('caTools')
set.seed(123) # Para que salgan los mismos resultados que en el curso
split = sample.split(dataset$Exited, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)

# Ajustar XGBoost al Conjunto de Entrenamiento
library(xgboost) # Instalar con install.packages('xgboost')
classifier = xgboost(data = as.matrix(training_set[, -11]),
                     label = training_set$Exited,
                     nrounds = 10)

# Aplicar algoritmo de k-fold cross validation
library(caret) # Instalar con install.packages('caret')
folds = createFolds(training_set$Exited, k = 10)
cv = lapply(folds, function(x) {
  training_fold = training_set[-x, ]
  test_fold = training_set[x, ]
  classifier = xgboost(data = as.matrix(training_set[, -11]),
                       label = training_set$Exited,
                       nrounds = 10)
  y_pred = predict(classifier, newdata = as.matrix(test_fold[, -11]))
  y_pred = y_pred >= 0.5
  cm = table(test_fold[, 11], y_pred)
  accuracy = (cm[1, 1] + cm[2, 2])/(cm[1, 1] + cm[1, 2] + cm[2, 1] +cm[2, 2])
  return(accuracy)
})

accuracy = mean(as.numeric(cv)) # Media de precisiones
accuracy_sd = sd(as.numeric(cv)) # Varianza de los resultados
