
# Redes Neuronales Artificiales

setwd("~/repos/machinelearning-az/datasets/Part 8 - Deep Learning")

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

# Escalado de valores
# Hay que indicar las columnas donde hacer el escalado
training_set[, -11] = scale(training_set[, -11])
testing_set[, -11] = scale(testing_set[, -11])

# Crear la RNA
library(h2o) # Instalar con install.packages('h2o')
h2o.init(nthreads = -1)
classifier = h2o.deeplearning(y = 'Exited',
                              training_frame = as.h2o(training_set),
                              activation = 'Rectifier',
                              hidden = c(6, 6),
                              epochs = 100,
                              train_samples_per_iteration = -2)

# Prediccion de los resultados con el conjunto de Testing
prob_pred = h2o.predict(classifier,
                        newdata = as.h2o(testing_set[, -11])) # Todo el dataset excluyendo la columna 11
y_pred = prob_pred > 0.5
y_pred = as.vector(y_pred)

# Crear la matriz de confusion
cm = table(testing_set[, 11], y_pred)

# Cerrar la sesion de H2O
h2o.shutdown()
