
# ACP

setwd("~/repos/machinelearning-az/datasets/Part 9 - Dimensionality Reduction")

# Importar el dataset
dataset = read.csv('data/Wine.csv')

# Dividir los datos en conjunto de entrenamiento y conjunto de test
library(caTools) # Instalar con install.packages('caTools')
set.seed(123) # Para que salgan los mismos resultados que en el curso
split = sample.split(dataset$Customer_Segment, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)

# Escalado de valores
# Hay que indicar las columnas donde hacer el escalado
training_set[, -14] = scale(training_set[, -14])
testing_set[, -14] = scale(testing_set[, -14])

# Proyeccion de las componentes principales
library(caret) # Instalar con install.packages('caret')
library(e1071) # Instalar con install.packages('e1071')
pca = preProcess(x = training_set[, -14], method = 'pca', pcaComp = 2)
training_set = predict(pca, training_set)
training_set = training_set[, c(2, 3, 1)] # Cambiar el orden de las columnas
testing_set = predict(pca, testing_set)
testing_set = testing_set[, c(2, 3, 1)] # Cambiar el orden de las columnas

# Ajustar el modelo de Regresion Logistica con el conjunto de entrenamiento
classifier = svm(formula = Customer_Segment ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'linear')

# Prediccion de los resultados con el conjunto de Testing
y_pred = predict(classifier, newdata = testing_set[, -3]) # Todo el dataset excluyendo la columna 3

# Crear la matriz de confusion
cm = table(testing_set[, 3], y_pred)

# Visualizacion del conjunto de Entrenamiento
# Con install.packages('ElemStatLearn') salta error
# Entrar en https://cran.r-project.org/src/contrib/Archive/ElemStatLearn/
# Descargar e instalar version ElemStatLearn_2015.6.26.2.tar.gz	
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.05)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.05)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('PC1', 'PC2')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'SVM (Conjunto de Entrenamiento)',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 2, 'deepskyblue',
                                         ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set, pch = 21, bg = ifelse(set[, 3] == 2, 'blue3',
                                  ifelse(set[, 3] == 1, 'green4', 'red3')))

# Visualizacion del conjunto de Testing
set = testing_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.05)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.05)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('PC1', 'PC2')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'SVM (Conjunto de Entrenamiento)',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 2, 'deepskyblue',
                                         ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set, pch = 21, bg = ifelse(set[, 3] == 2, 'blue3',
                                  ifelse(set[, 3] == 1, 'green4', 'red3')))
