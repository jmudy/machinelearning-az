
# Clasificacion con Arboles de Decision

setwd("~/repos/machinelearning-az/datasets/Part 3 - Classification")

# Importar el dataset
dataset = read.csv('data/Social_Network_Ads.csv')
dataset = dataset[, 3:5]

# Codificar la variable de clasificacion como factor
dataset$Purchased = factor(dataset$Purchased, levels = c(0, 1))

# Dividir los datos en conjunto de entrenamiento y conjunto de test
library(caTools) # Instalar con install.packages('caTools')
set.seed(123) # Para que salgan los mismos resultados que en el curso
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)


# Escalado de valores
# Hay que indicar las columnas donde hacer el escalado
#training_set[, 1:2] = scale(training_set[, 1:2])
#testing_set[, 1:2] = scale(testing_set[, 1:2])

# Ajustar el clasificador con el conjunto de entrenamiento
library(rpart) # Instalar con install.packages('rpart')
classifier = rpart(formula = Purchased ~.,
                   data = training_set)


# Prediccion de los resultados con el conjunto de Testing
y_pred = predict(classifier, newdata = testing_set[, -3],
                 type = 'class') # Todo el dataset excluyendo la columna 3

# Crear la matriz de confusion
cm = table(testing_set[, 3], y_pred)

# Visualizacion del conjunto de Entrenamiento
# Con install.packages('ElemStatLearn') salta error
# Entrar en https://cran.r-project.org/src/contrib/Archive/ElemStatLearn/
# Descargar e instalar version ElemStatLearn_2015.6.26.2.tar.gz	
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.1)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 50)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set, type = 'class')
plot(set[, -3],
     main = 'Arbol de Decision (Conjunto de Entrenamiento)',
     xlab = 'Edad', ylab = 'Sueldo Estimado',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

# Visualizacion del conjunto de Testing
set = testing_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.1)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 50)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set, type = 'class')
plot(set[, -3],
     main = 'Arbol de Decision (Conjunto de Testing)',
     xlab = 'Edad', ylab = 'Sueldo Estimado',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

# Representacion del arbol de clasificacion
plot(classifier)
text(classifier)
