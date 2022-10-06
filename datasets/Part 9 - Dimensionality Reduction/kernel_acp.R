
# Kernel ACP

setwd("~/repos/machinelearning-az/datasets/Part 9 - Dimensionality Reduction")

# Importar el dataset
dataset = read.csv('data/Social_Network_Ads.csv')
dataset = dataset[, 3:5]

# Dividir los datos en conjunto de entrenamiento y conjunto de test
library(caTools) # Instalar con install.packages('caTools')
set.seed(123) # Para que salgan los mismos resultados que en el curso
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)

# Escalado de valores
# Hay que indicar las columnas donde hacer el escalado
training_set[, 1:2] = scale(training_set[, 1:2])
testing_set[, 1:2] = scale(testing_set[, 1:2])

# Aplicar Kernel ACP
library(kernlab) # Instalar con install.packages('kernlab')
kpca = kpca(~ ., data = training_set[, -3], kernel = 'rbfdot', features = 2)
training_set_kpca = as.data.frame(predict(kpca, training_set))
training_set_kpca$Purchased = training_set$Purchased

testing_set_kpca = as.data.frame(predict(kpca, testing_set))
testing_set_kpca$Purchased = testing_set$Purchased

# Ajustar el modelo de Regresion Logistica con el conjunto de entrenamiento
classifier = glm(formula = Purchased ~ .,
                 data = training_set_kpca,
                 family = binomial)

# Prediccion de los resultados con el conjunto de Testing
prob_pred = predict(classifier, type = 'response',
                    newdata = testing_set_kpca[, -3]) # Todo el dataset excluyendo la columna 3

y_pred = ifelse(prob_pred > 0.5, 1, 0)

# Crear la matriz de confusion
cm = table(testing_set_kpca[, 3], y_pred)

# Visualizacion del conjunto de Entrenamiento
# Con install.packages('ElemStatLearn') salta error
# Entrar en https://cran.r-project.org/src/contrib/Archive/ElemStatLearn/
# Descargar e instalar version ElemStatLearn_2015.6.26.2.tar.gz	
library(ElemStatLearn)
set = training_set_kpca
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.05)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.05)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('V1', 'V2')
prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)
plot(set[, -3],
     main = 'Clasificación (Conjunto de Entrenamiento)',
     xlab = 'CP1', ylab = 'CP2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

# Visualizacion del conjunto de Testing
set = testing_set_kpca
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.05)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.05)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('V1', 'V2')
prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)
plot(set[, -3],
     main = 'Clasificación (Conjunto de Testing)',
     xlab = 'CP1', ylab = 'CP2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))
