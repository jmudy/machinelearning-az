
# Clustering con K-means

setwd("~/repos/machinelearning-az/datasets/Part 4 - Clustering")

# Importar los datos
dataset = read.csv('data/Mall_Customers.csv')
X = dataset[, 4:5]

# Metodo del codo
set.seed(6)
wcss = vector()
for (i in 1:10) {
  wcss[i] <- sum(kmeans(X, i)$withinss)
}
plot(1:10, wcss, type = 'b', main = 'Metodo del codo',
     xlab = 'Numero de clusters (k)', ylab = 'WCSS(k)')

# Aplicar el algoritmo de K-means con k optimo
set.seed(29)
kmeans <- kmeans(X, 5, iter.max = 300, nstart = 10)

# Visualizacion de los clusters
library(cluster) # Instalar con install.packages('cluster')
clusplot(X,
         kmeans$cluster,
         lines = 0, # Para que no dibuje lineas entre los puntos
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE, # Para que todos los simbolos sean iguales
         span = TRUE,
         main = 'Clustering de clientes',
         xlab = 'Ingresos anuales',
         ylab = 'Puntuacion (1- 100)')
