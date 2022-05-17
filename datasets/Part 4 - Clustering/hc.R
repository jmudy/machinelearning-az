
# Clustering Jerarquico

setwd("~/repos/machinelearning-az/datasets/Part 4 - Clustering")

# Importar los datos
dataset = read.csv('data/Mall_Customers.csv')
X = dataset[, 4:5]

# Utilizar el dendrograma para encontrar el numero optimo de clusters
dendrogram = hclust(dist(X, method = 'euclidean'),
                    method = 'ward.D')

plot(dendrogram,
     main = 'Dendrograma',
     xlab = 'Clientes del centro comercial',
     ylab = 'Distancia Euclidea')

# Ajustar el Clustering Jerarquico a nuestro dataset
hc = hclust(dist(X, method = 'euclidean'),
            method = 'ward.D')

y_hc = cutree(hc, k = 5)

# Visualizar los clusters
library(cluster) # Instalar con install.packages('cluster')
clusplot(X,
         y_hc,
         lines = 0, # Para que no dibuje lineas entre los puntos
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE, # Para que todos los simbolos sean iguales
         span = TRUE,
         main = 'Clustering de clientes',
         xlab = 'Ingresos anuales',
         ylab = 'Puntuacion (1- 100)')
