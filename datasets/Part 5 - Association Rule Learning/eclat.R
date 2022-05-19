
# Eclat

setwd("~/repos/machinelearning-az/datasets/Part 5 - Association Rule Learning")

# Importar los datos
library(arules) # Instalar con install.packages('arules')
dataset = read.csv('data/Market_Basket_Optimisation.csv', header = FALSE)
dataset = read.transactions('data/Market_Basket_Optimisation.csv',
                            sep = ',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 100)

# Entrenar algoritmo Eclat con el dataset
rules = eclat(data = dataset,
              parameter = list(support = 0.004, minlen = 2))

# Visualizacion de los resultados
inspect(sort(rules, by = 'support')[1:10])

# Representacion grafica de las reglas de asociacion
library(arulesViz)
plot(rules, method = "graph", engine = "htmlwidget")
