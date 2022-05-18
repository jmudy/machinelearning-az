
# Apriori

setwd("~/repos/machinelearning-az/datasets/Part 5 - Association Rule Learning")

# Importar los datos
dataset = read.csv('data/Market_Basket_Optimisation.csv', header = FALSE)

library(arules) # Instalar con install.packages('arules')
