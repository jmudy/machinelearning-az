
# Natural Language Processing

setwd("~/repos/machinelearning-az/datasets/Part 7 - Natural Language Processing")

# Importar el dataset
dataset_original = read.delim('data/Restaurant_Reviews.tsv', quote = '',
                     stringsAsFactors = FALSE)

# Limpieza de textos
library(tm) # Instalar con install.packages('tm')
library(SnowballC) # Instalar con install.packages('SnowballC')
corpus = VCorpus(VectorSource(dataset_original$Review))
corpus = tm_map(corpus, content_transformer(tolower)) # Convertir a minuscula
# Consultar el primer elemento del corpus
#as.character(corpus[[1]])
corpus = tm_map(corpus, removeNumbers) # Eliminar numeros de las reviews del corpus
corpus = tm_map(corpus, removePunctuation) # Eliminar signos de puntuacion
corpus = tm_map(corpus, removeWords, stopwords(kind = 'en')) # Eliminar palabras irrelevantes
corpus = tm_map(corpus, stemDocument) # Eliminar las "variantes" de palabras por su "raiz" ej: loved -> love
corpus = tm_map(corpus, stripWhitespace) # Eliminar espacios en blanco

# Crear el modelo Bag of Words
dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm, 0.999)

dataset = as.data.frame(as.matrix(dtm))
dataset$Liked = dataset_original$Liked

# Codificar la variable de clasificacion como factor
dataset$Liked = factor(dataset$Liked, levels = c(0, 1))

# Dividir los datos en conjunto de entrenamiento y conjunto de test
library(caTools) # Instalar con install.packages('caTools')
set.seed(123) # Para que salgan los mismos resultados que en el curso
split = sample.split(dataset$Liked, SplitRatio = 0.80)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)

# Ajustar el clasificador de Random Forest con el conjunto de entrenamiento
library(randomForest) # Instalar con install.packages('randomForest')
classifier = randomForest(x = training_set[, -692],
                          y = training_set$Liked,
                          ntree = 10)

# Prediccion de los resultados con el conjunto de Testing
y_pred = predict(classifier, newdata = testing_set[, -692]) # Todo el dataset excluyendo la columna 3

# Crear la matriz de confusion
cm = table(testing_set[, 692], y_pred)
