# -*- coding: utf-8 -*-
"""
Created on Fri May 27 00:01:38 2022

@author: mudar
"""

# Redes Neuronales Convolucionales

# Instalar Theano, Keras y Tensorflow
#pip install Theano
#conda install -c conda-forge keras
#pip install tensorflow


# Parte 1 - Construir el modelo de CNN

# Importar las librerias y paquetes
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Inicializar la CNN
classifier = Sequential()

# Paso 1 - Primera capa de convolucion y Max Pooling
classifier.add(Conv2D(filters = 32,
                      kernel_size = (3, 3),
                      input_shape = (64, 64, 3),
                      activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Paso 2 - Segunda capa de convolucion y Max Pooling
classifier.add(Conv2D(filters = 32,
                      kernel_size = (3, 3),
                      activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Paso 3 - Flattening
classifier.add(Flatten())

# Paso 4 - Fully-Connected
# Primera capa oculta
classifier.add(Dense(units = 128,
                     activation = 'relu'))
# Añadir la capa de salida
classifier.add(Dense(units = 1,
                     activation = 'sigmoid'))

# Compilar la CNN
classifier.compile(optimizer = 'adam',
                   loss = 'binary_crossentropy',
                   metrics = ['accuracy'])


# Parte 2 - Ajustar la CNN a las imagenes para entrar
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_dataset = train_datagen.flow_from_directory('data/training_set',
                                                     target_size = (64, 64),
                                                     batch_size = 32,
                                                     class_mode = 'binary')

testing_dataset = test_datagen.flow_from_directory('data/test_set',
                                                    target_size = (64, 64),
                                                    batch_size = 32,
                                                    class_mode = 'binary')

"""
classifier.fit_generator(training_dataset,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = testing_dataset,
                         validation_steps = 2000)

Salta error en la iteracion 250

Siguiendo las recomendaciones de los usuarios de stackoverflow

steps_per_epoch = len(X_train)//batch_size
validation_steps = len(X_test)//batch_size
"""

classifier.fit(training_dataset,
               steps_per_epoch = int(8000/32),
               epochs = 25,
               validation_data = testing_dataset,
               validation_steps = int(2000/32))
