# III. La création d'un modèle avec Keras

### 1. Les modèles séquentiels (Sequential Models)  

Les modèles séquentiels sont les modèles les plus simples et les plus couramment utilisés avec Keras. Ils sont 
constitués d'une pile de couches de neurones, où chaque couche est connectée à la couche précédente.

````jupyterpython
from keras.models import Sequential

model = Sequential([
    input_layer,
    conv_layer,
    pooling_layer,
    flatten_layer,
    dense_layer,
    output_layer
])
````

### 2. Les modèles fonctionnels (Functional Models)  

Les modèles fonctionnels sont plus flexibles que les modèles séquentiels, car ils permettent de créer des modèles avec 
des connexions non-linéaires entre les couches.

````jupyterpython
from keras.models import Model
from keras.layers import concatenate

# Définition de la forme de l'entrée 1
input_1 = Input(shape=(28, 28, 1))
# Définition de la forme de l'entrée 2
input_2 = Input(shape=(10,))

# Ajout d'une couche de convolution avec 32 filtres, une taille de noyau de 3x3 et une fonction d'activation ReLU
conv_layer_1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_1)
# Ajout d'une couche de pooling avec une taille de noyau de 2x2
pooling_layer_1 = MaxPooling2D(pool_size=(2, 2))(conv_layer_1)
# Ajout d'une couche d'aplatissement pour convertir le tenseur 3D en un tenseur 1D
flatten_layer_1 = Flatten()(pooling_layer_1)

# Ajout d'une couche dense avec 64 unités et une fonction d'activation ReLU à l'entrée 1
dense_layer_1 = Dense(units=64, activation='relu')(flatten_layer_1)
# Ajout d'une couche dense avec 64 unités et une fonction d'activation ReLU à l'entrée 2
dense_layer_2 = Dense(units=64, activation='relu')(input_2)

# Concaténation des sorties des deux couches denses
concat_layer = concatenate([dense_layer_1, dense_layer_2])

# Ajout d'une couche dense avec 10 unités et une fonction d'activation softmax à la sortie
output_layer = Dense(units=10, activation='softmax')(concat_layer)

# Définition du modèle avec les entrées et les sorties
model = Model(inputs=[input_1, input_2], outputs=output_layer)
````
