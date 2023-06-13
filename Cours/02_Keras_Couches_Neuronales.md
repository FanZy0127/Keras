# II. Les différentes couches de neurones

### 1. Les couches d'entrée (Input Layers)  

La couche d'entrée est la première couche du réseau de neurones. Elle sert à spécifier la taille de l'entrée attendue 
par le modèle. Par exemple, si nous travaillons avec des images en noir et blanc de 28x28 pixels, la couche d'entrée 
sera spécifiée comme suit :

````jupyterpython
from keras.layers import Input
from keras.models import Model

input_layer = Input(shape=(28, 28, 1))
````

### 2. Les couches de convolution (Convolution Layers)  

Les couches de convolution sont utilisées pour extraire des caractéristiques d'une image. Elles sont constituées de 
filtres qui balayent l'image et effectuent une opération de convolution pour chaque région. La sortie de cette opération
est une carte de caractéristiques.

Voici un exemple de couche de convolution avec Keras :

````jupyterpython
from keras.layers import Conv2D

conv_layer = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
````

### 3. Les couches de mise en commun (Pooling Layers)  

Les couches de mise en commun sont utilisées pour réduire la taille de la carte de caractéristiques tout en préservant 
les informations les plus importantes. Il existe plusieurs types de couches de mise en commun, telles que la mise en 
commun max ou la mise en commun moyenne.

````jupyterpython
from keras.layers import MaxPooling2D

pooling_layer = MaxPooling2D(pool_size=(2, 2))(conv_layer)
````

### 4. Les couches entièrement connectées (Fully Connected Layers) 

Les couches entièrement connectées sont utilisées pour classifier les données en sortie des couches précédentes. 
Elles sont constituées de neurones qui sont connectés à tous les neurones de la couche précédente.

````jupyterpython
from keras.layers import Flatten, Dense

flatten_layer = Flatten()(pooling_layer)
dense_layer = Dense(units=64, activation='relu')(flatten_layer)
output_layer = Dense(units=10, activation='softmax')(dense_layer)
````
