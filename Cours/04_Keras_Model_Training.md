# IV. L'entraînement d'un modèle avec Keras

### 1. La fonction de perte (Loss Function)

La fonction de perte mesure la différence entre la sortie du modèle et la sortie attendue. Elle est utilisée pour 
entraîner le modèle à ajuster les poids des neurones afin de minimiser cette différence.

````jupyterpython
from keras.losses import categorical_crossentropy

model.compile(loss=categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
````

### 2. L'optimiseur (Optimizer)  

L'optimiseur est responsable de la mise à jour des poids des neurones en fonction de la fonction de perte.

````jupyterpython
from keras.optimizers import Adam

model.compile(loss=categorical_crossentropy, optimizer=Adam(lr=0.001), metrics=['accuracy'])
````

#### a. Le processus d'entraînement (Training Process)  

Le processus d'entraînement consiste à fournir les données d'entraînement au modèle et à ajuster les poids des neurones 
en fonction de la fonction de perte.

````jupyterpython
history = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_val, y_val))
````

#### b. Les callbacks avec Keras

Les callbacks sont des fonctions appelées pendant l'entraînement du modèle à des moments précis, tels que la fin d'une 
époque ou la fin d'un batch.  
Les callbacks peuvent être utilisés pour effectuer des actions spécifiques, telles que l'enregistrement des poids du 
modèle ou la réduction du taux d'apprentissage en cas de stagnation de l'entraînement.

````jupyterpython
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping

callbacks = [ModelCheckpoint('best_model.h5', save_best_only=True), EarlyStopping(patience=3)]

history = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_val, y_val), callbacks=callbacks)
````

Dans cet exemple, le callback ModelCheckpoint permet de sauvegarder uniquement le meilleur modèle selon la métrique de 
validation.  
Le callback EarlyStopping permet d'arrêter l'entraînement si la métrique de validation ne s'améliore plus après un 
certain nombre d'époques spécifié par la patience.

### 3. Les métriques d'évaluation (Evaluation Metrics)

Après l'entraînement d'un modèle, il est important d'évaluer ses performances. Keras fournit plusieurs métriques 
d'évaluation pour mesurer la qualité des prédictions de votre modèle.  
Les métriques d'évaluation les plus courantes incluent l'accuracy, la précision, le rappel et le F1-score.

Dans l'exemple ci-dessous, nous allons utiliser le dataset MNIST pour entraîner un modèle de classification d'images et 
utiliser l'accuracy comme métrique d'évaluation.

````jupyterpython
import tensorflow as tf
from tensorflow import keras

# Charger les données MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalisation des données
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Création du modèle
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compilation du modèle
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Entraînement du modèle
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Evaluation du modèle
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Loss :', test_loss)
print('Accuracy :', test_acc)
````

Ici, nous avons utilisé l'accuracy comme métrique d'évaluation pour évaluer la performance de notre modèle.  
L'accuracy est le nombre de prédictions correctes divisé par le nombre total de prédictions. 


### 4. La sauvegarde et le chargement d'un modèle avec Keras

#### a. La sauvegarde d'un modèle (Saving a Model)  

La sauvegarde d'un modèle entraîné permet de le réutiliser ultérieurement pour effectuer des prédictions sur de 
nouvelles données.

````jupyterpython
model.save('my_model.h5')
````

#### b. Le chargement d'un modèle (Loading a Model)  

Le chargement d'un modèle sauvegardé permet de réutiliser le modèle pour effectuer des prédictions sur de nouvelles 
données.

````jupyterpython
from keras.models import load_model

loaded_model = load_model('my_model.h5')
````

En résumé, Keras est une bibliothèque Python simple et puissante pour la création de réseaux de neurones.  
Elle offre une interface facile à utiliser pour créer des modèles de réseaux de neurones, entraîner ces modèles, 
évaluer leurs performances et sauvegarder et charger des modèles entraînés.  
Keras est utilisée dans de nombreuses applications, telles que la vision par ordinateur, le traitement du langage 
naturel, l'apprentissage par renforcement, etc.

En plus des fonctionnalités de base présentées précédemment, Keras offre également de nombreuses autres fonctionnalités 
pour personnaliser et optimiser les modèles de réseaux de neurones.
