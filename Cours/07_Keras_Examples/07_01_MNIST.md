# VII. 7. L'application de Keras à des exemples concrets

### 1. La classification d'images (MNIST)

Le dataset MNIST est un ensemble de 70 000 images de chiffres écrits à la main (0 à 9), chacune de taille 28x28 pixels. 
L'objectif est de construire un modèle qui est capable de prédire le chiffre présent dans chaque image.

#### a. Préparation des données

Pour commencer, il faut mettre en place les imports, télécharger et charger les données en utilisant les fonctions 
**load_data** de Keras :

````jupyterpython
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten

# Chargement du dataset MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()
````

Les données sont divisées en deux jeux : 

- un jeu d'entraînement (60 000 images) 
- un jeu de test (10 000 images).  

Les images sont représentées en niveaux de gris, avec des valeurs de pixels comprises entre 0 et 255.  
Ainsi, il est recommandé de normaliser les données en divisant chaque valeur de pixel par 255 :

````jupyterpython
# Préparation des données d'entrée
X_train = X_train.reshape((60000, 28, 28, 1))
X_train = X_train.astype('float32') / 255.0
X_test = X_test.reshape((10000, 28, 28, 1))
X_test = X_test.astype('float32') / 255.0
````

Ensuite, les étiquettes (labels) sont des entiers entre 0 et 9, il faut les convertir en vecteurs binaires 
(one-hot encoding) pour entraîner le modèle :

````jupyterpython
# Préparation des étiquettes de sortie
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
````

#### b. Création du modèle

Le modèle pour la classification d'images est généralement construit à partir de couches de convolution 
(convolutional layers) et de couches de mise en commun (pooling layers), suivies de couches entièrement connectées 
(fully connected layers) pour la classification finale.

Nous allons créer un modèle séquentiel (sequential model) en utilisant Keras, qui permet de créer facilement 
des réseaux de neurones séquentiels. Ce modèle aura la structure suivante :

- Une couche de convolution avec 32 filtres de taille 3x3, une fonction d'activation ReLU et une entrée de forme (28, 28, 1)
- Une couche de mise en commun avec une fenêtre de 2x2
- Une deuxième couche de convolution avec 64 filtres de taille 3x3, une fonction d'activation ReLU
- Une deuxième couche de mise en commun avec une fenêtre de 2x2
- Une couche entièrement connectée avec 128 neurones et une fonction d'activation ReLU
- Une couche de sortie avec 10 neurones et une fonction d'activation softmax (pour la classification)

Voici le code pour créer le modèle :

````jupyterpython
# Création du modèle
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
````

#### c. Compilation du modèle

Nous compilons ensuite notre modèle en utilisant l'optimiseur Adam et la fonction de perte **categorical_crossentropy**. 
Nous entraînons ensuite notre modèle pendant 10 époques en utilisant un batch size de 64.  
Nous enregistrons également le modèle en utilisant la fonction **save()** du modèle.

````jupyterpython
# Compilation du modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrainement du modèle
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

# Sauvegarde du modèle
model.save('mnist_model.h5')
````

#### d. Évaluation et visualisation

Enfin, nous évaluons le modèle sur les données de test, visualisons les courbes d'apprentissage et prédisons la sortie 
pour une image de test.

Notez que les graphiques affichés montrent les résultats de l'entraînement de notre modèle. Ils montrent la précision 
et la perte d'entraînement et de validation à chaque époque.  

````jupyterpython
# Evaluation du modèle sur les données de test
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

# Visualisation des courbes d'apprentissage
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(accuracy) + 1)

plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
````

La prédiction pour l'image de test est également affichée sous forme de chiffre pour permettre une comparaison visuelle 
avec le chiffre réel associé à l'image.  
Pour ce faire, on utilise la fonction **argmax()** de NumPy pour obtenir l'indice de la classe prédite avec la 
probabilité la plus élevée dans le vecteur de sortie du modèle. Ensuite, on utilise la fonction **imshow()** de 
matplotlib pour afficher l'image de test et la fonction **title()** pour ajouter le chiffre prédit comme titre.

````jupyterpython
# Prédiction d'une image de test
img = X_test[0]
prediction = model.predict(img.reshape((1, 28, 28, 1)))

plt.imshow(img.squeeze(), cmap='gray')
plt.title(f"Prédiction du chiffre affiché : {prediction.argmax()}")
plt.show()

print(f"Le chiffre prédit est : {prediction.argmax()}")
````
