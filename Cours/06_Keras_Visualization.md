# VI. La visualisation des résultats

Une fois que vous avez entraîné votre modèle, il est important d'analyser les résultats pour comprendre comment il se 
comporte et comment l'améliorer.  
Keras propose plusieurs outils de visualisation pour vous aider dans cette tâche.


### a. La visualisation des courbes d'apprentissage

Les courbes d'apprentissage représentent l'évolution de la fonction de perte et de l'exactitude (accuracy) sur les 
ensembles d'entraînement et de validation au cours de l'apprentissage.  
Ces courbes permettent de visualiser la qualité de l'apprentissage, et notamment de détecter un surapprentissage 
(overfitting) ou un sous-apprentissage (underfitting).

Voici un exemple de code pour afficher les courbes d'apprentissage d'un modèle :

````jupyterpython
import matplotlib.pyplot as plt

# Entraînement du modèle
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Affichage de la courbe de perte
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Courbes d\'apprentissage')
plt.ylabel('Fonction de perte')
plt.xlabel('Époque')
plt.legend(['Entraînement', 'Validation'], loc='upper right')
plt.show()

# Affichage de la courbe d'exactitude
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Courbes d\'apprentissage')
plt.ylabel('Exactitude')
plt.xlabel('Époque')
plt.legend(['Entraînement', 'Validation'], loc='lower right')
plt.show()
````

Le premier graphique affiche l'évolution de la fonction de perte sur les ensembles d'entraînement et de validation, 
tandis que le deuxième graphique affiche l'évolution de l'exactitude sur ces mêmes ensembles.


### b. La visualisation des sorties intermédiaires (Feature Maps)

Les sorties intermédiaires des couches de convolution peuvent être visualisées sous forme de cartes de caractéristiques 
(feature maps).  
Ces cartes permettent de visualiser les caractéristiques détectées par chaque neurone de la couche de convolution, 
et ainsi de mieux comprendre comment fonctionne la couche.

Voici un exemple de code pour afficher les cartes de caractéristiques d'une couche de convolution :

````jupyterpython
from keras import model
from keras.models import Model
import matplotlib.pyplot as plt

# Liste des sorties de toutes les couches avec 'conv' dans leur nom
layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name]

# Création du modèle qui prend les mêmes entrées que `model` et en sortie les activations des couches convolutives
activation_model = Model(inputs=model.input, outputs=layer_outputs)

# Calcul des activations pour une donnée `X`
activations = activation_model.predict(X)

# Affichage de la première carte de caractéristiques de la première couche convolutive
plt.imshow(activations[0][0, :, :, 0], cmap='viridis')
plt.show()
````
