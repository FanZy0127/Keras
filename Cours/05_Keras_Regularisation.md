# V. La régularisation dans Keras

La régularisation est une technique utilisée pour prévenir le surapprentissage (overfitting) dans les modèles de machine
learning.  
Keras fournit plusieurs techniques de régularisation, y compris la régularisation L1 et L2 et la régularisation par 
abandon (Dropout).

### 1. Régularisation L1 et L2

La régularisation L1 et L2 sont des techniques de régularisation qui ajoutent une pénalité à la fonction de coût pour 
réduire la complexité du modèle.  
L1 ajoute une pénalité égale à la valeur absolue des poids, tandis que L2 ajoute une pénalité égale au carré des poids.

````jupyterpython
# Importation de la bibliothèque Keras et du module de régularisation
from tensorflow import keras
from tensorflow.keras import regularizers

# Création du modèle séquentiel 
model = keras.Sequential([
    
    # Couche d'aplatissement qui transforme les images en tableau 1D
    keras.layers.Flatten(input_shape=(28, 28)),
    
    # Couche dense avec une fonction d'activation "relu" et une régularisation L2
    # La régularisation L2 ajoute une pénalité à la fonction de perte pour les grands poids dans les couches du modèle
    # Ceci permet de prévenir l'overfitting et d'améliorer les performances de généralisation du modèle
    keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    
    # Couche dense de sortie avec une fonction d'activation "softmax"
    # La fonction d'activation "softmax" permet de normaliser les sorties pour représenter une distribution de probabilité
    # qui indique la probabilité que chaque classe soit la classe prédite pour une image donnée
    keras.layers.Dense(10, activation='softmax')
])
````

Ici, nous avons ajouté une régularisation L2 avec un coefficient de pénalité de 0,001 à la couche Dense.

### 2. La régularisation par abandon (Dropout)

La régularisation par abandon (*Dropout*) est une technique de régularisation qui consiste à supprimer aléatoirement 
certains neurones pendant l'entraînement pour réduire la dépendance entre les neurones et ainsi éviter le 
**surapprentissage** (*overfitting*).
La régularisation par abandon est une technique très courante en apprentissage profond et elle peut être utilisée avec 
toutes les couches de neurones de Keras.

Exemple : 

````jupyterpython
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential([
    Dense(64, activation='relu', input_shape=(784,)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
````

Ici, nous créons un modèle avec 3 couches de neurones : 

- une couche d'entrée 
- deux couches cachées 
- une couche de sortie.  

Les deux couches cachées utilisent la régularisation par abandon avec une probabilité de 0.5.

L'utilisation de la régularisation par abandon permet de réduire l'overfitting en empêchant les neurones de se 
spécialiser sur des exemples particuliers du jeu de données.  
À chaque étape d'entraînement, une partie aléatoire des neurones est désactivée, forçant le réseau à apprendre des 
caractéristiques plus robustes et généralisables.

Il est important de noter que la régularisation par abandon ne doit être appliquée qu'aux couches cachées du réseau, 
pas à la couche d'entrée ou à la couche de sortie.  
De plus, la probabilité de désactivation des neurones doit être choisie en fonction du niveau d'overfitting du modèle :  
une probabilité plus élevée réduit davantage l'overfitting, mais peut également réduire la performance du modèle sur 
les données d'entraînement.
