# Keras 
    
## Plan détaillé du cours

1. Introduction à Keras
    - Qu'est-ce que Keras ?
    - Pourquoi utiliser Keras ?
    - Installation de Keras
2. Les différentes couches de neurones
    - Les couches d'entrée (Input Layers)
    - Les couches de convolution (Convolution Layers)
    - Les couches de mise en commun (Pooling Layers)
    - Les couches entièrement connectées (Fully Connected Layers)
3. La création d'un modèle avec Keras
    - Les modèles séquentiels (Sequential Models)
    - Les modèles fonctionnels (Functional Models)
4. L'entraînement d'un modèle avec Keras
    - La fonction de perte (Loss Function)
    - L'optimiseur (Optimizer)
      - Le processus d'entraînement (Training Process)
      - Les callbacks avec Keras
    - Les métriques d'évaluation (Evaluation Metrics)
    - La sauvegarde et le chargement d'un modèle avec Keras
      - La sauvegarde d'un modèle (Saving a Model)
      - Le chargement d'un modèle (Loading a Model)
5. La régularisation dans Keras
    - La régularisation L1 et L2
    - La régularisation par abandon (Dropout)
6. La visualisation des résultats
    - La visualisation des courbes d'apprentissage
    - La visualisation des sorties intermédiaires (Feature Maps)
7. L'application de Keras à des exemples concrets
    - La classification d'images (MNIST)
    - La classification de texte (IMDB)
    - La génération de texte (Shakespeare)
    - La détection d'objets (YOLO)


# I. Introduction à Keras

### 1. Qu'est-ce que Keras ?

Keras est une bibliothèque open source de Deep Learning qui permet de créer des modèles de réseaux de neurones de 
manière rapide et facile. Elle est écrite en Python et s'appuie sur des bibliothèques de calcul numérique telles que 
TensorFlow ou Theano pour accélérer les calculs sur les GPUs.

### 2. Pourquoi utiliser Keras ?

Keras est une bibliothèque très populaire dans le domaine du Deep Learning car elle permet de créer des modèles de 
manière intuitive et rapide. Elle est également très flexible et permet de construire une grande variété de modèles de 
réseaux de neurones, de la classification d'images à la génération de texte en passant par la détection d'objets.

### 3. Installation de Keras

Keras peut être installé via pip en exécutant les commandes suivantes dans un terminal :

````bash
pip install tensorflow
pip install keras
````

L'installation de **Tensorflow** est nécessaire car Keras dépend de cette dernière bibliothèque.
