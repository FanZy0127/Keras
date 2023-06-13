import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten

# Chargement du dataset MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Préparation des données d'entrée
X_train = X_train.reshape((60000, 28, 28, 1))
X_train = X_train.astype('float32') / 255.0
X_test = X_test.reshape((10000, 28, 28, 1))
X_test = X_test.astype('float32') / 255.0

# Préparation des étiquettes de sortie
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Création du modèle
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compilation du modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrainement du modèle
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

# Sauvegarde du modèle
model.save('mnist_model.h5')

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

# Prédiction d'une image de test
img = X_test[15]
prediction = model.predict(img.reshape((1, 28, 28, 1)))

plt.imshow(img.squeeze(), cmap='gray')
plt.title(f"Prédiction du chiffre affiché : {prediction.argmax()}")
plt.show()

print(f"Le chiffre prédit est : {prediction.argmax()}")
