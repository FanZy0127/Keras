import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Flatten
from keras.utils import pad_sequences


# Chargement des données
# max_features = 20000
max_features = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Mise à l'échelle des données pour qu'elles aient la même longueur
# max_review_length = 10000
max_review_length = 500
x_train = pad_sequences(x_train, maxlen=max_review_length)
x_test = pad_sequences(x_test, maxlen=max_review_length)

# Création du modèle
model = Sequential()
model.add(Embedding(max_features, 128, input_length=max_review_length))
model.add(Flatten())
model.add(Dense(256, input_shape=(max_review_length,), activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.summary()

# Compilation du modèle
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entraînement du modèle
history = model.fit(x_train, y_train, epochs=4, batch_size=128, validation_data=(x_test, y_test))

# Enregistrement du modèle
model.save('imdb_model.h5')

# Plot des valeurs d'entraînement et de validation de l'accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot des valeurs d'entraînement et de validation de la loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Évaluation du modèle sur les données de test
score, acc = model.evaluate(x_test, y_test, verbose=0)
print('Test score:', score)
print('Test accuracy:', acc)

# Prédiction pour une nouvelle critique
word_to_id = imdb.get_word_index()
word_to_id = {k: (v + 3) for k, v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2
id_to_word = {value: key for key, value in word_to_id.items()}


def predict_sentiment(review: str):
    words = review.lower().split()
    review = [word_to_id[word] if word in word_to_id else 2 for word in words]
    review = pad_sequences([review], maxlen=max_review_length)
    prediction = model.predict(review)[0][0]
    return prediction


# Test de la prédiction pour une nouvelle critique positive
review = "This movie was really great, I highly recommend it!"
prediction = predict_sentiment(review)
print("Review: ", review)
print("Sentiment prediction (0 = negative, 1 = positive): ", prediction)
