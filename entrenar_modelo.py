import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

# Leer y preparar texto
with open("letras.txt", "r", encoding="utf-8") as file:
    texto = file.read().lower()

# Tokenizar palabras
tokenizer = Tokenizer()
tokenizer.fit_on_texts([texto])
total_words = len(tokenizer.word_index) + 1

# Guardar tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Crear secuencias de entrenamiento
input_sequences = []
for line in texto.split("\n"):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Rellenar secuencias
max_sequence_len = max(len(x) for x in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

# Dividir en datos X e y
X = input_sequences[:, :-1]
y = input_sequences[:, -1]
y = np.eye(total_words)[y]

# Definir el modelo
model = Sequential()
model.add(Embedding(total_words, 64, input_length=X.shape[1]))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(256))
model.add(Dropout(0.3))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Checkpoint para guardar el mejor modelo
checkpoint = ModelCheckpoint("modelo_lstm.h5", monitor='loss', save_best_only=True)

# Entrenar el modelo
model.fit(X, y, epochs=50, batch_size=128, callbacks=[checkpoint])
