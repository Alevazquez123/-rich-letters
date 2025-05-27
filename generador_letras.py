from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Cargar modelo y tokenizer
model = load_model("modelo_lstm.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_sequence_len = model.input_shape[1]

def sample_with_temperature(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-9) / temperature  # +1e-9 para evitar log(0)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generar_letra(seed_text, next_words=50, temperature=1.0):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')
        preds = model.predict(token_list, verbose=0)[0]
        predicted = sample_with_temperature(preds, temperature)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        if output_word == "":
            break
        seed_text += " " + output_word
    return seed_text

@app.route("/", methods=["GET"])
def index():
    return render_template("letras.html")

@app.route("/generar", methods=["POST"])
def generar():
    texto = request.form.get("texto", "")
    genero = request.form.get("genero", "")

    if not texto:
        return jsonify({"resultado": "", "recomendacion": "No se proporcionó texto inicial."})

    letra_generada = generar_letra(texto, next_words=50)

    recomendacion = f"Género seleccionado: {genero}. Próximamente recomendaciones."

    return jsonify({"resultado": letra_generada, "recomendacion": recomendacion})

if __name__ == "__main__":
    app.run(port=3000, debug=True)
