from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datetime import datetime
import os

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:////tmp/historial.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

class Historial(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    texto_inicial = db.Column(db.Text, nullable=False)
    letra_generada = db.Column(db.Text, nullable=False)
    fecha = db.Column(db.DateTime, default=datetime.utcnow)

model = load_model("modelo_lstm.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_sequence_len = model.input_shape[1]

def sample_with_temperature(preds, temperature=1.0):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-7) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generar_letra(seed_text, next_words=50, temperature=0.8):
    palabra_actual = seed_text.strip()
    resultado = [palabra_actual]
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([palabra_actual])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')
        preds = model.predict(token_list, verbose=0)[0]
        predicted_index = sample_with_temperature(preds, temperature)
        output_word = next((word for word, index in tokenizer.word_index.items() if index == predicted_index), "")
        if not output_word:
            break
        resultado.append(output_word)
        palabra_actual += " " + output_word
    formatted = ""
    for i in range(0, len(resultado), 9):
        formatted += " ".join(resultado[i:i+9]) + "\n"
    return formatted.strip()

@app.route("/", methods=["GET", "POST"])
def index():
    letra_generada = ""
    if request.method == "POST":
        texto_inicial = request.form.get("texto_inicial", "")
        if texto_inicial:
            letra_generada = generar_letra(texto_inicial, next_words=50, temperature=0.8)
            nueva_entrada = Historial(texto_inicial=texto_inicial, letra_generada=letra_generada)
            db.session.add(nueva_entrada)
            db.session.commit()
    return render_template("letras.html", letra_generada=letra_generada)

@app.route("/historial")
def historial():
    entradas = Historial.query.order_by(Historial.fecha.desc()).all()
    return render_template("historial.html", entradas=entradas)

# Crear la base de datos si no existe (cuando Gunicorn inicie)
with app.app_context():
    db.create_all()
    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 4000))
    app.run(host="0.0.0.0", port=port)

