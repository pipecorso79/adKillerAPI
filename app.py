from flask import Flask, request, jsonify, abort
from flask_cors import CORS
import pickle
import json
from predicter import predict
from decoder import decode_data

def load_model(model_name = 'modelo_ver_5seg.pkl'):
    with open(model_name, 'rb') as pkl_file:
        model = pickle.load(pkl_file)
    pkl_file.close()
    return model

app = Flask(__name__)
CORS(app)
model = load_model()
duration = 5

@app.route('/')
def hello_world():
    return '¡Hola, mundo!'

@app.route('/set-model', methods=['POST'])
def set_model():
    try:
        model_data = request.get_json()
        model_number = model_data["model"]
        switch_dict = {
            3: "modelo_ver_3seg.pkl",
            5: "modelo_ver_5seg.pkl",
            7: "modelo_ver_7seg.pkl",
        }
        model = load_model(switch_dict.get(model_number, "Número no manejado"))
        duration = model_number
        return jsonify({"modelo": model_number})
    except Exception as e:
        print(f"Error al elegir modelo: {e}")
        abort(500, description = f"Error al elegir modelo: {e}")

@app.route('/analizar-audio', methods=['POST'])
def analizar_audio():
    try:
        audio_data = decode_data(request.get_json(), "audio")
        result = predict(audio_data, model, duration)
        return jsonify({"resultado": result[0], "song": result[1]})
    except Exception as e:
        print(f"Error al procesar data: {e}")
        abort(500, description = f"Error al procesar data: {e}")


if __name__ == '__main__':
    app.run()

