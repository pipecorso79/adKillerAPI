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
        return jsonify({"resultado": result})
    except Exception as e:
        print(f"Error al procesar data: {e}")
        abort(500, description = f"Error al procesar data: {e}")

def procesar_audio_shazam(output_file_wav):
    output_file_mp3 = "a.mp3"
    audio = AudioSegment.from_file(output_file_wav, format="wav")
    audio.export(output_file_mp3, format="mp3")
    mp3_file_content_to_recognize = open('a.mp3', 'rb').read()

    shazam = Shazam(
        mp3_file_content_to_recognize
    )

    recognize_generator = shazam.recognizeSong()
    # if len(next(resultado_shazam)[1]['matches']) > 0:
    #             print("La clave 'matches' no está vacía.")
    #         else:
    #             print("La clave 'matches' está vacía.")

#     from pydub import AudioSegment
# from io import BytesIO

#     Ruta del archivo WAV de entrada (reemplázala con tu propio archivo WAV)
# input_wav_file = "entrada.wav"

# Cargar el archivo WAV
# audio = AudioSegment.from_wav(input_wav_file)

# Crear un objeto BytesIO para almacenar el audio en memoria
# output_buffer = BytesIO()

# Convertir el audio a formato MP3 y guardar en el objeto BytesIO
# audio.export(output_buffer, format="mp3")

# Obtener los datos MP3 como bytes
# mp3_data = output_buffer.getvalue()

# Ahora, mp3_data contiene los datos del archivo MP3 en memoria
# Puedes guardarlos, transmitirlos por la red o realizar otras operaciones con mp3_data

    return recognize_generator

if __name__ == '__main__':
    app.run()

