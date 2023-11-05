from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import pickle
import numpy as np
import tensorflow
import json
import os
import random
import math
from pydub import AudioSegment
# AudioSegment.ffmpeg = "C:\\FFMPEG_FILES_DIR\\ffmpeg\\bin\\ffmpeg.exe"
# AudioSegment.converter = "C:\\FFMPEG_FILES_DIR\\ffmpeg\\bin\\ffmpeg.exe"
import datetime
import uuid
import librosa

app = Flask(__name__)
CORS(app)

def cargar_modelo():
    with open('modelo_ver_5seg.pkl', 'rb') as archivo_pkl:
        modelo = pickle.load(archivo_pkl)
    archivo_pkl.close()
    return modelo

def generate_unique_filename():
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4()).split("-")[0]  # Obtiene el primer segmento del UUID
    return f"audio_{current_time}_{unique_id}"

def procesar_audio(audio_file, model, duracion_audio):
    prediccion = 10
    try:
        print("antes de mfcc")
        data = generar_mfcc(audio_file = audio_file, duracion_audio = duracion_audio)
        data = data["mfcc"]
        print("despues de mfcc")
        # print(data)
        # X = data[np.newaxis, ...]
        X = np.array(data)
        prediccion = model.predict(X)
        prediccion = np.argmax(prediccion, axis=1).tolist()[0]
    except Exception as e:
        print(f"Error con mfcc: {e}")
        return 10
    return prediccion

def generar_mfcc(audio_file, duracion_audio, n_mfcc = 13, n_fft = 2048, hop_length = 512, num_segmentos = 1, frecuencia_muestra = 22050):
    num_de_muestras_por_audio = frecuencia_muestra * duracion_audio
    data = {
        "mfcc": []
    }
    num_de_muestras_por_segmento = int(num_de_muestras_por_audio / num_segmentos) 
    num_de_vectores_mfcc_por_segmento = math.ceil(num_de_muestras_por_segmento / hop_length)
    signal, sr = librosa.load(audio_file, sr=frecuencia_muestra)

    print(f"Longitud de la señal: {len(signal)}")
    for segmento in range(num_segmentos):
        inicio_de_muestra = num_de_muestras_por_segmento * segmento 
        fin_de_muestra = inicio_de_muestra + num_de_muestras_por_segmento

        print(f"Inicio de muestra: {inicio_de_muestra}, Fin de muestra: {fin_de_muestra}")
        if(num_segmentos == 1):
            mfcc = librosa.feature.mfcc(y=signal[inicio_de_muestra:fin_de_muestra],
                                    sr=sr,
                                    n_fft = n_fft,
                                    n_mfcc = n_mfcc,
                                    hop_length = hop_length)
            mfcc = mfcc.T
            print(f"Forma de MFCC: {mfcc.shape}")
            if len(mfcc) == num_de_vectores_mfcc_por_segmento:
                data["mfcc"].append(mfcc.tolist())
        elif(inicio_de_muestra < signal.size and fin_de_muestra < signal.size):
            mfcc = librosa.feature.mfcc(y=signal[inicio_de_muestra: fin_de_muestra],
                                    sr=sr,
                                    n_fft = n_fft,
                                    n_mfcc = n_mfcc,
                                    hop_length = hop_length)
            mfcc = mfcc.T
            print(f"Forma de MFCC: {mfcc.shape}")
            if len(mfcc) == num_de_vectores_mfcc_por_segmento and not has_zeroes(mfcc):
                data["mfcc"].append(mfcc.tolist())
    return data

modelo = cargar_modelo()

@app.route('/')
def hello_world():
    return '¡Hola, mundo!'

@app.route('/analizar-audio', methods=['POST'])
def analizar_audio():
    print("LLegue a la API")
    try:
        # audio_file = request.files['archivo_audio']
        audio_data = request.get_json()

        base64_string = audio_data["audio"]  # El campo que contiene la cadena base64 en tu JSON
        #print("Data en string: \n{}".format(base64_string))
        audio_binary = base64.b64decode(base64_string)

        output_file_webm = f"{generate_unique_filename()}.webm"
        output_file_wav = f"{generate_unique_filename()}.wav"

        print("Guardando el audio en webM...")
        with open(output_file_webm, 'wb') as audio_file:
            audio_file.write(audio_binary)

        try:
            print("Generando sound...\n")
            sound = AudioSegment.from_file(output_file_webm, format="webm")#, codec="libopus")
            print("Generando export wax... \n")
            sound.export(output_file_wav, format="wav")
        except Exception as e:
            print(f"Error al convertir de WebM a WAV: {e}")
        # print("Objeto recibido!!")
        # print(audio_binary)
        # print("Fin del objeto...")

        # X_test, y_test = cargar_datos_test()
        # print("Prediciendo...")
        # resultado = predecir_random(X_test, y_test, modelo)
        # print("Prediccion: {} {}".format(
        #     resultado,
        #     "conversacion" if (resultado == 0) else "no conversacion"))
        try:
            # X_test, y_test = cargar_datos_test()
            # resultado = predecir_random(X_test, y_test, modelo)
            resultado = procesar_audio(output_file_wav, modelo, 5)
            # resultado_shazam = procesar_audio_shazam(output_file_wav)
            # print("Resultado de shazam: {}".format(resultado_shazam))
            return jsonify({"resultado": resultado})
        except Exception as e:
            print(f"Error al predecir: {e}")
            return jsonify({"resultado": 10})
    except Exception as e:
        return jsonify({"error": str(e)})

def procesar_audio_shazam(output_file_wav):
    shazam = Shazam(
        output_file_wav,
        lang='es',
        time_zone='Europe/Paris'
    )
    recognize_generator = shazam.recognizeSong()

    return recognize_generator

#Devuelve dos nparrays con datos de input y output para testing
def cargar_datos_test():
    with open("C:\\Users\\Usuario\\Desktop\\MACHINELEARNINGPROYECT\\matrices_X_test", "r") as fp:
        data_X = json.load(fp)

    with open("C:\\Users\\Usuario\\Desktop\\MACHINELEARNINGPROYECT\\matrices_y_test", "r") as fp:
        data_y = json.load(fp)

    X_test = np.array(data_X)
    y_test = np.array(data_y)

    return X_test, y_test

#Retorna la prediccion del modelo para el dato dado y se compara con la respuesta esperada
def predecir(X, y, model):
    X = X[np.newaxis, ...]
    prediccion = model.predict(X)
    prediccion = np.argmax(prediccion, axis=1)
    print(f"Esperado: {y} Prediccion: {prediccion}")
    return prediccion

#Devuelve un valor 0 o 1 dependiendo de si el modelo identifica conversacion o publicidad respectivamente
def predecir_random(X_test, y_test, model):
    size = X_test.shape[0]
    indice_aleatorio = random.randint(0, size - 1)

    return predecir(X_test[indice_aleatorio], y_test[indice_aleatorio], model).tolist()[0]

if __name__ == '__main__':
    app.run()

