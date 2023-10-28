from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import pickle
import numpy as np
import tensorflow
import json
import random

app = Flask(__name__)
CORS(app)

def cargar_modelo():
    with open('audio_model_mejoras_10.pkl', 'rb') as archivo_pkl:
        modelo = pickle.load(archivo_pkl)
    archivo_pkl.close()
    return modelo

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
        audio_binary = base64.b64decode(base64_string)
        # print("Objeto recibido!!")
        # print(audio_binary)
        # print("Fin del objeto...")

        X_test, y_test = cargar_datos_test()
        print("Prediciendo...")
        resultado = predecir_random(X_test, y_test, modelo)
        print("Prediccion: {} {}".format(
            resultado,
            "conversacion" if (resultado == 0) else "no conversacion"))
        return jsonify({"resultado": resultado})
    except Exception as e:
        return jsonify({"error": str(e)})

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

