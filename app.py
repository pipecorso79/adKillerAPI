from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow.keras as keras
import pickle #Para exportar el modelo
import numpy as np
import json
import random

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello_world():
    return '¡Hola, mundo!'

@app.route('/analizar-audio', methods=['POST'])
def analizar_audio():
    try:
        # audio_file = request.files['archivo_audio']
        print("Hola aaaaaa")
        modelo = cargar_modelo()
        ##modelo.summary()

        X_test, y_test = cargar_datos_test()
        print("Estos son los datos de test")
        print(X_test.shape)
        print(y_test.shape)
        print("Ahora hacemos una prediccion")
        # Aquí puedes agregar la lógica para analizar el archivo de audio
        # y devolver los resultados en formato JSON
        resultado = predecir_random(X_test, y_test, modelo)
        print(resultado)
        print(jsonify({"resultado": resultado}))
        return jsonify({"resultado": resultado})
    except Exception as e:
        return jsonify({"error": str(e)})

def cargar_modelo():
    with open('audio_model_mejoras_10.pkl', 'rb') as archivo_pkl:
        modelo = pickle.load(archivo_pkl)
    archivo_pkl.close()
    return modelo

def cargar_datos_test():
    with open("C:\\Users\\Usuario\\Desktop\\MACHINELEARNINGPROYECT\\matrices_X_test", "r") as fp:
        data_X = json.load(fp)

    with open("C:\\Users\\Usuario\\Desktop\\MACHINELEARNINGPROYECT\\matrices_y_test", "r") as fp:
        data_y = json.load(fp)

    X_test = np.array(data_X)
    y_test = np.array(data_y)

    return X_test, y_test

def predict(X, y, model):
    X = X[np.newaxis, ...]
    prediccion = model.predict(X)
    prediccion = np.argmax(prediccion, axis=1)
    print(f"Esperado: {y} Prediccion: {prediccion}")
    return prediccion

def predecir_random(X_test, y_test, model):
    size = X_test.shape[0]
    indice_aleatorio = random.randint(0, size - 1)

    return predict(X_test[indice_aleatorio], y_test[indice_aleatorio], model).tolist()[0]

if __name__ == '__main__':
    app.run()

