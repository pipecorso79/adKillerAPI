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

def cargar_modelo():
    with open('modelo_ver_5seg.pkl', 'rb') as archivo_pkl:
        modelo = pickle.load(archivo_pkl)
    archivo_pkl.close()
    return modelo

def procesar_audio(audio_file, model, duracion_audio):
    prediccion = 10
    try:
        print("antes de mfcc")
        data = generar_mfcc(audio_file = audio_file, duracion_audio = duracion_audio)
        data = data["mfcc"]
        print("despues de mfcc")
        print(data)
        # X = data[np.newaxis, ...]
        X = np.array(data)
        prediccion = model.predict(X)
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
            print("\n{}\n{}".format(len(mfcc), num_de_vectores_mfcc_por_segmento))
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


if __name__ == '__main__':
    modelo = cargar_modelo()
    output_file_wav= 'audio_20231105_194351_d01df909.wav'
    resultado = procesar_audio(output_file_wav, modelo, 5)

    