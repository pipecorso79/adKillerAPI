import librosa
import numpy as np
#from google.colab import drive
import os
import json
import math

import os
from pydub import AudioSegment


#Frecuencia en la cual se tomaran muestras en un audio
frecuencia_muestra = 22050
#TODO
#Este valor hay que cambiarlo, ya que todos los audios no tienen la misma duracion, hay que solucionar este problema...
#asumimos que todos tienen 10 segundos
duracion_audio = 5 #En segundos
num_de_muestras_por_audio = frecuencia_muestra * duracion_audio

#TODO
#Definir los path de los audios, tendriamos que tener dos, el de conversacion y el de no_conversacion
folder_path = "audio_files_cut_3seg" #path a la carpeta de los audios. C:\Users\Usuario\Desktop\MACHINELEARNINGPROYECT
json_path = "C:\\Users\\Usuario\\Desktop\\MACHINELEARNINGPROYECT\\json_data_dir\\json_5seg\\json_data_audio_cut_5seg.json" #path al archivo json donde guardar los datos extraidos de los audios


def recortar_audio(carpetA, carpetaB, duracion):
    # Verificar si la carpeta de destino existe, si no, crearla
    if not os.path.exists(carpetaB):
        os.makedirs(carpetaB)

    # Obtener la lista de archivos en la carpeta A
    archivos = os.listdir(carpetA)

    for archivo in archivos:
        if archivo.endswith(".mp3") or archivo.endswith(".wav"):
            # Cargar el archivo de audio
            audio = AudioSegment.from_file(os.path.join(carpetA, archivo))

            # Dividir el audio en segmentos de la duración especificada (en milisegundos)
            segmentos = [audio[i:i+duracion*1000] for i in range(0, len(audio), duracion*1000)]

            # Guardar los segmentos en la carpeta B
            for i, segmento in enumerate(segmentos):
                nombre_archivo = f"{os.path.splitext(archivo)[0]}_segmento_{i+1}.wav"
                segmento.export(os.path.join(carpetaB, nombre_archivo), format="wav")

def has_zeroes(matrix):
  #  if(matrix.tolist()[-1][-1] == 0):
       # print("----------")
       # print(matrix)
       # print("----------")
       # print()
       # print()
    for lista in matrix.tolist():
        if(lista[-1] == 0):
            return True
    return False


def save_mfcc(path_datos, json_path, n_mfcc = 13, n_fft = 2048, hop_length = 512, num_segmentos = 2):
  
  #diccionario para almacenar datos
    data = {
      "categoria": [], #Esto tendra la categoria del audio: conversacion o no_conversacion
      "mfcc": [], #Seria el input train, tendra todos los valores del mfcc
      "labels": [] #Seria el output, esta relacionado con a que categoria representa un mfcc
    }
    
    bad_data = {
        "categoria": [], #Esto tendra la categoria del audio: conversacion o no_conversacion
        "mfcc": [], #Seria el input train, tendra todos los valores del mfcc
        "labels": [] 
    }

  
    num_de_muestras_por_segmento = int(num_de_muestras_por_audio / num_segmentos) 
  
  #Como necesitamos que todos los valores de entrenamiento del modelo tengan la misma forma
  #definimos un numero fijo de vectores mfcc a utilizar.
    num_de_vectores_mfcc_por_segmento = math.ceil(num_de_muestras_por_segmento / hop_length)

  #TODO
  #COMO TENEMOS DOS FUENTES DE DATOS, VER DE SHUFFLEAR AL FINAL POR LAS DUDAS DE QUE NO SE USEN CORRECTAMENTE LOS DATOS
  #AL MOMENTO DEL ENTRENAMIENTO

  #dirpath: el path del directorio actual
  #dirname: el nombre del directorio
  #filenames: los nombres de los archivos del directorio
  #indice tiene el numero de la iteracion, que representara el genero
    for indice, (dirpath, dirnames, filenames) in enumerate(os.walk(folder_path)):

    #No queremos estar en la carpeta principal, sino en la de los generos que tiene los audios
        if dirpath is not folder_path:

      #separamos el path por cada / en el path en un array y agarramos el ultimo, que seria la categoria 
            folder_categoria = dirpath.split("\\")[-1]
            data["categoria"].append(folder_categoria)
            bad_data["categoria"].append(folder_categoria)
      #print(f"\nProcesando {folder_genero}")

      #Obtenemos los audios del directorio
            for audio_name in filenames:

                audio_path = os.path.join(dirpath, audio_name)
                signal, sr = librosa.load(audio_path, sr=frecuencia_muestra)

        #TODO
        #Otra consideracion, como la duracion de los audios de ruido ya es corta (5 seg)
        #Pensamos dejar el recorte solamente para los demas audios
        
        #Como tenemos pocos datos, vamos a dividir el procesamiento de los
        #audios en diferentes segmentos, entonces para un audio tendriamos
        #varios segmentos del mismo para usar como datos de entrenamiento
        #o testeo.
                for segmento in range(num_segmentos):

          #definimos el inicio y final del actual segmento
                    inicio_de_muestra = num_de_muestras_por_segmento * segmento 
                    fin_de_muestra = inicio_de_muestra + num_de_muestras_por_segmento

              # Extraemos el mfcc solo del intervalo definido arriba.
                    if(inicio_de_muestra < signal.size and fin_de_muestra < signal.size):
                        mfcc = librosa.feature.mfcc(y=signal[inicio_de_muestra: fin_de_muestra],
                                              sr=sr,
                                              n_fft = n_fft,
                                              n_mfcc = n_mfcc,
                                              hop_length = hop_length)


                        
                        mfcc = mfcc.T
                        print(mfcc)
#                print()
#                print()
#                print(mfcc.tolist())
#                print()
#                print()
#                print(mfcc.tolist()[-1][-1])
#                print()
#                print()
#                break
                        if len(mfcc) == num_de_vectores_mfcc_por_segmento and not has_zeroes(mfcc):
                            data["mfcc"].append(mfcc.tolist())
                            data["labels"].append(indice - 1)
                        else:
                            bad_data["mfcc"].append(mfcc.tolist())
                            bad_data["labels"].append(indice - 1)
                    #print(f"{audio_path}, segmento:{segmento}")
  
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
    
    with open("bad_data_cut_5seg.json", "w") as fp:
        json.dump(bad_data, fp, indent=4)

save_mfcc(path_datos = folder_path, json_path=json_path)