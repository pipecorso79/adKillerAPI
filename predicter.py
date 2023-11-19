import tensorflow
from audio_file_generator import convert_to_wav
import asyncio
import librosa
import math
import numpy as np
import os
from ShazamAPI import Shazam
from pydub import AudioSegment


def predict(audio_data, model, audio_duration):
    wav_file, webM_file = convert_to_wav(audio_data)
    try:
        result = do_predict(wav_file, model, audio_duration)
    except Exception as e:
        asyncio.run(delete_file(wav_file))
        asyncio.run(delete_file(webM_file))
        raise Exception(e)
    asyncio.run(delete_file(wav_file))
    asyncio.run(delete_file(webM_file))
    return result

def do_predict(wav_file, model, audio_duration):
    mfcc_data = generate_mfcc(audio_file = wav_file, audio_duration = audio_duration)
    mfcc_data = mfcc_data["mfcc"]

    X = np.array(mfcc_data)
    prediction = model.predict(X)
    shazam_prediction_generator = predict_song(wav_file)
    shazam_response = next(shazam_prediction_generator)
    

    if len(shazam_response[1]['matches']) > 0:
        print("La clave 'matches' no está vacía.")
        song = shazam_response[1]['track']['title'] + '\n' + shazam_response[1]['track']['subtitle']
        # falta sacar de aca el nombre de la cancion para poder paralo por result y mandarlo x el POST
        return 2, song
    else:
        song=''
        print("La clave 'matches' está vacía.")
        return np.argmax(prediction, axis=1).tolist()[0], song

    # print('========================== LA PREDICCION DE SHAZAM: ==========================')
    # print(shazam_prediction)

    return np.argmax(prediction, axis=1).tolist()[0]

def generate_mfcc(audio_file, audio_duration, n_mfcc = 13, n_fft = 2048, hop_length = 512, sample_rate = 22050):
    samples_per_audio = sample_rate * audio_duration
    data = {
        "mfcc": []
    }
    mfcc_vectors_per_segment = math.ceil(samples_per_audio / hop_length)
    signal, sr = librosa.load(audio_file, sr=sample_rate)

    sample_start = 0
    sample_end = sample_start + samples_per_audio

    mfcc = librosa.feature.mfcc(y=signal[sample_start:sample_end],
                            sr=sr,
                            n_fft = n_fft,
                            n_mfcc = n_mfcc,
                            hop_length = hop_length)
    mfcc = mfcc.T
    if len(mfcc) == mfcc_vectors_per_segment:
        data["mfcc"].append(mfcc.tolist())
    else:
        raise Exception("La cantidad de datos mfcc generadas no coinciden con la duracion de audio proporiconada. Verifique que la duracion del audio sea la indicada") 
    return data

async def delete_file(audio_file):
    try:
        os.remove(audio_file)
    except FileNotFoundError:
        print(f"El archivo '{audio_file}' no existe.")
    except Exception as e:
        print(f"Error al intentar eliminar el archivo '{audio_file}': {e}")


def predict_song(output_file_wav):
    output_file_mp3 = "a.mp3"
    audio = AudioSegment.from_file(output_file_wav, format="wav")
    audio.export(output_file_mp3, format="mp3")
    mp3_file_content_to_recognize = open('a.mp3', 'rb').read()

    shazam = Shazam(
        mp3_file_content_to_recognize
    )

    recognize_generator = shazam.recognizeSong()

    return recognize_generator
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