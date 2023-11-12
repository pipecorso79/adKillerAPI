import tensorflow
from audio_file_generator import convert_to_wav
import asyncio
import librosa
import math
import numpy as np
import os


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