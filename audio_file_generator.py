from pydub import AudioSegment
import datetime
import uuid
import os

def generate_unique_filename():
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4()).split("-")[0]
    return f"audio_{current_time}_{unique_id}"

def convert_to_wav(audio_data, path = ''):
    webM_file = path + f"{generate_unique_filename()}.webm"
    wav_file = path + f"{generate_unique_filename()}.wav"
    try:
        with open(webM_file, 'wb') as audio_file:
            audio_file.write(audio_data)
        sound = AudioSegment.from_file(webM_file, format="webm")
        sound.export(wav_file, format="wav")
        return wav_file, webM_file
    except Exception as e:
        print(f"Error en la conversion: {e}")