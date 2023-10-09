from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def hello_world():
    return '¡Hola, mundo!'

@app.route('/analizar-audio', methods=['POST'])
def analizar_audio():
    audio_file = request.files['archivo_audio']
    # Aquí puedes agregar la lógica para analizar el archivo de audio
    # y devolver los resultados en formato JSON

if __name__ == '__main__':
    app.run()

