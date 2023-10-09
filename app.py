from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def hello_world():
    return '¡Hola, mundo!'

@app.route('/analizar-audio', methods=['POST'])
def analizar_audio(campo):
    # audio_file = request.files['archivo_audio']
    print("Hola aaaaaa")
    # Aquí puedes agregar la lógica para analizar el archivo de audio
    # y devolver los resultados en formato JSON
    return {'campo': 'campo2'}

if __name__ == '__main__':
    app.run()

