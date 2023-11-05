import tensorflow as tf
import pickle

# # Cargar el modelo
# model = tf.keras.models.load_model('ruta/al/modelo')

def cargar_modelo():
    with open('modelo_ver_5seg.pkl', 'rb') as archivo_pkl:
        modelo = pickle.load(archivo_pkl)
    archivo_pkl.close()
    return modelo

if __name__ == '__main__':
    modelo = cargar_modelo()
    modelo.summary()
