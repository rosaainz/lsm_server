# Librerías a utilizar
# Esta librería se utiliza para interactuar con el sistema operativo, permitiendo manejar archivos y directorios.
import os
# Esta librería se utiliza para realizar operaciones de procesamiento de imágenes y visión por computadora.
import cv2
# Esta librería se utiliza para trabajar con arreglos y matrices, facilitando cálculos numéricos y operaciones matemáticas.
import numpy as np
# Esta librería se utiliza para la manipulación y análisis de datos, especialmente con estructuras como DataFrames.
import pandas as pd
# Esta librería se utiliza para serializar y deserializar objetos de Python, facilitando el almacenamiento y la carga de modelos.
import pickle
# Esta librería se utiliza para realizar tareas de detección y seguimiento de objetos en imágenes y videos, utilizando modelos de aprendizaje automático.
import mediapipe as mp
# Esta librería se utiliza para crear aplicaciones web y manejar solicitudes y respuestas HTTP de manera sencilla.
from flask import Flask, jsonify, flash, request, redirect, url_for
# Esta librería se utiliza para asegurar que los nombres de archivos sean seguros y apropiados para su uso en el sistema de archivos.
from werkzeug.utils import secure_filename

# Definir extensiones de archivo permitidas
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

# Inicializar herramientas de Mediapipe
mp_drawing = mp.solutions.drawing_utils 
mp_holistic = mp.solutions.holistic 


# Listas de nombres de modelos para diferentes clasificaciones
model_filenames = [
    'intensidad_1', 'intensidad_2', 'localizacion_1', 'localizacion_2',
    'localizacion_3', 'localizacion_4',
    'temporalidad_1', 'temporalidad_2', 'temporalidad_3', 'temporalidad_4'
]

# Agrupación de modelos por categoría
model_temporalidad = ['temporalidad_1', 'temporalidad_2', 'temporalidad_3', 'temporalidad_4']
model_localizacion = ['localizacion_1', 'localizacion_2', 'localizacion_3', 'localizacion_4']
model_intensidad = ['intensidad_1', 'intensidad_2']

# Crear la aplicación Flask
app = Flask(__name__)

# La función allowed_fi se utiliza para verificar si el archivo tiene una extensión permitida
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# La función load_model sirve para cargar un modelo desde un archivo
def load_model(model_filename):
     # Construir la ruta del modelo
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', f'{model_filename}.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f) # Cargar el modelo usando pickle
    print(f"Modelo {model_filename} cargado") 
    return model

# Función para encontrar la predicción máxima entre varias prediccione
def max_prediction(predictions):
    max_prob_value = None

    for prediction in predictions:
        body_language_prob = prediction.get('body_language_prob')
        if body_language_prob and len(body_language_prob) > 0:
            max_value_in_prob = max(body_language_prob) # Obtener el valor máximo de probabilidad
            if max_prob_value is None or max_value_in_prob > max_prob_value:
                max_prob_value = max_value_in_prob
                max_prob_dict = prediction # Actualizar la predicción máxima

    return max_prob_dict

# Cargar todos los modelos
models = {name: load_model(name) for name in model_filenames}
models_temporalidad = {name: load_model(name) for name in model_temporalidad}
models_localizacion = {name: load_model(name) for name in model_localizacion}
model_intensidad = {name: load_model(name) for name in model_intensidad}

# Ruta principal que devuelve un saludo, esto para probar una correcta conexion con la API
@app.route('/', methods=['GET'])
def mess():
    return jsonify("Hola")

# Ruta para procesar imágenes relacionadas con la temporalidad
@app.route('/processTemporalidad', methods=['POST'])
def processTemporalidad():
    # Comprobar si se ha enviado un archivo de imagen
    if 'image' not in request.files:
        return jsonify({'error':'media not provided'}), 400
    file = request.files['image']

    # Comprobar si el archivo tiene nombre
    if file.filename == '':
        return jsonify({'error': 'no file selected'}), 400

    # Verificar que el archivo tenga una extensión permitida
    if file and allowed_file(file.filename):
        # Leer la imagen en un formato adecuado
        image = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.6) as holistic:
            # Convertir la imagen a RGB para el procesamiento
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False

            # Realizar las detecciones en la imagen
            results = holistic.process(image_rgb)

            # Estructura de control try para extraer las coordenadas de los landmarks
            try:
                # Extraer landmarks del cuerpo
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                #  Extraer landmarks de la cara
                face = results.face_landmarks.landmark
                face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

                # Concatenar filas de datos
                row = pose_row + face_row

                # Crear DataFrame para hacer predicciones
                X = pd.DataFrame([row])
                predictions = []
                
                # Hacer predicciones usando los modelos de temporalidad
                for name, model in models_temporalidad.items():
                    body_language_class = model.predict(X)[0]
                    body_language_prob = model.predict_proba(X)[0].tolist()
                    predictions.append({
                        'model': name,
                        'image': file.filename,
                        'body_language_class': body_language_class,
                        'body_language_prob': body_language_prob
                    })
                
                # Imprimir predicciones y devolver la máxima
                print("MAX PREDICTION: ",max_prediction(predictions))
                return jsonify(max_prediction(predictions))
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'Invalid file format'}), 400  

# Ruta para procesar imágenes relacionadas con la localización
@app.route('/processLocalizacion', methods=['POST'])
def processLocalizacion():
     # Comprobar si se ha enviado un archivo de imagen
    if 'image' not in request.files:
        return jsonify({'error':'media not provided'}), 400
    file = request.files['image']

    # Comprobar si el archivo tiene nombre
    if file.filename == '':
        return jsonify({'error': 'no file selected'}), 400

    # Verificar que el archivo tenga una extensión permitida
    if file and allowed_file(file.filename):
         # Leer la imagen en un formato adecuado
        image = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.6) as holistic:
            # Convertir la imagen a RGB para el procesamiento
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False

             # Realizar las detecciones en la imagen
            results = holistic.process(image_rgb)

            # Estructura de control try para  extraer las coordenadas de los landmarks
            try:
                # Extraer landmarks del cuerpo
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                # Extraer landmarks de la cara
                face = results.face_landmarks.landmark
                face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

                # Concatenar filas de datos
                row = pose_row + face_row

                # Crear DataFrame para hacer prediccione
                X = pd.DataFrame([row])
                predictions = []

                # Hacer predicciones usando los modelos de temporalidad
                for name, model in models_localizacion.items():
                    body_language_class = model.predict(X)[0]
                    body_language_prob = model.predict_proba(X)[0].tolist()
                    predictions.append({
                        'model': name,
                        'image': file.filename,
                        'body_language_class': body_language_class,
                        'body_language_prob': body_language_prob
                    })

                # Imprimir predicciones y devolver la máxima
                print("MAX PREDICTION: ",max_prediction(predictions))
                return jsonify(max_prediction(predictions))
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'Invalid file format'}), 400 

# Ruta para procesar imágenes relacionadas con la intensidad
@app.route('/processIntensidad', methods=['POST'])
def processIntensidad():
    # Comprobar si se ha enviado un archivo de imagen
    if 'image' not in request.files:
        return jsonify({'error':'media not provided'}), 400
    file = request.files['image']

    # Comprobar si el archivo tiene nombre
    if file.filename == '':
        return jsonify({'error': 'no file selected'}), 400

    # Verificar que el archivo tenga una extensión permitida
    if file and allowed_file(file.filename):
        # Leer la imagen en un formato adecuado
        image = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.6) as holistic:
            # Convertir la imagen a RGB para el procesamiento
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False

            # Realizar las detecciones en la imagen
            results = holistic.process(image_rgb)

            # Estructura de control try para  extraer las coordenadas de los landmarks
            try:
                # Extraer landmarks del cuerpo
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                # Extraer landmarks de la cara
                face = results.face_landmarks.landmark
                face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

                # Concatenar filas de datos
                row = pose_row + face_row

                # Crear DataFrame para hacer predicciones
                X = pd.DataFrame([row])
                predictions = []

                # Hacer predicciones usando los modelos de temporalidad
                for name, model in model_intensidad.items():
                    body_language_class = model.predict(X)[0]
                    body_language_prob = model.predict_proba(X)[0].tolist()
                    predictions.append({
                        'model': name,
                        'image': file.filename,
                        'body_language_class': body_language_class,
                        'body_language_prob': body_language_prob
                    })
                
                # Imprimir predicciones y devolver la máxima
                print("MAX PREDICTION: ",max_prediction(predictions))
                return jsonify(max_prediction(predictions))

            except Exception as e:
                return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'Invalid file format'}), 400
    
# Verifica si el archivo se está ejecutando directamente
#if __name__ == '__main__':  
    # Inicia el servidor Flask
    # - host="0.0.0.0": permite que el servidor sea accesible desde cualquier dirección IP en la red
    # - port=4000: especifica el puerto en el que el servidor escuchará las solicitudes
    # - debug=True: activa el modo de depuración, lo que permite ver errores en el navegador
    #   y recarga automática del servidor al realizar cambios en el código
    #app.run(host="0.0.0.0", port=8080, debug=True)