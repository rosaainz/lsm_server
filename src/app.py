import os
import cv2
import numpy as np
import pandas as pd
import pickle
import mediapipe as mp
from flask import Flask, jsonify, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '/Users/rosasainz/Documents/uam/lsm_server/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
mp_drawing = mp.solutions.drawing_utils 
mp_holistic = mp.solutions.holistic 

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#verificar la extension del archivo
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#cargar el modelo
def load_model(model_filename):
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', f'{model_filename}.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"Modelo {model_filename} cargado")
    return model

def max_prediction(predictions):
    max_prob = -1.0
    best_prediction = None

    for prediction in predictions:
        body_language_prob = prediction['body_language_prob']
        prob = body_language_prob[0] if isinstance(body_language_prob, list) else body_language_prob

        if prob > max_prob:
            max_prob = prob
            best_prediction = prediction
    return best_prediction

#modelos
model_filenames = [
    'intensidad_1', 'intensidad_2', 'localizacion_1', 'localizacion_2',
    'localizacion_3', 'localizacion_4', 'numeros',
    'temporalidad_1', 'temporalidad_2', 'temporalidad_3', 'temporalidad_4'
]

model_temporalidad = [
    'temporalidad_1', 'temporalidad_2', 'temporalidad_3', 'temporalidad_4'
]

model_localizacion = [
    'localizacion_1', 'localizacion_2',
    'localizacion_3', 'localizacion_4'
]

model_intensidad = [
    'intensidad_1', 'intensidad_2'
]

#Cargar todos los modelos
models = {name: load_model(name) for name in model_filenames}
models_temporalidad = {name: load_model(name) for name in model_temporalidad}
models_localizacion = {name: load_model(name) for name in model_localizacion}
model_intensidad = {name: load_model(name) for name in model_intensidad}

@app.route('/', methods=['GET'])
def mess():
    return jsonify("Hola")

@app.route('/upload', methods=['POST'])
def upload_media():
    if 'image' not in request.files:
        return jsonify({'error':'media not provided'}), 400
    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'no file selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print("filename: "+file.filename)
        file.save(file_path)
    return jsonify({'msg':file.filename}) 

@app.route('/processImages', methods=['POST'])
def processImages():
    imageCount = None
    print(request.headers)
    if 'image' not in request.files:
        return jsonify({'error':'media not provided'}), 400
    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'no file selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print("filename: "+file.filename)
        file.save(file_path)

        # Leer la imagen
        image = cv2.imread(file_path)

        with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.6) as holistic:
            # Recolor Feed
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False

            # Make Detections
            results = holistic.process(image_rgb)

            # Export coordinates
            try:
                # Extract Pose landmarks
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                # Extract Face landmarks
                face = results.face_landmarks.landmark
                face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

                # Concatenate rows
                row = pose_row + face_row

                # Make Detections
                X = pd.DataFrame([row])
                predictions = []
                for name, model in models.items():
                    body_language_class = model.predict(X)[0]
                    body_language_prob = model.predict_proba(X)[0].tolist()
                    predictions.append({
                        'model': name,
                        'image': file.filename,
                        'body_language_class': body_language_class,
                        'body_language_prob': body_language_prob
                    })
                # Asignar el nombre de archivo de la última predicción
                filename = predictions[-1]['image'] if predictions else None
                response = {
                    "status": "OK",
                     "imageCount": filename
                }
                return jsonify(predictions)
        
            except Exception as e:
                return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file format'}), 400
    
if __name__ == '__main__':  
    app.run(host="0.0.0.0", port=4000, debug=True)