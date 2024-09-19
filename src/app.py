import os
import cv2
import numpy as np
import pandas as pd
import pickle
import mediapipe as mp
from flask import Flask, jsonify, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

#UPLOAD_FOLDER = '/Users/rosasainz/Documents/uam/lsm_server/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
mp_drawing = mp.solutions.drawing_utils 
mp_holistic = mp.solutions.holistic 


#modelos
model_filenames = [
    'intensidad_1', 'intensidad_2', 'localizacion_1', 'localizacion_2',
    'localizacion_3', 'localizacion_4',
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

app = Flask(__name__)
#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#crear carpeta
#if not os.path.exists(UPLOAD_FOLDER):
#    os.makedirs(UPLOAD_FOLDER)

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
    max_prob_value = None

    for prediction in predictions:
        body_language_prob = prediction.get('body_language_prob')
        if body_language_prob and len(body_language_prob) > 0:
            max_value_in_prob = (body_language_prob)
            if max_prob_value is None or max_value_in_prob > max_prob_value:
                max_prob_value = max_value_in_prob
                max_prob_dict = prediction

    return max_prob_dict


#Cargar todos los modelos
models = {name: load_model(name) for name in model_filenames}
models_temporalidad = {name: load_model(name) for name in model_temporalidad}
models_localizacion = {name: load_model(name) for name in model_localizacion}
model_intensidad = {name: load_model(name) for name in model_intensidad}

@app.route('/', methods=['GET'])
def mess():
    return jsonify("Hola")

#@app.route('/upload', methods=['POST'])
#def upload_media():
#    if 'image' not in request.files:
#        return jsonify({'error':'media not provided'}), 400
#    file = request.files['image']

#    if file.filename == '':
#        return jsonify({'error': 'no file selected'}), 400

#    if file and allowed_file(file.filename):
#        filename = secure_filename(file.filename)
#        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#        print("filename: "+file.filename)
#        file.save(file_path)
#    return jsonify({'msg':file.filename}) 

@app.route('/processTemporalidad', methods=['POST'])
def processTemporalidad():
    if 'image' not in request.files:
        return jsonify({'error':'media not provided'}), 400
    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'no file selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        #file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        #print("filename: "+file.filename)
        #file.save(file_path)
           # Leer la imagen
        image = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

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
                for name, model in models_temporalidad.items():
                    body_language_class = model.predict(X)[0]
                    body_language_prob = model.predict_proba(X)[0].tolist()
                    predictions.append({
                        'model': name,
                        'image': file.filename,
                        'body_language_class': body_language_class,
                        'body_language_prob': body_language_prob
                    })
                for prediction in predictions:
                    print(prediction)
                print("MAX PREDICTION: ",max_prediction(predictions))
                return jsonify(max_prediction(predictions))
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'Invalid file format'}), 400  

@app.route('/processLocalizacion', methods=['POST'])
def processLocalizacion():
    if 'image' not in request.files:
        return jsonify({'error':'media not provided'}), 400
    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'no file selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        #file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        #print("filename: "+file.filename)
        #file.save(file_path)
           # Leer la imagen
        image = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

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
                for name, model in models_localizacion.items():
                    body_language_class = model.predict(X)[0]
                    body_language_prob = model.predict_proba(X)[0].tolist()
                    predictions.append({
                        'model': name,
                        'image': file.filename,
                        'body_language_class': body_language_class,
                        'body_language_prob': body_language_prob
                    })
                for prediction in predictions:
                    print(prediction)
                print("MAX PREDICTION: ",max_prediction(predictions))
                return jsonify(max_prediction(predictions))
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'Invalid file format'}), 400 

@app.route('/processIntensidad', methods=['POST'])
def processIntensidad():
    if 'image' not in request.files:
        return jsonify({'error':'media not provided'}), 400
    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'no file selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        #file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        #print("filename: "+file.filename)
        #file.save(file_path)
        # Leer la imagen
        image = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

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
                for name, model in model_intensidad.items():
                    body_language_class = model.predict(X)[0]
                    body_language_prob = model.predict_proba(X)[0].tolist()
                    predictions.append({
                        'model': name,
                        'image': file.filename,
                        'body_language_class': body_language_class,
                        'body_language_prob': body_language_prob
                    })
                for prediction in predictions:
                    print(prediction)
                print("MAX PREDICTION: ",max_prediction(predictions))
                return jsonify(max_prediction(predictions))

            except Exception as e:
                return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'Invalid file format'}), 400

@app.route('/processImages', methods=['POST'])
def processImages():
    if 'image' not in request.files:
        return jsonify({'error':'media not provided'}), 400
    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'no file selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        #file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        #print("filename: "+file.filename)
        #file.save(file_path)

        # Leer la imagen
        image_read = np.frombuffer(file.read(), np.uint8)
        image_process = cv2.imdecode(image_read, cv2.IMREAD_COLOR)
        image = cv2.imread(image_process)

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
                return jsonify(predictions)
        
            except Exception as e:
                return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file format'}), 400
    
if __name__ == '__main__':  
    app.run(host="0.0.0.0", port=4000, debug=True)