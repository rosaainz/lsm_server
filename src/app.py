import os
import cv2
import numpy as np
import pandas as pd
import pickle
import mediapipe as mp
from flask import Flask, jsonify, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from users import users

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
    print("Modelo cargado")
    return model

#Obtener modelo
model = load_model("temporalidad_3")

@app.route('/', methods=['GET'])
def ping():
    return jsonify({"response": "hello world"})

@app.route('/users')
def userHandler():
    return jsonify({"users": users})

@app.route('/users', methods=['POST'])
def add_user():
    return jsonify(request.get_json())

@app.route('/users/<int:id>')
def get_user_by_id(id):
    return_value={}
    print(type(id))
    for user in users:
        if user["id"] == id:
            return_value={
                'name': user["name"],
                'lastname':user["lastname"]
            }
    return jsonify(return_value)

@app.route('/upload', methods=['POST'])
def upload_media():
    if 'file' not in request.files:
        return jsonify({'error':'media not provided'}), 400
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'no file selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return jsonify({'msg':'media uploaded successfully'}) 

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error':'media not provided'}), 400
    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'no file selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return jsonify({'msg':'media uploaded successfully'}) 

if __name__ == '__main__':  
    app.run(host="0.0.0.0", port=4000, debug=True)