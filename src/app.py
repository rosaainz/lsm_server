import os
from flask import Flask, jsonify, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from users import users

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#crear carpeta
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/', methods=['GET'])
def ping():
    return jsonify({"response": "hello world"})

@app.route('/users')
def userHandler():
    return jsonify({"users": users})



if __name__ == '__main__':  
    app.run(host="0.0.0.0", port=4000, debug=True)