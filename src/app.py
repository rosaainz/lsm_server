import os
from flask import Flask, jsonify, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from users import users

UPLOAD_FOLDER = '/Users/rosasainz/Documents/uam/lsm_server/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#verificar la extension del archivo
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

if __name__ == '__main__':  
    app.run(host="0.0.0.0", port=4000, debug=True)