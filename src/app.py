import os
from flask import Flask, jsonify, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from users import users

app = Flask(__name__)


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




if __name__ == '__main__':  
    app.run(host="0.0.0.0", port=4000, debug=True)