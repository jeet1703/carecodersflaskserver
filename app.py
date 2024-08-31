from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.models import load_model

app = Flask(__name__)
CORS(app)  # Enable CORS to allow requests from the React app

model = load_model('skin.keras')  # Load the pre-trained model

def img_to_array(img, target_size=(224, 224)):
    img = img.resize(target_size)
    numpydata = np.asarray(img)
    return numpydata

def predict_skin_cancer(image):
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    out = (np.argmax(model.predict(image_array), axis=1))[0]
    return 'benign' if out == 0 else 'malignant'

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        image = Image.open(file)
        result = predict_skin_cancer(image)
        return jsonify({'result': result})

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "Server is running"}), 200

if __name__ == "__main__":
    app.run(debug=True)
