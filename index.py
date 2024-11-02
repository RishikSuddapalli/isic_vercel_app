from flask import Flask, request, render_template
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import requests
import os

class Model:
    def __init__(self, file_id):
        # Construct Google Drive download URL
        model_filename = 'ENB1_8Class30.h5'
        download_url = f"https://drive.google.com/file/d/{file_id}"

        # Download the model if it hasn't been downloaded already
        if not os.path.exists(model_filename):
            print("Downloading model from Google Drive...")
            response = requests.get(download_url, stream=True)
            with open(model_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print("Download complete.")
        
        # Load the model
        self.model = tf.keras.models.load_model(model_filename)
        self.class_names = [
            'Actinic keratosis', 'Basal cell carcinoma', 'Benign keratosis',
            'Dermatofibroma', 'Melanocytic nevus', 'Melanoma',
            'Squamous cell carcinoma', 'Vascular lesion'
        ]

    def predict(self, image):
        image = tf.image.resize(image, (240, 240))
        image = image / 255.0
        image = tf.expand_dims(image, axis=0)
        predictions = self.model.predict(image)
        return self.class_names[tf.argmax(predictions[0])]

app = Flask(__name__)
CORS(app)

# Load the model
model = Model("1mOPikpQYetgI4hxKXEcZWsPoMe0HpIBF/view?usp=sharing")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"

    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    # Read the image
    img = Image.open(file.stream)
    img_array = np.array(img)

    # Make prediction
    prediction = model.predict(img_array)
    return prediction

if __name__ == '__main__':
    app.run(debug=True) 
