from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import tensorflow as tf
import io

app = Flask(__name__)
CORS(app)

# Define Model class and load the model
class Model:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
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
        predicted_class = self.class_names[tf.argmax(predictions[0])]
        confidence = float(tf.reduce_max(predictions[0]))
        return {"class": predicted_class, "confidence": confidence}

# Load the model
model_path = 'ENB1_8Class30.h5'
model = Model(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Read and process the image
        img = Image.open(file.stream).convert("RGB")
        img_array = np.array(img)

        # Make prediction
        prediction = model.predict(img_array)
        return jsonify(prediction)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

    