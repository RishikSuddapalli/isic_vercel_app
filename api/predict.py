from flask import Flask, request
from flask_cors import CORS
import numpy as np
from PIL import Image
import tensorflow as tf
import io

app = Flask(__name__)
CORS(app)

# Load the model
model_path = 'ENB1Classification_projecte15.h5'  # Update this path
model = tf.keras.models.load_model(model_path)
class_names = ['Actinic keratosis', 'Basal cell carcinoma', 'Benign keratosis', 
               'Dermatofibroma', 'Melanocytic nevus', 'Melanoma', 
               'Squamous cell carcinoma', 'Vascular lesion']

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # Read the image
    img = Image.open(file.stream)
    img_array = np.array(img)

    # Preprocess the image
    img_array = tf.image.resize(img_array,(224,224))
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    return predicted_class

if __name__ == '__main__':
    app.run()
