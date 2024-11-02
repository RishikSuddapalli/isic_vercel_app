import tensorflow as tf

class Model:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = ['Actinic keratosis', 'Basal cell carcinoma', 'Benign keratosis', 
                            'Dermatofibroma', 'Melanocytic nevus', 'Melanoma', 
                            'Squamous cell carcinoma', 'Vascular lesion']

    def predict(self, image):
        image = tf.image.resize(image, (240, 240))
        image = image / 255.0
        image = tf.expand_dims(image, axis=0)
        predictions = self.model.predict(image)
        return self.class_names[tf.argmax(predictions[0])]
