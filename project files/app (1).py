from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from keras.applications.efficientnet import EfficientNetB3, preprocess_input
from keras.utils import custom_object_scope
import numpy as np
import os
from PIL import Image

app = Flask(__name__)

with custom_object_scope({'EfficientNetB3': EfficientNetB3}):
    # Load the new Keras model format
    model = load_model('model/chicken_model.keras', compile=False)

classes = ['Coccidiosis', 'Healthy', 'New Castle Disease', 'Salmonella']  # Adjust to your class labels

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    file = request.files['file']
    img = Image.open(file).convert("RGB")
    img = img.resize((224, 224))  # resize as per model input
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction[0])
    result = classes[class_idx]
    confidence = float(np.max(prediction[0]))

    print(f"DEBUG: prediction={prediction}, class_idx={class_idx}, confidence={confidence}")

    return jsonify({'prediction': result, 'confidence': round(confidence * 100, 2)})

if __name__ == '__main__':
    app.run(debug=True)
