from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf
import io

app = Flask(__name__)
CORS(app)  # Allow all origins for CORS

# Load your trained model (make sure to replace 'model.h5' with your model file)
model = tf.keras.models.load_model('D:/Web Dev/Imp/pythonbackend/AI/model.json')

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Read the image and perform prediction
    img = Image.open(file.stream)
    img = img.resize((224, 224))  # Resize to match model input size
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)

    # Create a response (you can customize this according to your model's output)
    response = {
        'predicted_class': str(predicted_class[0]),
        'confidence': float(np.max(predictions))
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
