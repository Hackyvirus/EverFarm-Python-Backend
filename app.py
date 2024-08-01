from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf
import io

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model('D:/Web Dev/Imp/pythonbackend/AI/model.json')

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    img = Image.open(file.stream)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)

    response = {
        'predicted_class': str(predicted_class[0]),
        'confidence': float(np.max(predictions))
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
