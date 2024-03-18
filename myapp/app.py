from flask import Flask, request, jsonify
from flask_cors import CORS  # Importa CORS
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import io

# Cargar el modelo de TensorFlow
modelo = load_model('lemon_model.h5')

app = Flask(__name__)
CORS(app)  # Habilita CORS para toda la aplicación

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'file is required'}), 400
    file = request.files['file']
    img = Image.open(io.BytesIO(file.read()))
    img = ImageOps.grayscale(img)  # Convertir a escala de grises
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.reshape(img_array, (1, 150, 150, 1))  # Asegurarse de que tiene 1 canal
    prediction = modelo.predict(img_array)
    predicted_class = 'limón en mal estado' if prediction[0][0] > 0.5 else 'limón en buen estado'
    
    # Devolver directamente la clase predicha en la respuesta
    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
