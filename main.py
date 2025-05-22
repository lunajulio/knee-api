from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
import tempfile
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Habilitar CORS para todas las rutas

# Cargar el modelo al iniciar la aplicación
MODEL_PATH = 'model_mobilenetv2_binary_final.h5'
model = load_model(MODEL_PATH)
inv_class_indices = {0: 'negative', 1: 'positive'}

def preprocess_knee_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen en {image_path}")
    height, width = img.shape[:2]
    if height <= 120:
        raise ValueError(f"La imagen no tiene altura suficiente para el recorte: {img.shape}")
    cropped = img[60:height-60, :]
    resized = cv2.resize(cropped, (224, 104))
    padded = cv2.copyMakeBorder(resized, 60, 60, 0, 0, cv2.BORDER_CONSTANT, value=0)
    equalized = cv2.equalizeHist(padded)
    equalized_rgb = np.stack([equalized]*3, axis=-1)  # (224,224,3)
    equalized_rgb = equalized_rgb.astype('float32') / 255.0
    return equalized_rgb

@app.route('/predict', methods=['POST'])
def predict():
    # Depuración detallada
    print("=== NUEVA SOLICITUD RECIBIDA ===")
    print(f"Método: {request.method}")
    print(f"Content-Type: {request.content_type}")
    print(f"Archivos recibidos: {list(request.files.keys())}")
    print(f"Formulario recibido: {list(request.form.keys())}")
    print(f"Datos JSON recibidos: {request.get_json(silent=True)}")
    
    # Si hay algún archivo, intentar procesarlo independientemente del nombre
    if request.files:
        print("Se encontraron archivos en la solicitud")
        # Tomar el primer archivo, independientemente del nombre
        file_key = list(request.files.keys())[0]
        file = request.files[file_key]
        print(f"Procesando archivo: {file.filename} con clave: {file_key}")
        
        try:
            # Guardar la imagen en un archivo temporal
            temp = tempfile.NamedTemporaryFile(delete=False)
            file.save(temp.name)
            temp.close()
            
            # Preprocesar la imagen
            img = preprocess_knee_image(temp.name)
            img = np.expand_dims(img, axis=0)  # (1, 224, 224, 3)
            
            # Hacer la predicción
            pred = model.predict(img)
            pred_class = int(pred[0][0] >= 0.5)
            pred_label = inv_class_indices[pred_class]
            
            # Eliminar el archivo temporal
            os.unlink(temp.name)

            # Depuración de la predicción
            print(f"Predicción: {pred_label}, Confianza: {pred[0][0]}")
            
            # Devolver el resultado
            return jsonify({
                'prediction': pred_label,
                'confidence': float(pred[0][0])
            })
        
        except Exception as e:
            return jsonify({
                'error': str(e), 
                'message': 'Error al procesar la imagen'
            }), 500
    
    # Si no hay archivos pero hay solicitud JSON, buscar imágenes en base64
    elif request.is_json:
        data = request.get_json()
        if 'image' in data and isinstance(data['image'], str):
            try:
                # Código para manejar imágenes en base64 si fuera necesario
                pass
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
    # Si llegamos aquí, no se pudo procesar ninguna imagen
    return jsonify({
        'error': 'No se envió ninguna imagen válida', 
        'files_received': list(request.files.keys()),
        'form_received': list(request.form.keys()),
        'content_type': request.content_type
    }), 400

# Agregar una ruta para la página principal con un formulario HTML
@app.route('/', methods=['GET'])
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Predicción de Imágenes de Rodilla</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .form-container { margin-top: 20px; }
            button { padding: 10px 20px; background-color: #4CAF50; color: white; border: none; cursor: pointer; }
        </style>
    </head>
    <body>
        <h1>Análisis de Imágenes de Rodilla</h1>
        <p>Sube una imagen para clasificarla como positiva o negativa.</p>
        
        <div class="form-container">
            <form action="/predict" method="post" enctype="multipart/form-data">
                <input type="file" name="image" accept="image/*" required>
                <button type="submit">Predecir</button>
            </form>
        </div>
    </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)