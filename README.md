# API para Detección en Imágenes de Rodilla

Esta API permite analizar imágenes de rodilla y clasificarlas como positivas o negativas utilizando un modelo de deep learning basado en MobileNetV2.

## Requisitos Previos

- Python 3.9 o superior
- Pip (gestor de paquetes de Python)
- Windows, macOS o Linux

## Paso a Paso para Configurar y Usar la API

### 1. Crear Entorno Virtual

Primero, crea un entorno virtual para aislar las dependencias del proyecto:

```powershell
# Navega a la carpeta del proyecto
cd ruta\a\knee_api

# Crea un entorno virtual
python -m venv venv

# Activa el entorno virtual
.\venv\Scripts\Activate
```

### 2. Instalar Dependencias

Una vez activado el entorno virtual, instala las dependencias necesarias:

```powershell
# Instala las dependencias desde el archivo requeriments.txt
pip install -r requeriments.txt
```

> **Nota**: La instalación puede tardar varios minutos debido a librerías como TensorFlow y OpenCV.

### 3. Ejecutar la API

Después de instalar las dependencias, puedes ejecutar la API:

```powershell
# Ejecuta la aplicación
python app.py
```

Deberías ver una salida similar a:
```
 * Serving Flask app '...'
 * Debug mode: on
 * Running on http://0.0.0.0:5000/ (Press CTRL+C to quit)
```

### 4. Acceder a la Interfaz Web

Abre tu navegador web y navega a:

```
http://localhost:5000/
```

Verás una interfaz simple con:
- Un título: "Análisis de Imágenes de Rodilla"
- Un botón para subir imágenes
- Un botón "Predecir" para enviar la imagen

### 5. Usar la API

1. Haz clic en el botón para seleccionar una imagen de rodilla
2. Selecciona la imagen que deseas analizar
3. Haz clic en "Predecir"
4. Espera a que el sistema procese la imagen
5. Se mostrará el resultado de la predicción (positivo o negativo) junto con el nivel de confianza

### 6. API REST

También puedes usar la API directamente mediante peticiones HTTP:

```
POST /predict
```

Parámetros:
- `image`: Archivo de imagen para analizar

Ejemplo de respuesta:
```json
{
  "prediction": "positive",
  "confidence": 0.89
}
```

## Estructura del Proyecto

- `app.py`: Código principal de la API Flask
- `model_mobilenetv2_binary_final.h5`: Modelo entrenado para clasificación
- `requeriments.txt`: Lista de dependencias
- `venv_new/`: Entorno virtual con dependencias instaladas (si existe)

## Notas Técnicas

- La aplicación ejecuta un servidor web en el puerto 5000
- El modelo espera imágenes de rodilla que serán preprocesadas antes de la clasificación
- Las imágenes son redimensionadas y normalizadas automáticamente

## Solución de Problemas

Si encuentras problemas con la instalación de dependencias:
1. Asegúrate de tener la versión correcta de Python
2. En algunos sistemas, puede ser necesario instalar algunas dependencias manualmente:
   ```
   pip install tensorflow opencv-python flask
   ```
3. Si hay problemas con TensorFlow, considera usar una versión específica compatible con tu sistema
