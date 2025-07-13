import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import sys
import os

# Configuración del umbral (ajustable según validación)
OPTIMAL_THRESHOLD = 0.5  # Reemplaza con el valor calculado en train_model.py si es diferente

# Ruta del modelo
model_path = 'C:/Users/thiag/Desktop/ia si/FloraCore-IA/models/final/model.h5'

# Cargar el modelo
try:
    model = tf.keras.models.load_model(model_path)
    print(f"✅ Modelo cargado desde {model_path}")
except Exception as e:
    print(f"❌ Error cargando el modelo: {e}")
    sys.exit(1)

def predict_image(img_path):
    """
    Realiza una predicción sobre una imagen y muestra los resultados con detalles.
    """
    try:
        # Cargar y preprocesar la imagen
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Realizar la predicción
        prediction = model.predict(img_array, verbose=0)[0][0]
        confidence = prediction if prediction > OPTIMAL_THRESHOLD else (1 - prediction)
        class_name = "ENFERMA" if prediction > OPTIMAL_THRESHOLD else "SANA"

        # Mostrar resultados detallados
        print(f"🧪 Resultado: {class_name} (Confianza: {confidence*100:.2f}%)")
        print(f"  - Predicción cruda: {prediction:.4f}")
        print(f"  - Umbral usado: {OPTIMAL_THRESHOLD:.4f}")

    except (IOError, ValueError) as e:
        print(f"❌ Error procesando la imagen {img_path}: Imagen corrupta o no válida ({e})")
    except Exception as e:
        print(f"❌ Error procesando la imagen {img_path}: {e}")

if __name__ == '__main__':
    # Verificar argumentos
    if len(sys.argv) != 2:
        print("Uso: python src/predict.py <ruta_de_la_imagen>")
        sys.exit(1)
    
    img_path = sys.argv[1]
    if not os.path.isfile(img_path):
        print(f"❌ La ruta proporcionada '{img_path}' no es un archivo válido.")
        sys.exit(1)
    
    # Ejecutar predicción
    predict_image(img_path)