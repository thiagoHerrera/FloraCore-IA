import tensorflow as tf
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np
import sys

model_path = 'models/final/model.h5'
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    print(f"Error cargando el modelo: {e}")
    sys.exit(1)

def predict_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        prob = prediction[0][0]
        if prob > 0.5:
            print(f"{img_path} => Enferma ({prob*100:.2f}%)")
        else:
            print(f"{img_path} => Sana ({(1-prob)*100:.2f}%)")
    except Exception as e:
        print(f"Error procesando {img_path}: {e}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Uso: python src/predict.py <ruta_de_la_imagen>")
        sys.exit(1)
    img_path = sys.argv[1]
    predict_image(img_path)