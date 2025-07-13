from PIL import Image
import os
import shutil
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore

def check_images(directory):
    corrupted_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                # Simular el procesamiento del ImageDataGenerator
                img = load_img(file_path, target_size=(224, 224))  # Mismo tamaño que en train_model.py
                img_array = img_to_array(img)  # Convertir a array como lo hace el generador
                print(f"✅ Imagen válida: {file_path}")
            except Exception as e:
                print(f"❌ Error en la imagen {file_path}: {e}")
                corrupted_files.append(file_path)
    return corrupted_files

# Rutas absolutas
train_dir = 'C:/Users/thiag/Desktop/ia si/FloraCore-IA/data/processed/train'
val_dir = 'C:/Users/thiag/Desktop/ia si/FloraCore-IA/data/processed/val'
print("Verificando imágenes en el conjunto de entrenamiento...")
corrupted_train = check_images(train_dir)
print("\nVerificando imágenes en el conjunto de validación...")
corrupted_val = check_images(val_dir)

if corrupted_train or corrupted_val:
    print("\nArchivos problemáticos encontrados:")
    corrupted_dir = 'C:/Users/thiag/Desktop/ia si/FloraCore-IA/data/processed/corrupted'
    os.makedirs(corrupted_dir, exist_ok=True)
    for file in corrupted_train + corrupted_val:
        print(file)
        shutil.move(file, os.path.join(corrupted_dir, os.path.basename(file)))
        print(f"Movido: {file} -> {corrupted_dir}")
else:
    print("\n✅ No se encontraron archivos problemáticos.")