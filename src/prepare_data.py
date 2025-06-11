import os
import shutil
import random

# Carpetas de entrada y salida
RAW_DIR = "data/raw/plantvillage-dataset/PlantVillage"
PROCESSED_DIR = "data/processed"
TRAIN_DIR = os.path.join(PROCESSED_DIR, "train")
VAL_DIR = os.path.join(PROCESSED_DIR, "val")

# Creamos carpetas destino
for subset in ['train', 'val']:
    for category in ['sana', 'enferma']:
        os.makedirs(os.path.join(PROCESSED_DIR, subset, category), exist_ok=True)

# Identificamos clases
healthy_classes = ['Pepper_bell___healthy', 'Potato___healthy', 'Tomato_healthy']

for class_dir in os.listdir(RAW_DIR):
    class_path = os.path.join(RAW_DIR, class_dir)

    if not os.path.isdir(class_path):
        continue

    # Ignorar carpeta duplicada "PlantVillage"
    if class_dir == "PlantVillage":
        continue


    # Determinamos si es sana o enferma
    if class_dir in healthy_classes:
        label = 'sana'
    else:
        label = 'enferma'

    # Listamos imágenes
    images = os.listdir(class_path)
    random.shuffle(images)

    # Split 80% train, 20% val
    split_idx = int(len(images) * 0.8)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    # Copiamos imágenes a train
    for img in train_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(TRAIN_DIR, label, img)
        shutil.copyfile(src, dst)

    # Copiamos imágenes a val
    for img in val_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(VAL_DIR, label, img)
        shutil.copyfile(src, dst)

print("Dataset procesado correctamente ✅")
