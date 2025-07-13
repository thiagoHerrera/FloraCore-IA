import os
import shutil
import random
RAW_DIR = "data/raw/plantvillage-dataset/PlantVillage"
PROCESSED_DIR = "data/processed"
TRAIN_DIR = os.path.join(PROCESSED_DIR, "train")
VAL_DIR = os.path.join(PROCESSED_DIR, "val")

# 💣 Eliminar anteriores
if os.path.exists(TRAIN_DIR):
    shutil.rmtree(TRAIN_DIR)
if os.path.exists(VAL_DIR):
    shutil.rmtree(VAL_DIR)

# 🏗️ Crear carpetas nuevas
for subset in ['train', 'val']:
    for category in ['sana', 'enferma']:
        os.makedirs(os.path.join(PROCESSED_DIR, subset, category), exist_ok=True)


# Clasificamos carpetas automáticamente
for class_dir in os.listdir(RAW_DIR):
    class_path = os.path.join(RAW_DIR, class_dir)
    if not os.path.isdir(class_path) or class_dir == "PlantVillage":
        continue

    label = 'sana' if 'healthy' in class_dir.lower() else 'enferma'
    images = [img for img in os.listdir(class_path) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images)

    split_idx = int(len(images) * 0.8)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    for img in train_images:
        shutil.copyfile(os.path.join(class_path, img), os.path.join(TRAIN_DIR, label, img))

    for img in val_images:
        shutil.copyfile(os.path.join(class_path, img), os.path.join(VAL_DIR, label, img))

print("✅ Dataset procesado correctamente.")
