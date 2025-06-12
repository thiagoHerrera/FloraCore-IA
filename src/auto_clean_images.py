import os
from PIL import Image
import shutil

def auto_clean_images(directory, backup_dir):
    # Crea una carpeta de respaldo si no existe
    os.makedirs(backup_dir, exist_ok=True)
    
    removed_count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                # Calcula la ruta de respaldo relativa
                rel_path = os.path.relpath(file_path, directory)
                backup_path = os.path.join(backup_dir, rel_path)
                os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                
                try:
                    with Image.open(file_path) as img:
                        img.verify()  # Verificación básica
                        img.load()    # Verificación más exhaustiva
                        print(f"Imagen válida: {file_path}")
                except Exception as e:
                    print(f"Imagen corrupta detectada: {file_path} - Error: {e}")
                    # Mueve la imagen corrupta a la carpeta de respaldo
                    shutil.move(file_path, backup_path)
                    removed_count += 1
                    print(f"Imagen movida a respaldo: {backup_path}")
    
    print(f"Total de imágenes movidas a respaldo: {removed_count}")

# Directorios a limpiar
train_dir = r'C:\Users\thiag\Downloads\FloraCore-IA\FloraCore-IA\data\processed\train'
val_dir = r'C:\Users\thiag\Downloads\FloraCore-IA\FloraCore-IA\data\processed\val'
backup_base = r'C:\Users\thiag\Downloads\FloraCore-IA\FloraCore-IA\data\processed\backup'

auto_clean_images(train_dir, os.path.join(backup_base, 'train'))
auto_clean_images(val_dir, os.path.join(backup_base, 'val'))