#check_images.py ANALIZA TODAS LAS IMAGENES Y VERIFICA CUAL ES VALIDA Y CUAL NO, Y SI ES INVALIDA
#LA ELIMINA.
import os
from PIL import Image

def check_and_clean_images(directory):
    removed_count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        img.verify()  # Verificación básica
                        img.load()   # Carga completa de la imagen
                        print(f"Imagen válida: {file_path}")
                except Exception as e:
                    print(f"Imagen corrupta o inválida: {file_path} - Error: {e}")
                    os.remove(file_path)
                    removed_count += 1
                    print(f"Imagen eliminada: {file_path}")
    print(f"Total de imágenes eliminadas: {removed_count}")

# Directorios a verificar y limpiar
check_and_clean_images(r'C:\Users\thiag\Downloads\FloraCore-IA\FloraCore-IA\data\processed\train')
check_and_clean_images(r'C:\Users\thiag\Downloads\FloraCore-IA\FloraCore-IA\data\processed\val')