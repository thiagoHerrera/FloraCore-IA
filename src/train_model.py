import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, Callback
import os
import numpy as np
from collections import Counter
from sklearn.utils import class_weight

# Limitar hilos para evitar problemas de concurrencia
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Paths
base_dir = 'C:/Users/thiag/Desktop/ia si/FloraCore-IA/data/processed'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
model_save_path = 'C:/Users/thiag/Desktop/ia si/FloraCore-IA/models/final/model.h5'

# Verificar que las carpetas existan
if not os.path.exists(train_dir):
    raise FileNotFoundError(f"La carpeta {train_dir} no existe. Verifica la ruta.")
if not os.path.exists(val_dir):
    raise FileNotFoundError(f"La carpeta {val_dir} no existe. Verifica la ruta.")

# Parámetros
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30

class PrintAccuracy(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Época {epoch + 1} - Val Acc: {logs['val_accuracy']*100:.2f}%")

# Generador personalizado con logging
class SafeImageDataGenerator(ImageDataGenerator):
    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = []
        batch_y = []
        for i in index_array:
            try:
                fname = self.filenames[i]
                img_path = os.path.join(self.directory, fname)
                img = load_img(img_path, target_size=self.target_size)
                x = img_to_array(img)
                x = self.standardize(x)
                batch_x.append(x)
                batch_y.append(self.classes[i])
            except Exception as e:
                print(f"Error procesando {img_path}: {e}")
                continue
        if not batch_x:
            print("Advertencia: Batch vacío, omitiendo...")
            return np.array([]), np.array([])
        return np.array(batch_x), np.array(batch_y)

# Generadores de datos
train_datagen = SafeImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

# Verificar balance de clases
print("Distribución de clases en el conjunto de entrenamiento:", Counter(train_generator.classes))
print("Distribución de clases en el conjunto de validación:", Counter(val_generator.classes))
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))

# Modelo base
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = True

# Conectar capas personalizadas
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
print_accuracy = PrintAccuracy()

# Entrenamiento
try:
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=[early_stopping, print_accuracy],
        class_weight=class_weights
    )
except Exception as e:
    print(f"Error durante el entrenamiento: {e}")
    raise

# Guardar modelo
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
model.save(model_save_path)
print(f"✅ Modelo guardado en {model_save_path}")

# Visualizar resultados
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()