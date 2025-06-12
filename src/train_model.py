import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import os
from tensorflow.keras.callbacks import EarlyStopping, Callback # type: ignore

# Paths
base_dir = 'data/processed'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
model_save_path = 'models/final/model.h5'

# Parámetros
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# Callback para imprimir precisión por época
class PrintAccuracy(Callback):
    def on_epoch_end(self, epoch, logs):
        print(f"Época {epoch + 1}/{EPOCHS} - Precisión de validación: {logs['val_accuracy'] * 100:.2f}%")

# Preprocesamiento de datos con manejo de errores
def custom_image_generator(directory, datagen, target_size, batch_size, class_mode):
    generator = datagen.flow_from_directory(
        directory,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=True
    )
    while True:
        try:
            batch_x, batch_y = next(generator)
            yield batch_x, batch_y
        except Exception as e:
            print(f"Error al procesar una imagen en {directory}: {e}. Saltando batch problemático...")
            continue

# Generadores de datos
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = custom_image_generator(
    train_dir,
    train_datagen,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_generator = custom_image_generator(
    val_dir,
    val_datagen,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Modelo MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
print_accuracy = PrintAccuracy()

# Entrenamiento
history = model.fit(
    train_generator,
    steps_per_epoch=12517 // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=3121 // BATCH_SIZE,
    callbacks=[early_stopping, print_accuracy]
)

# Obtener y mostrar la precisión final
final_val_accuracy = history.history['val_accuracy'][-1]
print(f"Precisión final de validación: {final_val_accuracy * 100:.2f}%")

# Guardar el modelo
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
model.save(model_save_path)

print(f"Modelo guardado en {model_save_path}")