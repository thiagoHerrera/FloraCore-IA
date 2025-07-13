import tensorflow as tf
print("Dispositivos físicos disponibles:", tf.config.list_physical_devices())
print("GPUs detectadas:", tf.config.list_physical_devices('GPU'))