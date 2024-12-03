import tensorflow as tf
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
import keras_tuner as kt

# Configuración de rutas
train_dir = 'melanoma_cancer_dataset/train'
test_dir = 'melanoma_cancer_dataset/test'

# Tamaño de las imágenes
IMG_SIZE = (150, 150)


# Función para cargar imágenes y etiquetas
def load_images_from_directory(directory):
    images = []
    labels = []
    for label, subdir in enumerate(['benign', 'malignant']):
        subdir_path = os.path.join(directory, subdir)
        for filename in os.listdir(subdir_path):
            img_path = os.path.join(subdir_path, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, IMG_SIZE)
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)


# Cargar datos y normalizar
X, y = load_images_from_directory(train_dir)
X = X / 255.0

# Dividir en train y validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Definir el modelo con hiperparámetros
def build_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(
        filters=hp.Int('filters', min_value=32, max_value=128, step=32),
        kernel_size=hp.Choice('kernel_size', values=[3, 5]),
        activation='relu',
        input_shape=(150, 150, 3)
    ))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(
        units=hp.Int('units', min_value=64, max_value=256, step=64),
        activation='relu'
    ))
    model.add(tf.keras.layers.Dropout(hp.Float('dropout', 0.2, 0.5, step=0.1)))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        ),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# Crear un tuner
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=1,
    directory='my_dir',
    project_name='melanoma_tuning'
)

# Realizar la búsqueda de hiperparámetros
tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Obtener el mejor modelo
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
Mejor número de filtros: {best_hps.get('filters')}
Mejor tamaño del kernel: {best_hps.get('kernel_size')}
Mejor número de unidades: {best_hps.get('units')}
Mejor tasa de aprendizaje: {best_hps.get('learning_rate')}
Mejor dropout: {best_hps.get('dropout')}
""")

# Entrenar el mejor modelo
best_model = tuner.hypermodel.build(best_hps)
history = best_model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val))

# Guardar el modelo
best_model.save('best_melanoma_model.h5')
