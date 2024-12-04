import tensorflow as tf
import numpy as np
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from keras import Input
from sklearn.model_selection import train_test_split
import keras_tuner as kt

print("GPUs disponibles: ", tf.config.list_physical_devices('GPU'))

# Configuración de rutas
train_dir = 'melanoma_cancer_dataset/train'
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
    model = tf.keras.Sequential([
        Input(shape=(150, 150, 3)),
        tf.keras.layers.Conv2D(
            filters=hp.Int('filters', min_value=32, max_value=128, step=32),
            kernel_size=hp.Choice('kernel_size', values=[3, 5]),
            activation='relu'
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(
            units=hp.Int('units', min_value=64, max_value=256, step=64),
            activation='relu'
        ),
        tf.keras.layers.Dropout(hp.Float('dropout', 0.2, 0.5, step=0.1)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

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
    max_trials=20,
    executions_per_trial=1,
    directory='my_dir2',
    project_name='melanoma_tuning2'
)

# Crear el callback de EarlyStopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Callback personalizado para guardar resultados
class SaveResultsCallback(tf.keras.callbacks.Callback):
    def __init__(self, trial_name):
        self.trial_name = trial_name

    def on_train_end(self, logs=None):
        # Guardar los resultados en CSV
        history_df = pd.DataFrame(self.model.history.history)
        history_df.to_csv(f'results/{self.trial_name}_results.csv', index=False)

        # Graficar resultados
        plt.figure(figsize=(10, 5))
        plt.plot(history_df['accuracy'], label='Train Accuracy')
        plt.plot(history_df['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(f'results/{self.trial_name}_accuracy.png')
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(history_df['loss'], label='Train Loss')
        plt.plot(history_df['val_loss'], label='Validation Loss')
        plt.title('Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'results/{self.trial_name}_loss.png')
        plt.close()

# Crear directorio para guardar resultados
os.makedirs('results', exist_ok=True)

# Realizar la búsqueda de hiperparámetros
tuner.search(
    X_train, y_train,
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, SaveResultsCallback('trial')]
)

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
history = best_model.fit(
    X_train, y_train,
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, SaveResultsCallback('final_model')]
)

# Guardar el mejor modelo
best_model.save('best_melanoma_model2.h5')
