import tensorflow as tf
import numpy as np
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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

# Construir el modelo con los hiperparámetros especificados
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(150, 150, 3)),
    tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dropout(rate=0.4),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilar el modelo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Crear el callback de EarlyStopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Callback personalizado para guardar resultados en carpetas específicas
class SaveResultsCallback(tf.keras.callbacks.Callback):
    def __init__(self, base_path='results', folder_name=None):
        super().__init__()
        self.base_path = base_path
        self.folder_name = folder_name

    def on_train_end(self, logs=None):
        # Si se proporciona un nombre, usarlo; de lo contrario, buscar un nombre único
        if self.folder_name:
            trial_folder = os.path.join(self.base_path, self.folder_name)
        else:
            trial_folder = get_unique_trial_folder(self.base_path)

        # Crear la carpeta de resultados
        os.makedirs(trial_folder, exist_ok=True)

        # Guardar resultados en CSV
        history_df = pd.DataFrame(self.model.history.history)
        csv_filename = os.path.join(trial_folder, "trial_results.csv")
        history_df.to_csv(csv_filename, index=False)

        # Graficar y guardar Accuracy
        plt.figure(figsize=(10, 5))
        plt.plot(history_df['accuracy'], label='Train Accuracy')
        plt.plot(history_df['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(os.path.join(trial_folder, "accuracy.png"))
        plt.close()

        # Graficar y guardar Loss
        plt.figure(figsize=(10, 5))
        plt.plot(history_df['loss'], label='Train Loss')
        plt.plot(history_df['val_loss'], label='Validation Loss')
        plt.title('Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(trial_folder, "loss.png"))
        plt.close()

        print(f"Results saved to {trial_folder}")

def get_unique_trial_folder(base_path):
    counter = 1
    while os.path.exists(f"{base_path}/trial_{counter}"):
        counter += 1
    return f"{base_path}/trial_{counter}"


# Entrenar el modelo
os.makedirs('results', exist_ok=True)
history = model.fit(
    X_train, y_train,
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, SaveResultsCallback('final_model')]
)

# Guardar el modelo
model.save('best_melanoma_model_manual.h5')
