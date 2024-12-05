import tensorflow as tf
import numpy as np
import os
import cv2
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

# Tamaño de las imágenes (debe coincidir con el tamaño utilizado en el entrenamiento)
IMG_SIZE = (150, 150)


# Función para cargar y preprocesar imágenes desde una carpeta
def load_images_from_folder(folder):
    images = []
    labels = []
    for label, subdir in enumerate(['benign', 'malignant']):
        subdir_path = os.path.join(folder, subdir)
        for filename in tqdm(os.listdir(subdir_path), desc=f"Cargando {subdir}"):
            img_path = os.path.join(subdir_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, IMG_SIZE)
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)


# Función para evaluar el modelo
def evaluate_model(model_path, test_folder):
    # Cargar el modelo guardado
    model = tf.keras.models.load_model(model_path)

    # Cargar imágenes de la carpeta de prueba
    X_test, y_test = load_images_from_folder(test_folder)
    X_test = X_test / 255.0  # Normalizar imágenes

    # Realizar predicciones
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int).flatten()  # Convertir a 0 o 1

    # Calcular y mostrar métricas
    print("\nResultados del modelo:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))


# Ruta del modelo guardado y carpeta de prueba
model_path = 'models/best_melanoma_model_manual.h5'
test_folder = 'melanoma_cancer_dataset/test'  # Cambiar por la carpeta deseada

# Evaluar el modelo
evaluate_model(model_path, test_folder)
