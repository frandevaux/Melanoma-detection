import tensorflow as tf
import cv2
import numpy as np

# Tamaño de las imágenes (debe coincidir con el tamaño utilizado en el entrenamiento)
IMG_SIZE = (150, 150)


# Función para cargar y preprocesar una imagen
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen en la ruta: {image_path}")
    img = cv2.resize(img, IMG_SIZE)  # Redimensionar
    img = img / 255.0  # Normalizar
    img = np.expand_dims(img, axis=0)  # Expandir dimensiones para el modelo
    return img


# Función para predecir el resultado
def predict_image(model_path, image_path):
    # Cargar el modelo guardado
    model = tf.keras.models.load_model(model_path)

    # Preprocesar la imagen
    img = preprocess_image(image_path)

    # Realizar la predicción
    prediction = model.predict(img)[0][0]

    # Interpretar el resultado
    if prediction > 0.5:
        result = "Malignant (Maligno)"
    else:
        result = "Benign (Benigno)"

    print(f"Predicción: {result} (Confianza: {prediction:.2f})")


# Ruta del modelo y de la imagen
model_path = 'models/best_melanoma_model.h5'
image_path = 'melanoma_cancer_dataset/random/benign.jpg'  # Cambiar por la ruta de la imagen a evaluar

# Realizar la predicción
predict_image(model_path, image_path)
