# Melanoma Detection with Convolutional Neural Networks

A project by Francisco Devaux for the Artificial Intelligence II course, Bachelor's Degree in Computer Science at Universidad Nacional de Cuyo.

-----

## 📝 Problem Description

Skin cancer is one of the most common types of cancer worldwide. Melanoma, a severe form of this disease, can be fatal if not detected early. However, early detection significantly increases the chances of successful treatment and recovery.

The objective of this project is to build a deep learning model that classifies images of moles as either **benign** or **malignant**. Using a labeled image dataset, a model is trained to differentiate between these two classes based on automatically extracted visual features.

The dataset was obtained from Kaggle and can be accessed via the following link: [Melanoma Skin Cancer Dataset of 10000 Images](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images).

This solution has the potential to serve as a support tool for dermatologists, helping them to prioritize suspicious cases and improve diagnostic accuracy.

-----

## 📊 1. Data Preparation and Visualization

The first step involved loading and preprocessing the image data. The distribution of benign and malignant classes in both the training and test sets was analyzed. As seen below, the classes are well-balanced, which is ideal for training an unbiased model.

<img width="790" height="590" alt="imagen" src="https://github.com/user-attachments/assets/ee0eb617-9257-42d5-82a9-2bf57d916e9b" />

Here are some examples of benign and malignant moles from the dataset:

<img width="1171" height="593" alt="imagen" src="https://github.com/user-attachments/assets/4d116d54-236f-426d-8784-73d05c8a2ff4" />

-----

## 🤖 2. Model Building

To find an optimal architecture for the Convolutional Neural Network (CNN), a hyperparameter search was conducted using the `keras-tuner` library. A function was defined to build the CNN model, varying the following key parameters over 20 trials:

  * **Convolutional Layer Filters**: Number of filters (e.g., 32, 64, 128).
  * **Kernel Size**: The size of the convolutional kernel (3x3 or 5x5).
  * **Dense Layer Units**: Number of neurons in the dense layer (e.g., 64, 128, 256).
  * **Dropout Rate**: The rate for the dropout regularization (from 0.2 to 0.5).
  * **Learning Rate**: The learning rate for the Adam optimizer (e.g., 1e-2, 1e-3, 1e-4).

To save computation time and prevent overfitting, an `EarlyStopping` callback was used to halt the training process if no improvement in the validation loss was observed after 12 epochs.

The best hyperparameters found were:

  * **Filters**: 128
  * **Kernel Size**: 3
  * **Units**: 128
  * **Learning Rate**: 0.0001
  * **Dropout**: 0.3

<img width="790" height="405" alt="imagen" src="https://github.com/user-attachments/assets/0d1cd033-3425-4e86-b101-1e28993603c5" />


-----

## 📈 3. Model Evaluation

The best-performing model was then evaluated on the unseen test dataset. The predictions were used to calculate key performance metrics.

The model's classification report is as follows:

```
              precision    recall  f1-score   support

      Benign       0.89      0.93      0.91       500
   Malignant       0.93      0.89      0.91       500

    accuracy                           0.91      1000
   macro avg       0.91      0.91      0.91      1000
weighted avg       0.91      0.91      0.91      1000
```

The confusion matrix below shows that the model correctly classified the majority of both benign and malignant images.

<img width="642" height="545" alt="imagen" src="https://github.com/user-attachments/assets/ce2558ff-c6be-4a9f-b47d-ab1942e31c35" />


The ROC curve demonstrates the model's excellent performance in binary classification, with an Area Under the Curve (AUC) of **0.98**.

<img width="688" height="545" alt="imagen" src="https://github.com/user-attachments/assets/e6226a89-3023-4dd9-81a0-3aed315a29fc" />


-----

## 🏁 4. Conclusion

This project successfully developed a deep learning model based on a Convolutional Neural Network to classify mole images as benign or malignant. By using `keras-tuner` for hyperparameter optimization, an effective architecture was identified, trained, and evaluated.

The final model achieved an **accuracy of 0.91** and a high **AUC-ROC score of 0.98**, with strong precision, recall, and f1-scores for both classes. The confusion matrix confirms its ability to correctly classify the vast majority of images.

These results suggest that the model could serve as a valuable tool to support dermatologists in the early detection of melanoma. However, it is crucial to remember that this model is intended as a support tool and does not replace the clinical judgment of a qualified healthcare professional.
