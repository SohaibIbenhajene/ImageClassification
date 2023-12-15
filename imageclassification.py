import streamlit as st
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from tensorflow.keras.utils import image_dataset_from_directory
from sklearn.metrics import confusion_matrix, classification_report

# Define constants
NUM_CLASSES = 5
IMG_SIZE = 64
HEIGTH_FACTOR = 0.2
WIDTH_FACTOR = 0.2

# Function to create the model
def create_model():
    model = tf.keras.Sequential([
        layers.Resizing(IMG_SIZE, IMG_SIZE),
        layers.Rescaling(1./255),
        layers.RandomFlip("horizontal"),
        layers.RandomTranslation(HEIGTH_FACTOR, WIDTH_FACTOR),
        layers.RandomZoom(0.2),
        layers.Conv2D(64, (3, 3), input_shape=(IMG_SIZE, IMG_SIZE, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(NUM_CLASSES, activation="softmax")
    ])

    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to get predictions from a dataset
def get_predictions(model, dataset):
    predictions = []
    true_labels = []

    for images, labels in dataset:
        # Make predictions on your model using your test set
        predictions_batch = model.predict(images, verbose=0)
        # Extend 'predictions' with the predicted class indices for each image in the batch
        predictions.extend(np.argmax(predictions_batch, axis=1))
        # Extend 'true_labels' with the true class indices for each image in the batch
        true_labels.extend(np.argmax(labels, axis=1))

    return np.array(predictions), np.array(true_labels)

# Streamlit App
def main():
    st.title("Image Classification with Streamlit")

    # Set the parameters for your data
    batch_size = 16
    image_size = (64, 64)
    validation_split = 0.2

    # Create the training dataset from the 'train' directory
    training_set = image_dataset_from_directory(
        directory='datasets/training_set',
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=image_size,
        validation_split=validation_split,
        subset='training',
        seed=123
    )

    # Create the validation dataset from the 'train' directory
    validation_set = image_dataset_from_directory(
        directory='datasets/training_set',
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=image_size,
        validation_split=validation_split,
        subset='validation',
        seed=123
    )

    # Create the testing dataset from the 'test' directory
    test_set = image_dataset_from_directory(
        directory='datasets/test_set',
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=image_size
    )

    # Sidebar
    st.sidebar.header("Model Configuration")
    epochs = st.sidebar.slider("Select the number of epochs", 1, 20, 10)
    train_button = st.sidebar.button("Train Model")

    if train_button:
        # Create and train the model
        model = create_model()
        history = model.fit(training_set, validation_data=validation_set, steps_per_epoch=10, epochs=epochs)

        # Display training history
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.plot(history.history['loss'], label='training loss')
        ax1.plot(history.history['val_loss'], label='validation loss')
        ax1.set_title('Loss curves')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        ax2.plot(history.history['accuracy'], label='training accuracy')
        ax2.plot(history.history['val_accuracy'], label='validation accuracy')
        ax2.set_title('Accuracy curves')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

        fig.tight_layout()
        st.pyplot(fig)

        # Evaluate the model on the test set
        test_loss, test_acc = model.evaluate(test_set)
        st.write(f'Test accuracy (model): {test_acc:.2%}')

        # Get predictions on the test set
        test_predictions, test_true_labels = get_predictions(model, test_set)

        # Create confusion matrix
        conf_matrix = confusion_matrix(test_true_labels, test_predictions, labels=np.arange(NUM_CLASSES))
        st.write("Confusion Matrix:")
        st.write(conf_matrix)

        # Print classification report with labels
        class_labels = ['Cat', 'Dog', 'Elephant', 'Giraffe', 'Rabbit']
        class_report = classification_report(test_true_labels, test_predictions, target_names=class_labels)
        st.write("\nClassification Report:")
        st.write(class_report)

if __name__ == "__main__":
    main()