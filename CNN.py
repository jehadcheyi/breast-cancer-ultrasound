# Mount Google Drive in Colab to access files
from google.colab import drive
drive.mount('/content/drive')

import os

# Set the base path for the dataset in Google Colab
base_path = '/content/drive/My Drive/BUSI_enh_over'

# Check if the directory exists and list files
if os.path.exists(base_path):
    files = os.listdir(base_path)
    print("Files in the directory:")
    print(files)
else:
    print("Directory not found!")

# Import necessary libraries
import tensorflow as tf
import numpy as np
import os
import cv2
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, Nadam
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load images from a specified directory and preprocess them
def load_images(image_directory, label_value, dataset, label):
    images = [img for img in os.listdir(image_directory)]
    for image_name in images:
        # Only process PNG images that are not masks
        if image_name.split('.')[1] == 'png' and '_mask' not in image_name:
            image = cv2.imread(os.path.join(image_directory, image_name))  # Read the image
            if image is not None:
                # Convert to PIL Image and resize
                image = Image.fromarray(image, 'RGB').resize((224, 224))  # Change image size here
                dataset.append(np.array(image))  # Add the image to the dataset
                label.append(label_value)  # Add the corresponding label

# Set image directories for training and validation sets
train_directory = '/content/drive/My Drive/BUSI_enh_over/train'
test_directory = '/content/drive/My Drive/BUSI_enh_over/val'

# Initialize lists to hold datasets and labels
train_dataset = []
train_label = []
test_dataset = []
test_label = []

# Load images for the training set
load_images(os.path.join(train_directory, 'benign'), 0, train_dataset, train_label)
load_images(os.path.join(train_directory, 'malignant'), 1, train_dataset, train_label)
load_images(os.path.join(train_directory, 'normal'), 2, train_dataset, train_label)

# Load images for the test set
load_images(os.path.join(test_directory, 'benign'), 0, test_dataset, test_label)
load_images(os.path.join(test_directory, 'malignant'), 1, test_dataset, test_label)
load_images(os.path.join(test_directory, 'normal'), 2, test_dataset, test_label)

# Convert the datasets and labels to numpy arrays for processing
X_train = np.array(train_dataset)
y_train = np.array(train_label)
X_test = np.array(test_dataset)
y_test = np.array(test_label)

# Normalize pixel values to [0, 1] and apply zero-centering approach
X_train = (X_train.astype('float32') / 255.0) - 0.5
X_test = (X_test.astype('float32') / 255.0) - 0.5

# Set up data augmentation settings to enhance training data
train_datagen = ImageDataGenerator(
    rotation_range=15,          # Randomly rotate images by up to 15 degrees
    width_shift_range=0.2,      # Randomly shift images horizontally
    height_shift_range=0.2,     # Randomly shift images vertically
)

# Create a data generator for the training set
train_generator = train_datagen.flow(X_train, y_train, batch_size=32)

# Define the CNN architecture using Sequential model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3), padding='same'),
    MaxPooling2D((2, 2), strides=(2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2), strides=(2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2), strides=(2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2), strides=(2, 2)),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2), strides=(2, 2)),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2), strides=(1, 1)),
    
    GlobalAveragePooling2D(),   # Global average pooling to reduce dimensions
    Dense(512, activation='relu'),  # Fully connected layer with 512 units
    Dropout(0.5),               # Dropout layer to reduce overfitting
    Dense(1024, activation='relu'),  # Fully connected layer with 1024 units
    Dropout(0.5),               # Dropout layer to reduce overfitting
    Dense(3, activation='softmax'),  # Output layer for 3 classes with softmax activation
])

# Display the model summary
model.summary()

# Compile the model with Adam optimizer and sparse categorical crossentropy loss function
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, epochs=100, validation_data=(X_test, y_test))

# Plot training history for accuracy and loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training accuracy')  # Plot training accuracy
plt.plot(history.history['val_accuracy'], label='Validation accuracy')  # Plot validation accuracy
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training loss')  # Plot training loss
plt.plot(history.history['val_loss'], label='Validation loss')  # Plot validation loss
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate model performance on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy}")

# Predict classes for the test data
y_pred_classes = np.argmax(model.predict(X_test), axis=1)

# Generate and print the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred_classes))

# Calculate and display confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
print("Confusion Matrix:")
print(cm)

# Plot the confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['benign', 'malignant', 'normal'],  # Label the x-axis
            yticklabels=['benign', 'malignant', 'normal'])  # Label the y-axis
plt.title('Confusion Matrix')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()
