import os
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model

# Define paths for dataset categories
base_path = r'C:\Users\lenovo\Desktop\my article\Dataset_BUSI_with_GT'
categories = ['benign', 'malignant', 'normal']

# Load the dataset function
def load_data(base_path, categories, img_size=(224, 224)):
    images = []  # List to store images
    masks = []   # List to store masks
    for category in categories:
        path = os.path.join(base_path, category)  # Construct the category path
        # Get all image files that do not contain '_mask' in their names
        image_files = [f for f in glob(os.path.join(path, '*.png')) if '_mask' not in f]
        for img_path in image_files:
            # Read image in grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            # Construct corresponding mask path
            mask_path = img_path.replace('.png', '_mask.png')
            # Read mask in grayscale
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            # Check if both image and mask are loaded successfully
            if img is not None and mask is not None:
                # Resize image and mask to the specified size
                img_resized = cv2.resize(img, img_size)
                mask_resized = cv2.resize(mask, img_size)
                images.append(img_resized)  # Append resized image to the list
                masks.append(mask_resized)    # Append resized mask to the list
            else:
                print(f"Warning: Image or mask not found for {img_path}")  # Log a warning if not found
    return np.array(images), np.array(masks)  # Return numpy arrays of images and masks

# Preprocess the data function
def preprocess_data(images, masks):
    # Normalize image and mask pixel values to the range [0, 1]
    images = np.array(images) / 255.0
    masks = np.array(masks) / 255.0
    images = np.expand_dims(images, axis=-1)  # Expand dimensions to include channel
    masks = np.expand_dims(masks, axis=-1)    # Expand dimensions to include channel
    return images, masks  # Return preprocessed images and masks

# Build the U-Net model function
def unet_model(input_size=(224, 224, 1)):
    inputs = Input(input_size)  # Input layer
    # Encoding path
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(512, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(512, (3, 3), activation='relu', padding='same')(c5)

    # Decoding path with skip connections
    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c4])  # Skip connection from encoding path
    c6 = Conv2D(256, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(256, (3, 3), activation='relu', padding='same')(c6)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c3])  # Skip connection
    c7 = Conv2D(128, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(128, (3, 3), activation='relu', padding='same')(c7)

    u8 = UpSampling2D((2, 2))(c7)
    u8 = concatenate([u8, c2])  # Skip connection
    c8 = Conv2D(64, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(64, (3, 3), activation='relu', padding='same')(c8)

    u9 = UpSampling2D((2, 2))(c8)
    u9 = concatenate([u9, c1])  # Skip connection
    c9 = Conv2D(32, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(32, (3, 3), activation='relu', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)  # Output layer

    model = Model(inputs=[inputs], outputs=[outputs])  # Create model
    return model

# Compile the U-Net model
model = unet_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load and preprocess the data
images, masks = load_data(base_path, categories)  # Load data
images_preprocessed, masks_preprocessed = preprocess_data(images, masks)  # Preprocess data

# Split the data into training and validation sets
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(images_preprocessed, masks_preprocessed, test_size=0.2, random_state=42)

# Train the model
history = model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_val, y_val))

# Save the model
# model.save('unet_model2.h5')  # Uncomment this line to save the model

print("Model training complete and saved as 'UNET_model.h5'")

# Display the images
import matplotlib.pyplot as plt

# Load a sample image, mask, and prediction from your dataset
image_index = 0  # Change this index to visualize different images
sample_image = images[image_index]  # Original image
sample_mask = masks[image_index]  # Ground truth mask
# Predict the mask using the trained model
predicted_mask = model.predict(np.expand_dims(images_preprocessed[image_index], axis=0)).squeeze()

# Create the figure and subplots for visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Original Image
axes[0, 0].imshow(sample_image, cmap='gray')
axes[0, 0].set_title("Original Image")
axes[0, 0].axis('off')

# Preprocessed Image
axes[0, 1].imshow(images_preprocessed[image_index].squeeze(), cmap='gray')
axes[0, 1].set_title("Preprocessed Image")
axes[0, 1].axis('off')

# Ground Truth Mask
axes[0, 2].imshow(sample_mask, cmap='gray')
axes[0, 2].set_title("Ground Truth Mask")
axes[0, 2].axis('off')

# Predicted Mask
axes[1, 0].imshow(predicted_mask, cmap='gray')
axes[1, 0].set_title("Predicted Mask")
axes[1, 0].axis('off')

# U-Net Architecture
unet_image = plt.imread('unet_model.png')  # Load U-Net architecture image
axes[1, 1].imshow(unet_image)
axes[1, 1].set_title("U-Net Architecture")
axes[1, 1].axis('off')

# Overlay Prediction on Original Image
overlayed_image = cv2.addWeighted(sample_image, 0.7, (predicted_mask * 255).astype(np.uint8), 0.3, 0)
axes[1, 2].imshow(overlayed_image, cmap='gray')
axes[1, 2].set_title("Overlayed Prediction")
axes[1, 2].axis('off')

# Adjust layout for better visualization
plt.tight_layout()
plt.show()
