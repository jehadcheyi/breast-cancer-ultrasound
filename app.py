import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import os

# Load your pre-trained model
model = load_model('CNN96.h5')

# Define class labels
class_labels = ['benign', 'malignant', 'normal']

def preprocess_image(image):
    """Preprocess the uploaded image to match model input requirements"""
    # Convert to numpy array
    image = np.array(image)
    
    # Convert to RGB if it's grayscale
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = image[:, :, :3]
    
    # Resize to 224x224
    image = cv2.resize(image, (224, 224))
    
    # Normalize pixel values to [0, 1]
    image = image / 255.0
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image

def predict_image(image):
    """Make prediction on the uploaded image"""
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Make prediction
    predictions = model.predict(processed_image)[0]
    
    # Create dictionary of class probabilities
    confidences = {class_labels[i]: float(predictions[i]) for i in range(len(class_labels))}
    
    return confidences

# Gradio interface
title = "Medical Image Classification"
description = """
Upload a medical image (224x224) to classify it as benign, malignant, or normal.
This app uses a pre-trained CNN model (CNN96.h5) for classification.
"""

examples = [
    ["example_benign.jpg"],
    ["example_malignant.jpg"],
    ["example_normal.jpg"]
]

# Create the Gradio interface
demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(label="Upload Medical Image", type="pil"),
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    title=title,
    description=description,
    examples=examples,
    allow_flagging="never"
)

# Launch the app
if __name__ == "__main__":
    demo.launch()