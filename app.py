import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model - assuming CNN96.h5 is in the same directory as app.py
try:
    model_path = 'cnn96.h5'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found in the current directory")
    
    logger.info(f"Loading model from: {model_path}")
    model = load_model(model_path)
    logger.info("Model loaded successfully!")
    
    # Verify model input/output shapes
    logger.info(f"Model input shape: {model.input_shape}")
    logger.info(f"Model output shape: {model.output_shape}")
    
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

# Define class labels
class_labels = ['benign', 'malignant', 'normal']

def preprocess_image(image):
    """Preprocess the uploaded image to match model input requirements"""
    try:
        # Convert to numpy array
        image = np.array(image)
        
        # Convert to RGB if needed
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = image[:, :, :3]
        
        # Resize to 224x224 (model's expected input size)
        image = cv2.resize(image, (224, 224))
        
        # Normalize pixel values to [0, 1]
        image = image / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return None

def predict_image(image):
    """Make prediction on the uploaded image"""
    if model is None:
        return {"Error": "Model failed to load. Please check if CNN96.h5 is in the same directory."}
    
    try:
        # Preprocess the image
        processed_image = preprocess_image(image)
        if processed_image is None:
            return {"Error": "Failed to process image"}
        
        # Verify the processed image shape matches model's expected input
        logger.info(f"Processed image shape: {processed_image.shape}")
        
        # Make prediction
        predictions = model.predict(processed_image)[0]
        logger.info(f"Raw predictions: {predictions}")
        
        # Create dictionary of class probabilities
        confidences = {class_labels[i]: float(predictions[i]) for i in range(len(class_labels))}
        
        return confidences
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return {"Error": f"An error occurred during prediction: {str(e)}"}

# Gradio interface
title = "Medical Image Classification"
description = """
Upload a medical image to classify it as benign, malignant, or normal.
This app uses a pre-trained CNN model (CNN96.h5) for classification.
"""

# Create the Gradio interface
demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(label="Upload Medical Image", type="pil"),
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    title=title,
    description=description,
    examples=[
        os.path.join(os.path.dirname(__file__), "example_benign.jpg"),
        os.path.join(os.path.dirname(__file__), "example_malignant.jpg"),
        os.path.join(os.path.dirname(__file__), "example_normal.jpg")
    ] if os.path.exists("example_benign.jpg") else None,
    allow_flagging="never"
)

# Launch the app
if __name__ == "__main__":
    demo.launch()