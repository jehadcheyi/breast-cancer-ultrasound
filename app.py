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

# Load model
try:
    model_path = 'cnn96.h5'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found")
    
    logger.info(f"Loading model from: {model_path}")
    model = load_model(model_path)
    logger.info("Model loaded successfully!")
    
    # Verify model architecture
    logger.info(f"Model input shape: {model.input_shape}")
    logger.info(f"Model output shape: {model.output_shape}")
    
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

# Define class labels
class_labels = ['benign', 'malignant', 'normal']

def preprocess_image(image):
    """Preprocess image to exactly match training conditions"""
    try:
        # Convert to numpy array
        image = np.array(image)
        
        # Convert to RGB if needed
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = image[:, :, :3]
        
        # Resize to model's expected input size (224x224)
        image = cv2.resize(image, (400, 400))  # Changed from 400x400 to 224x224
        
        # Normalize pixel values (must match training normalization)
        image = image / 255.0  # Assuming model was trained with [0,1] normalization
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return None

def predict_image(image):
    """Make prediction on the uploaded image"""
    if model is None:
        return {"Error": "Model failed to load"}
    
    try:
        # Preprocess the image
        processed_image = preprocess_image(image)
        if processed_image is None:
            return {"Error": "Failed to process image"}
        
        # Verify input shape matches model expectations
        logger.info(f"Processed image shape: {processed_image.shape}")
        if processed_image.shape[1:] != model.input_shape[1:]:
            return {"Error": f"Input shape mismatch. Expected {model.input_shape[1:]}, got {processed_image.shape[1:]}"}
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)[0]
        logger.info(f"Raw predictions: {predictions}")
        
        # Apply softmax if needed (some models don't have softmax in last layer)
        if not np.isclose(np.sum(predictions), 1.0, atol=0.01):
            predictions = tf.nn.softmax(predictions).numpy()
            logger.info(f"After softmax: {predictions}")
        
        # Create dictionary of class probabilities
        confidences = {class_labels[i]: float(predictions[i]) for i in range(len(class_labels))}
        
        return confidences
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return {"Error": f"Prediction error: {str(e)}"}

# Gradio interface
title = "Breast Cancer Ultrasound Classification"
description = """
Upload an ultrasound image to classify as benign, malignant, or normal.
Note: Model trained on 224x224 images - other sizes will be resized.
"""

demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(label="Upload Ultrasound Image", type="pil"),
    outputs=gr.Label(num_top_classes=3, label="Prediction Results"),
    title=title,
    description=description,
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()