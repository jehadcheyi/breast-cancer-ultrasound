import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import os
import logging
from tensorflow.keras import regularizers

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model
try:
    model_path = 'busi.h5'
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

# Define class labels (must match training order)
class_labels = ['benign', 'malignant', 'normal']

def preprocess_image(image):
    """Preprocess image to EXACTLY match training conditions"""
    try:
        # Convert to numpy array
        image = np.array(image)
        
        # Convert to RGB if needed (matches your training data loading)
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = image[:, :, :3]
        
        # Resize to model's expected input size (224x224)
        image = Image.fromarray(image).resize((224, 224))
        image = np.array(image)
        
        # Normalize pixel values EXACTLY like during training
        image = (image.astype('float32') / 255.0) - 0.5  # Zero-centering
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return None

def predict_image(image):
    """Make prediction on the uploaded image"""
    if model is None:
        return {"Error": "Model failed to load. Please check if cnn96.h5 exists."}
    
    try:
        # Preprocess the image
        processed_image = preprocess_image(image)
        if processed_image is None:
            return {"Error": "Failed to process image"}
        
        # Verify input shape
        if processed_image.shape[1:] != model.input_shape[1:]:
            return {"Error": f"Input shape mismatch. Expected {model.input_shape[1:]}, got {processed_image.shape[1:]}"}
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)[0]
        logger.info(f"Raw predictions: {predictions}")
        
        # Create dictionary of class probabilities
        confidences = {
            'benign': float(predictions[0]),
            'malignant': float(predictions[1]),
            'normal': float(predictions[2])
        }
        
        return confidences
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return {"Error": f"Prediction error: {str(e)}"}

# Gradio interface
title = "Breast Cancer Ultrasound Classification"
description = """
Upload an ultrasound image (224x224) to classify as:
- Benign (0)
- Malignant (1) 
- Normal (2)

Model architecture:
- 7 Conv layers with MaxPooling
- 1024-unit Dense layer with L2 regularization
- 50% Dropout
- Trained with Adam (lr=0.0001)
"""

demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(label="Upload Ultrasound Image", type="pil"),
    outputs=gr.Label(num_top_classes=3, label="Prediction Results"),
    title=title,
    description=description,
    examples=[
        ["example_benign.png"],
        ["example_malignant.png"],
        ["example_normal.png"]
    ] if all(os.path.exists(f"example_{cls}.png") for cls in ['benign', 'malignant', 'normal']) else None,
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()