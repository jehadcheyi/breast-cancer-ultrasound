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

# Load models with error handling
try:
    # Load classification model
    cls_model_path = 'cnn96.h5'
    if not os.path.exists(cls_model_path):
        raise FileNotFoundError(f"Classification model file {cls_model_path} not found")
    
    logger.info(f"Loading classification model from: {cls_model_path}")
    cls_model = load_model(cls_model_path)
    logger.info("Classification model loaded successfully!")
    
    # Load segmentation model
    seg_model_path = 'unet.h5'
    if not os.path.exists(seg_model_path):
        raise FileNotFoundError(f"Segmentation model file {seg_model_path} not found")
    
    logger.info(f"Loading segmentation model from: {seg_model_path}")
    seg_model = load_model(seg_model_path)
    logger.info("Segmentation model loaded successfully!")
    
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    cls_model = None
    seg_model = None

# Define class labels
class_labels = ['benign', 'malignant', 'normal']

def preprocess_image(image, model_type='classification'):
    """Preprocess image for either classification or segmentation"""
    try:
        # Convert to numpy array
        image = np.array(image)
        
        # Convert to RGB if needed
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = image[:, :, :3]
        
        # Resize based on model type
        if model_type == 'classification':
            image = Image.fromarray(image).resize((224, 224))
            image = np.array(image)
            # Normalize for classification model
            image = (image.astype('float32') / 255.0) - 0.5
        else:  # segmentation
            original_shape = image.shape[:2]
            image = Image.fromarray(image).resize((256, 256))  # Common UNet input size
            image = np.array(image)
            # Normalize for segmentation model
            image = image.astype('float32') / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image, original_shape if model_type == 'segmentation' else image
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return None

def apply_segmentation(image):
    """Apply segmentation to the image"""
    if seg_model is None:
        return None
    
    try:
        # Preprocess for segmentation
        processed_img, original_shape = preprocess_image(image, model_type='segmentation')
        
        # Predict segmentation mask
        mask = seg_model.predict(processed_img, verbose=0)[0]
        mask = (mask > 0.5).astype('uint8') * 255  # Threshold and scale to 0-255
        
        # Resize mask to original image size
        mask = cv2.resize(mask, (original_shape[1], original_shape[0]))
        
        # Convert single channel to RGB for visualization
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        
        # Create overlay (50% opacity)
        overlay = cv2.addWeighted(np.array(image), 0.7, mask_rgb, 0.3, 0)
        
        return overlay, mask
    except Exception as e:
        logger.error(f"Segmentation error: {str(e)}")
        return None

def predict_image(image):
    """Make prediction and segmentation on the uploaded image"""
    if cls_model is None or seg_model is None:
        return {"Error": "Models failed to load. Please check model files."}, None
    
    try:
        # Classification
        processed_img = preprocess_image(image, model_type='classification')
        predictions = cls_model.predict(processed_img, verbose=0)[0]
        confidences = {
            'benign': float(predictions[0]),
            'malignant': float(predictions[1]),
            'normal': float(predictions[2])
        }
        
        # Segmentation
        seg_result = apply_segmentation(image)
        
        return confidences, seg_result[0] if seg_result else None
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return {"Error": f"Prediction error: {str(e)}"}, None

# Gradio interface
title = "Breast Cancer Ultrasound Analysis"
description = """
Upload an ultrasound image for:
1. Classification (benign/malignant/normal)
2. Lesion segmentation

Models used:
- CNN96: Classification model (224x224 input)
- UNet: Segmentation model (256x256 input)
"""

with gr.Blocks() as demo:
    gr.Markdown(f"## {title}")
    gr.Markdown(description)
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload Ultrasound Image", type="pil")
            submit_btn = gr.Button("Analyze")
        
        with gr.Column():
            label_output = gr.Label(label="Classification Results")
            segmentation_output = gr.Image(label="Segmentation Result")
    
    # Example images
    gr.Examples(
        examples=[
            ["example_benign.png"],
            ["example_malignant.png"],
            ["example_normal.png"]
        ] if all(os.path.exists(f"example_{cls}.png") for cls in ['benign', 'malignant', 'normal']) else None,
        inputs=image_input
    )
    
    submit_btn.click(
        fn=predict_image,
        inputs=image_input,
        outputs=[label_output, segmentation_output]
    )

if __name__ == "__main__":
    demo.launch()