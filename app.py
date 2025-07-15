import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageEnhance
import cv2
import os
import logging
from tensorflow.keras import regularizers

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load models with error handling
try:
    # Load segmentation model
    seg_model = load_model('unet.h5')
    logger.info("Segmentation model loaded successfully!")
except Exception as e:
    logger.error(f"Error loading segmentation model: {e}")
    seg_model = None

try:
    # Load classification model
    cls_model_path = 'busi.h5'
    if not os.path.exists(cls_model_path):
        raise FileNotFoundError(f"Model file {cls_model_path} not found")
    
    cls_model = load_model(cls_model_path)
    logger.info("Classification model loaded successfully!")
    
    # Verify model architecture
    logger.info(f"Classification model input shape: {cls_model.input_shape}")
    logger.info(f"Classification model output shape: {cls_model.output_shape}")
    
except Exception as e:
    logger.error(f"Error loading classification model: {str(e)}")
    cls_model = None

# Define class labels (must match training order)
class_labels = ['benign', 'malignant', 'normal']

# Define custom enhancement functions
def fuzzy_enhancement(image):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(2.0)

def sharpness_enhancement(image):
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(1.0)

def brightness_enhancement(image):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(0.5)

def rescale_image(image, target_size=(500, 500)):
    return image.resize(target_size)

def preprocess_for_segmentation(image):
    """Preprocess image for segmentation"""
    try:
        # Convert to numpy array
        image = np.array(image)
        original_shape = image.shape[:2]
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Resize and normalize
        image = cv2.resize(image, (224, 224))
        image = image.astype('float32') / 255.0
        
        # Add batch and channel dimensions
        image = np.expand_dims(image, axis=(0, -1))
        
        return image, original_shape
    except Exception as e:
        logger.error(f"Segmentation preprocessing error: {e}")
        return None, None

def preprocess_for_classification(image):
    """Preprocess image for classification"""
    try:
        # Convert to numpy array
        image = np.array(image)
        
        # Convert to RGB if needed
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = image[:, :, :3]
        
        # Resize to model's expected input size
        image = Image.fromarray(image).resize((224, 224))
        image = np.array(image)
        
        # Normalize pixel values
        image = (image.astype('float32') / 255.0) - 0.5  # Zero-centering
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    except Exception as e:
        logger.error(f"Classification preprocessing error: {str(e)}")
        return None

def create_segmentation_overlay(original_image, mask):
    """Create enhanced overlay visualization"""
    try:
        # Convert to PIL Images
        original_pil = Image.fromarray(original_image)
        mask_pil = Image.fromarray((mask * 255).astype('uint8'))
        
        # Rescale both images
        rescaled_original = rescale_image(original_pil)
        rescaled_mask = rescale_image(mask_pil)
        
        # Create green mask overlay
        green_mask = Image.new("RGB", rescaled_mask.size, (0, 255, 0))
        green_mask.putalpha(rescaled_mask.convert("L"))
        
        # Combine images and apply enhancements
        overlayed = Image.alpha_composite(rescaled_original.convert("RGBA"), green_mask)
        overlayed = fuzzy_enhancement(overlayed)
        overlayed = sharpness_enhancement(overlayed)
        overlayed = brightness_enhancement(overlayed)
        
        return overlayed
    except Exception as e:
        logger.error(f"Overlay creation error: {e}")
        return None

def analyze_ultrasound(image):
    """Perform both segmentation and classification"""
    if seg_model is None or cls_model is None:
        return None, "One or both models failed to load", {}
    
    try:
        # Convert and store original image
        original_image = np.array(image)
        
        # Perform segmentation
        seg_processed, original_shape = preprocess_for_segmentation(original_image)
        if seg_processed is None:
            return None, "Segmentation preprocessing failed", {}
        
        mask = seg_model.predict(seg_processed, verbose=0)[0]
        mask = cv2.resize(mask.squeeze(), (original_shape[1], original_shape[0]))
        overlay = create_segmentation_overlay(original_image, mask)
        
        if overlay is None:
            return None, "Overlay creation failed", {}
        
        # Perform classification
        cls_processed = preprocess_for_classification(original_image)
        if cls_processed is None:
            return overlay, "Segmentation complete (classification failed)", {}
        
        predictions = cls_model.predict(cls_processed, verbose=0)[0]
        confidences = {
            'benign': float(predictions[0]),
            'malignant': float(predictions[1]),
            'normal': float(predictions[2])
        }
        
        return overlay, "Analysis complete", confidences
    
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return None, f"Error during analysis: {str(e)}", {}

# Create Gradio interface
title = "Breast Cancer Ultrasound Analysis"
description = """
Upload an ultrasound image for comprehensive analysis:
1. Segmentation of potential lesions (green overlay)
2. Classification (benign/malignant/normal with confidence scores)
"""

with gr.Blocks() as demo:
    gr.Markdown(f"## {title}")
    gr.Markdown(description)
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload Ultrasound Image", type="pil")
            analyze_btn = gr.Button("Analyze Image", variant="primary")
        
        with gr.Column():
            output_image = gr.Image(label="Segmentation Result")
            output_message = gr.Textbox(label="Status")
            output_label = gr.Label(num_top_classes=3, label="Classification Results")
    
    analyze_btn.click(
        fn=analyze_ultrasound,
        inputs=image_input,
        outputs=[output_image, output_message, output_label]
    )

if __name__ == "__main__":
    demo.launch()