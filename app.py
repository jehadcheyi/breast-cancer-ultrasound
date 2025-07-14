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
    # Load segmentation model (UNet)
    seg_model_path = 'UNET_model.h5'
    if not os.path.exists(seg_model_path):
        raise FileNotFoundError(f"Segmentation model file {seg_model_path} not found")
    seg_model = load_model(seg_model_path)
    logger.info("Segmentation model loaded successfully!")
    logger.info(f"Segmentation model input shape: {seg_model.input_shape}")
    
    # Load classification model (CNN96)
    cls_model_path = 'cnn96.h5'
    if not os.path.exists(cls_model_path):
        raise FileNotFoundError(f"Classification model file {cls_model_path} not found")
    cls_model = load_model(cls_model_path)
    logger.info("Classification model loaded successfully!")
    logger.info(f"Classification model input shape: {cls_model.input_shape}")
    
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    seg_model = None
    cls_model = None

# Define class labels
class_labels = ['benign', 'malignant', 'normal']

# Image enhancement functions
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
    """Preprocess image for UNet segmentation (224x224 grayscale)"""
    try:
        image = np.array(image)
        original_shape = image.shape[:2]
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Resize and normalize
        image = cv2.resize(image, (224, 224))
        image = image.astype('float32') / 255.0
        
        # Add dimensions
        image = np.expand_dims(image, axis=-1)  # Add channel
        image = np.expand_dims(image, axis=0)   # Add batch
        
        return image, original_shape
    except Exception as e:
        logger.error(f"Segmentation preprocessing error: {str(e)}")
        return None, None

def preprocess_for_classification(image):
    """Preprocess segmented image for CNN96 classification (400x400 RGB)"""
    try:
        image = np.array(image)
        
        # Convert to RGB if needed
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = image[:, :, :3]
        
        # Resize and normalize with zero-centering
        image = Image.fromarray(image).resize((400, 400))
        image = np.array(image)
        image = (image.astype('float32') / 255.0) - 0.5
        
        return np.expand_dims(image, axis=0)
    except Exception as e:
        logger.error(f"Classification preprocessing error: {str(e)}")
        return None

def create_segmentation_overlay(original_image, mask):
    """Create enhanced overlay visualization"""
    try:
        original_pil = Image.fromarray(original_image)
        mask_pil = Image.fromarray((mask * 255).astype('uint8'))
        
        rescaled_original = rescale_image(original_pil)
        rescaled_mask = rescale_image(mask_pil)
        
        green_mask = Image.new("RGB", rescaled_mask.size, (0, 255, 0))
        green_mask.putalpha(rescaled_mask.convert("L"))
        
        overlayed = Image.alpha_composite(rescaled_original.convert("RGBA"), green_mask)
        overlayed = fuzzy_enhancement(overlayed)
        overlayed = sharpness_enhancement(overlayed)
        overlayed = brightness_enhancement(overlayed)
        
        return np.array(overlayed), mask
    except Exception as e:
        logger.error(f"Overlay creation error: {str(e)}")
        return None, None

def analyze_image(image):
    """Full analysis pipeline: segmentation -> classification"""
    if seg_model is None or cls_model is None:
        return {"Error": "Models failed to load"}, None, None
    
    try:
        original_image = np.array(image)
        
        # Step 1: Segmentation
        seg_input, original_shape = preprocess_for_segmentation(original_image)
        if seg_input is None:
            return {"Error": "Segmentation preprocessing failed"}, None, None
            
        mask = seg_model.predict(seg_input, verbose=0)[0]
        mask = cv2.resize(mask.squeeze(), (original_shape[1], original_shape[0]))
        
        # Create overlay and extract segmented lesion
        overlay, mask = create_segmentation_overlay(original_image, mask)
        if overlay is None:
            return {"Error": "Overlay creation failed"}, None, None
        
        # Step 2: Classification on segmented lesion
        # Use the mask to extract the lesion area
        segmented_lesion = cv2.bitwise_and(original_image, original_image, mask=(mask > 0.5).astype(np.uint8))
        
        cls_input = preprocess_for_classification(segmented_lesion)
        if cls_input is None:
            return {"Error": "Classification preprocessing failed"}, overlay, None
            
        predictions = cls_model.predict(cls_input, verbose=0)[0]
        cls_result = {
            'benign': float(predictions[0]),
            'malignant': float(predictions[1]),
            'normal': float(predictions[2])
        }
        
        return cls_result, overlay, segmented_lesion
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return {"Error": f"Analysis failed: {str(e)}"}, None, None

# Gradio interface
title = "Breast Cancer Ultrasound Analysis"
description = """
1. First segments the ultrasound image to identify lesions
2. Then classifies the segmented lesion as benign, malignant, or normal

Models:
- Segmentation: U-Net (224x224 grayscale input)
- Classification: CNN96 (400x400 RGB input, trained on segmented lesions)
"""

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(description)
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload Ultrasound Image", type="pil")
            analyze_btn = gr.Button("Analyze", variant="primary")
        
        with gr.Column():
            with gr.Tab("Classification Results"):
                cls_output = gr.Label(label="Lesion Diagnosis")
            with gr.Tab("Segmentation"):
                seg_output = gr.Image(label="Lesion Segmentation")
            with gr.Tab("Segmented Lesion"):
                lesion_output = gr.Image(label="Isolated Lesion")
    
    analyze_btn.click(
        fn=analyze_image,
        inputs=image_input,
        outputs=[cls_output, seg_output, lesion_output]
    )

if __name__ == "__main__":
    demo.launch()