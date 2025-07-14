import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageEnhance
import cv2
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load models with error handling
try:
    # Load classification model
    cls_model_path = 'cnn96.h5'
    if not os.path.exists(cls_model_path):
        raise FileNotFoundError(f"Classification model file {cls_model_path} not found")
    cls_model = load_model(cls_model_path)
    logger.info("Classification model loaded successfully!")
    
    # Load segmentation model
    seg_model_path = 'UNET_model.h5'
    if not os.path.exists(seg_model_path):
        raise FileNotFoundError(f"Segmentation model file {seg_model_path} not found")
    seg_model = load_model(seg_model_path)
    logger.info("Segmentation model loaded successfully!")
    
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    cls_model = None
    seg_model = None

# Define class labels
class_labels = ['benign', 'malignant', 'normal']

# Image enhancement functions from your code
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

def preprocess_for_classification(image):
    """Preprocess image for CNN96 classification"""
    try:
        # Convert to numpy array and resize
        image = np.array(image)
        image = cv2.resize(image, (224, 224))
        
        # Convert to RGB if needed
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = image[:, :, :3]
        
        # Normalize with zero-centering
        image = (image.astype('float32') / 255.0) - 0.5
        return np.expand_dims(image, axis=0)
    except Exception as e:
        logger.error(f"Classification preprocessing error: {str(e)}")
        return None

def preprocess_for_segmentation(image):
    """Preprocess image for UNet segmentation"""
    try:
        # Convert to numpy array and resize
        image = np.array(image)
        original_shape = image.shape[:2]
        image = cv2.resize(image, (256, 256))
        
        # Convert to grayscale and normalize
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = image.astype('float32') / 255.0
        
        return np.expand_dims(image, axis=(0, -1)), original_shape
    except Exception as e:
        logger.error(f"Segmentation preprocessing error: {str(e)}")
        return None

def create_segmentation_overlay(image, mask):
    """Create enhanced overlay from your pipeline"""
    try:
        # Convert to PIL Images
        image_pil = Image.fromarray(image)
        mask_pil = Image.fromarray((mask * 255).astype('uint8'))
        
        # Rescale
        rescaled_image = rescale_image(image_pil)
        rescaled_mask = rescale_image(mask_pil)
        
        # Create green mask overlay
        green_mask = Image.new("RGB", rescaled_mask.size, (0, 255, 0))
        green_mask.putalpha(rescaled_mask.convert("L"))
        
        # Overlay and enhance
        overlayed = Image.alpha_composite(rescaled_image.convert("RGBA"), green_mask)
        overlayed = fuzzy_enhancement(overlayed)
        overlayed = sharpness_enhancement(overlayed)
        overlayed = brightness_enhancement(overlayed)
        
        return np.array(overlayed)
    except Exception as e:
        logger.error(f"Overlay creation error: {str(e)}")
        return None

def analyze_image(image):
    """Full analysis pipeline: segmentation + classification"""
    if cls_model is None or seg_model is None:
        return {"Error": "Models failed to load"}, None
    
    try:
        # Convert input to numpy array
        original_image = np.array(image)
        
        # Classification
        cls_input = preprocess_for_classification(original_image)
        if cls_input is None:
            return {"Error": "Classification preprocessing failed"}, None
            
        predictions = cls_model.predict(cls_input, verbose=0)[0]
        cls_result = {
            'benign': float(predictions[0]),
            'malignant': float(predictions[1]),
            'normal': float(predictions[2])
        }
        
        # Segmentation
        seg_input, original_shape = preprocess_for_segmentation(original_image)
        if seg_input is None:
            return cls_result, None
            
        mask = seg_model.predict(seg_input, verbose=0)[0]
        mask = cv2.resize(mask.squeeze(), (original_shape[1], original_shape[0]))
        
        # Create overlay
        overlay = create_segmentation_overlay(original_image, mask)
        
        return cls_result, overlay
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return {"Error": f"Analysis failed: {str(e)}"}, None

# Gradio interface
title = "Breast Cancer Ultrasound Analysis"
description = """
Upload an ultrasound image for:
1. Automatic lesion segmentation (UNet model)
2. Tumor classification (CNN96 model)

Features:
- Enhanced segmentation visualization
- Confidence scores for each class
- Professional medical imaging pipeline
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
                cls_output = gr.Label(label="Diagnosis Confidence")
            with gr.Tab("Segmentation"):
                seg_output = gr.Image(label="Lesion Segmentation")
    
    # Example images
    gr.Examples(
        examples=[
            ["example_benign.png"],
            ["example_malignant.png"],
            ["example_normal.png"]
        ] if all(os.path.exists(f"example_{cls}.png") for cls in ['benign', 'malignant', 'normal']) else None,
        inputs=image_input,
        outputs=[cls_output, seg_output],
        fn=analyze_image,
        cache_examples=True
    )
    
    analyze_btn.click(
        fn=analyze_image,
        inputs=image_input,
        outputs=[cls_output, seg_output]
    )

if __name__ == "__main__":
    demo.launch()