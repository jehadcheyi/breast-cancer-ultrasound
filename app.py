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
    seg_model_path = 'unet.h5'
    if not os.path.exists(seg_model_path):
        raise FileNotFoundError(f"Segmentation model file {seg_model_path} not found")
    seg_model = load_model(seg_model_path)
    logger.info("Segmentation model loaded successfully!")

    cls_model_path = 'cnn96.h5'
    if not os.path.exists(cls_model_path):
        raise FileNotFoundError(f"Classification model file {cls_model_path} not found")
    cls_model = load_model(cls_model_path)
    logger.info("Classification model loaded successfully!")

except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    seg_model = None
    cls_model = None

# Labels
class_labels = ['benign', 'malignant', 'normal']

# Enhancement functions (matching your segmentation script)
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
    try:
        image = np.array(image)
        original_shape = image.shape[:2]

        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        image = cv2.resize(image, (224, 224))
        image = image.astype('float32') / 255.0

        image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, axis=0)

        return image, original_shape
    except Exception as e:
        logger.error(f"Segmentation preprocessing error: {str(e)}")
        return None, None

def preprocess_for_classification(image):
    try:
        image = np.array(image)

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = image[:, :, :3]

        image = Image.fromarray(image).resize((400, 400))
        image = np.array(image)
        image = (image.astype('float32') / 255.0) - 0.5

        return np.expand_dims(image, axis=0)
    except Exception as e:
        logger.error(f"Classification preprocessing error: {str(e)}")
        return None

def create_segmentation_overlay(original_image, mask):
    try:
        # Convert to numpy array if not already
        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image)
        
        # Convert to grayscale if needed
        if len(original_image.shape) == 3:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        
        # Resize mask to match original image
        mask = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]))
        
        # Threshold and apply morphological operations (matching your segmentation script)
        _, binary_mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask.astype('uint8'), cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        mask = binary_mask
        
        # Convert grayscale to RGB for overlay
        overlayed_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        
        # Apply jet colormap to mask (matching your segmentation script)
        overlayed_mask = (mask * 255).astype(np.uint8)
        overlayed_mask_colormap = cv2.applyColorMap(overlayed_mask, cv2.COLORMAP_JET)
        
        # Blend the images (70% original, 30% mask)
        overlayed_image = cv2.addWeighted(overlayed_image, 0.7, overlayed_mask_colormap, 0.3, 0)
        
        # Convert to PIL Image for enhancement
        overlayed_image = Image.fromarray(overlayed_image)
        
        # Apply enhancements (matching your segmentation script)
        overlayed_image = fuzzy_enhancement(overlayed_image)
        overlayed_image = sharpness_enhancement(overlayed_image)
        overlayed_image = brightness_enhancement(overlayed_image)
        
        # Rescale to 500x500 (matching your segmentation script)
        overlayed_image = rescale_image(overlayed_image, (500, 500))
        
        return np.array(overlayed_image)
    except Exception as e:
        logger.error(f"Overlay creation error: {str(e)}")
        return None

def analyze_image(image):
    if seg_model is None or cls_model is None:
        return {"Error": "Models failed to load"}, None, None

    try:
        original_image = np.array(image)

        seg_input, original_shape = preprocess_for_segmentation(original_image)
        if seg_input is None:
            return {"Error": "Segmentation preprocessing failed"}, None, None

        mask = seg_model.predict(seg_input, verbose=0)[0]
        mask = mask.squeeze()  # Remove single-dimensional entries

        # Create overlay using the matching segmentation approach
        overlay = create_segmentation_overlay(original_image, mask)
        if overlay is None:
            return {"Error": "Overlay creation failed"}, None, None

        # Classification
        cls_input = preprocess_for_classification(overlay)
        if cls_input is None:
            return {"Error": "Classification preprocessing failed"}, overlay, None

        predictions = cls_model.predict(cls_input, verbose=0)[0]
        cls_result = {
            'benign': float(predictions[0]),
            'malignant': float(predictions[1]),
            'normal': float(predictions[2])
        }

        return cls_result, overlay, None

    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return {"Error": f"Analysis failed: {str(e)}"}, None, None

# Gradio interface
title = "Breast Cancer Ultrasound Analysis"
description = """
1. Segments the ultrasound image (color overlay shows lesion)
2. Classifies the image with overlay
Models:
- U-Net: 224x224 grayscale input
- CNN96: 400x400 RGB with overlay
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
                seg_output = gr.Image(label="Image with Lesion Overlay")

    analyze_btn.click(
        fn=analyze_image,
        inputs=image_input,
        outputs=[cls_output, seg_output, gr.Image(visible=False)]
    )

if __name__ == "__main__":
    demo.launch()