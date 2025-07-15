import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageEnhance, ImageOps
import cv2
import os
import logging
from tensorflow.keras import regularizers

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load models with enhanced verification
def load_model_with_verification(model_path, model_type='classification'):
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")
        
        model = load_model(model_path)
        logger.info(f"{model_type.capitalize()} model loaded successfully from: {model_path}")
        
        # Model verification
        if model_type == 'classification':
            if model.input_shape[1:] != (224, 224, 3):
                logger.warning(f"Classification model expects input shape (400, 400, 3), but found {model.input_shape[1:]}")
            if model.output_shape[-1] != 3:
                logger.warning(f"Classification model expects 3 output classes, but found {model.output_shape[-1]}")
        elif model_type == 'segmentation':
            if model.input_shape[1:] != (224, 224, 1):
                logger.warning(f"Segmentation model expects input shape (224, 224, 1), but found {model.input_shape[1:]}")
        
        return model
    except Exception as e:
        logger.error(f"Error loading {model_type} model: {str(e)}")
        return None

# Load models
seg_model = load_model_with_verification('unet.h5', 'segmentation')
cls_model = load_model_with_verification('busi.h5', 'classification')

# Define class labels
class_labels = ['benign', 'malignant', 'normal']

# Enhanced preprocessing functions
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
    """Enhanced preprocessing for segmentation"""
    try:
        # Convert to numpy array
        image = np.array(image)
        original_shape = image.shape[:2]
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply histogram equalization
        image = cv2.equalizeHist(image)
        
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
    """Enhanced preprocessing for classification"""
    try:
        # Convert to numpy array
        image = np.array(image)
        
        # Convert to RGB if needed
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = image[:, :, :3]
        
        # Apply CLAHE for contrast enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge((l,a,b))
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Resize to model's expected input size
        image = Image.fromarray(image).resize((224, 224))
        image = np.array(image)
        
        # Enhanced normalization
        image = image.astype('float32') / 255.0
        mean = np.array([0.485, 0.456, 0.406])  # ImageNet mean
        std = np.array([0.229, 0.224, 0.225])   # ImageNet std
        image = (image - mean) / std
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    except Exception as e:
        logger.error(f"Classification preprocessing error: {str(e)}")
        return None

def create_segmentation_overlay(original_image, mask):
    """Enhanced overlay visualization"""
    try:
        # Convert to PIL Images
        original_pil = Image.fromarray(original_image)
        mask_pil = Image.fromarray((mask * 255).astype('uint8'))
        
        # Rescale both images
        rescaled_original = rescale_image(original_pil)
        rescaled_mask = rescale_image(mask_pil)
        
        # Create semi-transparent green mask overlay
        green_mask = Image.new("RGBA", rescaled_mask.size, (0, 255, 0, 128))
        mask_alpha = rescaled_mask.convert("L").point(lambda x: min(x, 200))
        green_mask.putalpha(mask_alpha)
        
        # Combine images
        overlayed = Image.alpha_composite(
            rescaled_original.convert("RGBA"), 
            green_mask
        )
        
        # Apply enhancements
        overlayed = fuzzy_enhancement(overlayed)
        overlayed = sharpness_enhancement(overlayed)
        overlayed = brightness_enhancement(overlayed)
        
        return overlayed
    except Exception as e:
        logger.error(f"Overlay creation error: {e}")
        return None

def analyze_ultrasound(image):
    """Enhanced analysis with diagnostic checks"""
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
        
        # Calculate lesion area percentage
        lesion_area = np.sum(mask > 0.5) / (mask.shape[0] * mask.shape[1])
        
        overlay = create_segmentation_overlay(original_image, mask)
        if overlay is None:
            return None, "Overlay creation failed", {}
        
        # Perform classification
        cls_processed = preprocess_for_classification(original_image)
        if cls_processed is None:
            return overlay, "Segmentation complete (classification failed)", {}
        
        predictions = cls_model.predict(cls_processed, verbose=0)[0]
        
        # Apply lesion area consideration for normal cases
        if lesion_area < 0.05:  # Very small lesion area
            predictions[2] *= 1.5  # Boost normal probability
            predictions /= np.sum(predictions)  # Renormalize
        
        confidences = {
            'benign': float(predictions[0]),
            'malignant': float(predictions[1]),
            'normal': float(predictions[2])
        }
        
        # Add diagnostic information
        status = f"Analysis complete (Lesion area: {lesion_area:.1%})"
        
        return overlay, status, confidences
    
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return None, f"Error during analysis: {str(e)}", {}

# Create enhanced Gradio interface
title = "Breast Cancer Ultrasound Analysis"
description = """
Upload an ultrasound image for comprehensive analysis:
1. Segmentation of potential lesions (green overlay)
2. Classification (benign/malignant/normal with confidence scores)
3. Lesion area percentage calculation

Note: For normal images, the system considers lesion area to improve accuracy.
"""

with gr.Blocks() as demo:
    gr.Markdown(f"## {title}")
    gr.Markdown(description)
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload Ultrasound Image", type="pil")
            analyze_btn = gr.Button("Analyze Image", variant="primary")
            
            # Add example images if available
            example_files = ["example_benign.png", "example_malignant.png", "example_normal.png"]
            if all(os.path.exists(f) for f in example_files):
                gr.Examples(
                    examples=[[f] for f in example_files],
                    inputs=image_input,
                    label="Example Images"
                )
        
        with gr.Column():
            output_image = gr.Image(label="Segmentation Result")
            output_message = gr.Textbox(label="Analysis Status")
            output_label = gr.Label(num_top_classes=3, label="Classification Results")
            
            # Add interpretation guide
            with gr.Accordion("Interpretation Guide", open=False):
                gr.Markdown("""
                - **Benign**: Non-cancerous tissue (typically rounded, smooth margins)
                - **Malignant**: Cancerous tissue (often irregular, spiculated margins)
                - **Normal**: Healthy tissue with minimal or no lesions
                - **Lesion Area**: Percentage of image showing potential lesions
                """)
    
    analyze_btn.click(
        fn=analyze_ultrasound,
        inputs=image_input,
        outputs=[output_image, output_message, output_label]
    )

if __name__ == "__main__":
    demo.launch()