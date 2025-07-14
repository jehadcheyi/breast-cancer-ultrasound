import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageEnhance
import cv2
import os

# Define custom enhancement functions from your code
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

# Load the pre-trained U-Net model
try:
    model = load_model('UNET_model.h5')
    print("Segmentation model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def preprocess_image(image):
    """Preprocess image for segmentation"""
    try:
        # Convert to numpy array
        image = np.array(image)
        original_shape = image.shape[:2]
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Resize and normalize
        image = cv2.resize(image, (256, 256))
        image = image.astype('float32') / 255.0
        
        # Add batch and channel dimensions
        image = np.expand_dims(image, axis=(0, -1))
        
        return image, original_shape
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return None, None

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
        print(f"Overlay creation error: {e}")
        return None

def segment_ultrasound(image):
    """Main segmentation function"""
    if model is None:
        return None, "Model failed to load"
    
    try:
        # Convert and store original image
        original_image = np.array(image)
        
        # Preprocess for segmentation
        processed_image, original_shape = preprocess_image(original_image)
        if processed_image is None:
            return None, "Image preprocessing failed"
        
        # Generate segmentation mask
        mask = model.predict(processed_image, verbose=0)[0]
        mask = cv2.resize(mask.squeeze(), (original_shape[1], original_shape[0]))
        
        # Create overlay visualization
        overlay = create_segmentation_overlay(original_image, mask)
        
        if overlay is None:
            return None, "Overlay creation failed"
            
        return overlay, "Segmentation successful"
    
    except Exception as e:
        print(f"Segmentation error: {e}")
        return None, f"Error during segmentation: {str(e)}"

# Create Gradio interface
title = "Breast Cancer Ultrasound Segmentation"
description = """
Upload an ultrasound image to generate a segmentation mask highlighting potential lesions.
The app uses a U-Net model trained on breast ultrasound images.
"""

with gr.Blocks() as demo:
    gr.Markdown(f"## {title}")
    gr.Markdown(description)
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload Ultrasound Image", type="pil")
            segment_btn = gr.Button("Segment Image", variant="primary")
        
        with gr.Column():
            output_image = gr.Image(label="Segmentation Result")
            output_message = gr.Textbox(label="Status")
    
    segment_btn.click(
        fn=segment_ultrasound,
        inputs=image_input,
        outputs=[output_image, output_message]
    )

if __name__ == "__main__":
    demo.launch()