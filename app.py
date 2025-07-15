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

# Path to example images folder
EXAMPLE_FOLDER = "example"
if not os.path.exists(EXAMPLE_FOLDER):
    os.makedirs(EXAMPLE_FOLDER)
    logger.warning(f"Created empty examples folder at {EXAMPLE_FOLDER}")

# Get example images
example_images = []
if os.path.exists(EXAMPLE_FOLDER):
    for file in os.listdir(EXAMPLE_FOLDER):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            example_images.append(os.path.join(EXAMPLE_FOLDER, file))
    logger.info(f"Found {len(example_images)} example images")

# Labels
class_labels = ['benign', 'malignant', 'normal']

# Enhancement functions
def fuzzy_enhancement(image):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(2.0)

def sharpness_enhancement(image):
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(1.0)

def brightness_enhancement(image):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(0.5)

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
        original_pil = Image.fromarray(original_image)
        mask_pil = Image.fromarray((mask * 255).astype('uint8'))

        # Refine mask using contours
        mask_cv = np.array(mask_pil)
        contours, _ = cv2.findContours(mask_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay_mask = np.zeros_like(mask_cv)
        cv2.drawContours(overlay_mask, contours, -1, color=255, thickness=-1)  # fill
        mask_pil = Image.fromarray(overlay_mask)

        # Green overlay
        green_mask = Image.new("RGB", original_pil.size, (0, 255, 0))
        green_mask.putalpha(mask_pil.convert("L"))

        overlayed = Image.alpha_composite(original_pil.convert("RGBA"), green_mask)

        overlayed = fuzzy_enhancement(overlayed)
        overlayed = sharpness_enhancement(overlayed)
        overlayed = brightness_enhancement(overlayed)

        return np.array(overlayed)
    except Exception as e:
        logger.error(f"Overlay creation error: {str(e)}")
        return None

def analyze_image(image):
    if seg_model is None or cls_model is None:
        return {"Error": "Models failed to load"}, None, None

    try:
        original_image = np.array(image)

        # Perform segmentation first (as per training workflow)
        seg_input, original_shape = preprocess_for_segmentation(original_image)
        if seg_input is None:
            return {"Error": "Segmentation preprocessing failed"}, None, None

        mask = seg_model.predict(seg_input, verbose=0)[0]
        mask = cv2.resize(mask.squeeze(), (original_shape[1], original_shape[0]))

        # Threshold + morphological refinement
        _, binary_mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask.astype('uint8'), cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        mask = binary_mask

        # Create green overlay (always create it for classification)
        overlay = create_segmentation_overlay(original_image, mask)
        if overlay is None:
            return {"Error": "Overlay creation failed"}, None, None

        # Classification with overlay (as trained)
        cls_input = preprocess_for_classification(overlay)
        if cls_input is None:
            return {"Error": "Classification preprocessing failed"}, None, None

        predictions = cls_model.predict(cls_input, verbose=0)[0]
        cls_result = {
            'benign': float(predictions[0]),
            'malignant': float(predictions[1]),
            'normal': float(predictions[2])
        }

        # Only show overlay if not normal
        final_display = overlay if np.argmax(predictions) != 2 else original_image

        return cls_result, final_display, None

    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return {"Error": f"Analysis failed: {str(e)}"}, None, None

# Gradio interface
title = "Breast Ultrasound Diagnostic Assistant"
description = """
**AI-powered analysis of breast ultrasound images**  
This tool helps medical professionals by:
1. **Segmenting** potential lesions (highlighted in green)
2. **Classifying** findings as benign, malignant, or normal  
3. **Displaying** original images when no abnormalities are detected

*Model Specifications*:  
- Segmentation: U-Net (224×224 grayscale)  
- Classification: Custom CNN (400×400 RGB with segmentation overlay)
"""

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"## {title}")
    gr.Markdown(description)
    
    with gr.Row():
        # Input Column
        with gr.Column():
            with gr.Group():
                gr.Markdown("### Patient Image Input")
                with gr.Row():
                    image_input = gr.Image(
                        label="Upload ultrasound scan (PNG, JPG, JPEG, BMP)",
                        type="pil",
                        height=300
                    )
                with gr.Row():
                    analyze_btn = gr.Button(
                        "Analyze Image",
                        variant="primary",
                        size="lg"
                    )
            
            if example_images:
                with gr.Group():
                    gr.Markdown("### Sample Cases")
                    gr.Markdown("Click any example below to load it:")
                    example_gallery = gr.Gallery(
                        value=example_images,
                        label=None,
                        columns=3,
                        height="auto",
                        object_fit="contain"
                    )

        # Results Column
        with gr.Column():
            with gr.Group():
                gr.Markdown("### Analysis Results")
                
                with gr.Accordion("Classification Report", open=True):
                    cls_output = gr.Label(
                        label="Diagnostic Confidence",
                        num_top_classes=3,
                        show_label=False
                    )
                
                with gr.Accordion("Segmentation Visualization", open=True):
                    seg_output = gr.Image(
                        label="Lesion Segmentation",
                        height=300,
                        interactive=False
                    )
                
                gr.Markdown("""
                **Interpretation Guide**:  
                - Green areas indicate detected lesions  
                - Normal cases show the original image  
                - Confidence scores range from 0 (low) to 1 (high)
                """)

    # Example selection handler
    if example_images:
        example_gallery.select(
            fn=lambda evt: example_images[evt.index],
            outputs=image_input
        )

    analyze_btn.click(
        fn=analyze_image,
        inputs=image_input,
        outputs=[cls_output, seg_output, gr.Image(visible=False)]
    )

if __name__ == "__main__":
    demo.launch()