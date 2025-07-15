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

def preprocess_for_segmentation(image):
    try:
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        original_shape = image.shape[:2]
        
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        image = cv2.resize(image, (224, 224))
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=-1)
        return np.expand_dims(image, axis=0), original_shape
    except Exception as e:
        logger.error(f"Segmentation preprocessing error: {str(e)}")
        return None, None

def preprocess_for_classification(image):
    try:
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        
        image = cv2.resize(image, (400, 400))
        image = (image.astype('float32') / 255.0) - 0.5
        return np.expand_dims(image, axis=0)
    except Exception as e:
        logger.error(f"Classification preprocessing error: {str(e)}")
        return None

def create_segmentation_overlay(original_image, mask):
    try:
        if isinstance(original_image, Image.Image):
            original_pil = original_image
            original_image = np.array(original_image)
        else:
            original_pil = Image.fromarray(original_image)
        
        mask = (mask * 255).astype('uint8')
        mask = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]))
        
        # Refine mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay_mask = np.zeros_like(mask)
        cv2.drawContours(overlay_mask, contours, -1, 255, -1)
        
        # Create overlay
        green_mask = Image.new("RGB", original_pil.size, (0, 255, 0))
        green_mask.putalpha(Image.fromarray(overlay_mask).convert("L"))
        
        overlayed = Image.alpha_composite(original_pil.convert("RGBA"), green_mask)
        return np.array(overlayed)
    except Exception as e:
        logger.error(f"Overlay creation error: {str(e)}")
        return None

def analyze_image(image):
    if seg_model is None or cls_model is None:
        return {"Error": "Models failed to load"}, None
    
    try:
        # Convert to numpy array if PIL Image
        if isinstance(image, Image.Image):
            original_image = np.array(image)
        else:
            original_image = image.copy()
        
        # Segmentation
        seg_input, original_shape = preprocess_for_segmentation(original_image)
        if seg_input is None:
            return {"Error": "Segmentation preprocessing failed"}, None
        
        mask = seg_model.predict(seg_input, verbose=0)[0].squeeze()
        mask = cv2.resize(mask, (original_shape[1], original_shape[0]))
        
        # Threshold and clean mask
        _, binary_mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)
        kernel = np.ones((5,5), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask.astype('uint8'), cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        # Create overlay
        overlay = create_segmentation_overlay(original_image, binary_mask)
        if overlay is None:
            return {"Error": "Overlay creation failed"}, None
        
        # Classification
        cls_input = preprocess_for_classification(overlay)
        if cls_input is None:
            return {"Error": "Classification preprocessing failed"}, None
        
        predictions = cls_model.predict(cls_input, verbose=0)[0]
        cls_result = {
            'benign': float(predictions[0]),
            'malignant': float(predictions[1]),
            'normal': float(predictions[2])
        }
        
        # Return original image if normal, otherwise overlay
        final_display = original_image if np.argmax(predictions) == 2 else overlay
        
        return cls_result, final_display
    
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return {"Error": f"Analysis failed: {str(e)}"}, None

# Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Breast Ultrasound Diagnostic Assistant  
    **AI-powered analysis of breast ultrasound images**  
    This tool helps medical professionals by:  
    - Segmenting potential lesions (highlighted in green)  
    - Classifying findings as benign, malignant, or normal  
    - Displaying original images when no abnormalities are detected  
    
    *Model Specifications*:  
    - Segmentation: U-Net (224×224 grayscale)  
    - Classification: Custom CNN (400×400 RGB with segmentation overlay)  
    """)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Patient Image Input")
            image_input = gr.Image(
                label="Upload ultrasound scan (PNG, JPG, JPEG, BMP)",
                type="pil",
                height=300
            )
            analyze_btn = gr.Button("Analyze Image", variant="primary")
            
            if example_images:
                gr.Markdown("### Sample Cases")
                gr.Markdown("Click any example below to load it:")
                example_gallery = gr.Gallery(
                    value=example_images,
                    columns=3,
                    height="auto",
                    object_fit="contain"
                )

        with gr.Column():
            gr.Markdown("### Analysis Results")
            with gr.Tab("Classification"):
                cls_output = gr.Label(label="Diagnostic Confidence")
            with gr.Tab("Segmentation"):
                seg_output = gr.Image(label="Lesion Segmentation", height=300)
            
            gr.Markdown("""
            **Interpretation Guide**:  
            - Green areas indicate detected lesions  
            - Normal cases show the original image  
            - Confidence scores range from 0 (low) to 1 (high)  
            """)

    if example_images:
        example_gallery.select(
            fn=lambda evt: example_images[evt.index],
            outputs=image_input
        )

    analyze_btn.click(
        fn=analyze_image,
        inputs=image_input,
        outputs=[cls_output, seg_output]
    )

if __name__ == "__main__":
    demo.launch()