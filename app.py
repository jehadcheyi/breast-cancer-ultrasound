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

        return cls_result, final_display, mask

    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return {"Error": f"Analysis failed: {str(e)}"}, None, None

# ... (keep all the previous imports and model loading code unchanged until the CSS section)

# Custom CSS for dark theme with black background
custom_css = """
:root {
    --primary: #6c63ff;
    --primary-dark: #5a52e0;
    --text: #e5e7eb;
    --text-light: #f3f4f6;
    --background: #000000;
    --card-bg: #1a1a1a;
    --card-border: #2d2d2d;
    --border: #333333;
}

body {
    font-family: 'Inter', sans-serif;
    color: var(--text);
    background-color: var(--background) !important;
}

.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto;
    padding: 20px;
    background: var(--background) !important;
}

h1, h2, h3, h4 {
    color: var(--text-light);
}

.description {
    color: var(--text);
    margin-bottom: 24px;
    line-height: 1.6;
}

.card {
    background: var(--card-bg) !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    padding: 24px;
    margin-bottom: 24px;
    border: 1px solid var(--card-border) !important;
}

.tab-button {
    font-weight: 500 !important;
    background: var(--card-bg) !important;
}

.primary-button {
    background: var(--primary) !important;
    color: white !important;
    border: none !important;
    padding: 10px 24px !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease;
}

.primary-button:hover {
    background: var(--primary-dark) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}

.gallery {
    border-radius: 12px !important;
    overflow: hidden;
    background: var(--card-bg) !important;
    border: 1px solid var(--border) !important;
}

.gallery-item {
    border-radius: 8px !important;
    transition: transform 0.2s;
    border: 1px solid var(--border) !important;
}

.gallery-item:hover {
    transform: scale(1.02);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}

.label-container {
    background: var(--card-bg);
    border-radius: 8px;
    padding: 16px;
    margin-top: 16px;
    border: 1px solid var(--border);
}

.progress-bar {
    height: 8px !important;
    border-radius: 4px !important;
}

.image-preview {
    border-radius: 12px !important;
    overflow: hidden;
    border: 1px solid var(--border) !important;
}

.segmentation-mask {
    background: #000;
    border-radius: 12px !important;
    overflow: hidden;
    border: 1px solid var(--border) !important;
}

/* Style the label component for dark theme */
.gr-label {
    color: var(--text) !important;
}

/* Style the confidence bars */
.gr-label .confidence-bar {
    background: var(--primary) !important;
}

.gr-label .confidence-label {
    color: var(--text-light) !important;
}

/* Input styling */
.gr-image-upload {
    background: var(--card-bg) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 12px !important;
}

.gr-image-upload:hover {
    border-color: var(--primary) !important;
}
"""

# Gradio interface (keep the rest of the code the same as before)
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    # Header Section with dark theme adjustments
    with gr.Column(elem_classes=["card"]):
        gr.Markdown(f"## {title}", elem_classes=["header"])
        gr.Markdown(description, elem_classes=["description"])
    
    # Main Content
    with gr.Row():
        # Input Column
        with gr.Column(scale=1, min_width=400):
            with gr.Column(elem_classes=["card"]):
                gr.Markdown("### Upload Image", elem_classes=["header"])
                image_input = gr.Image(label="", type="pil", elem_classes=["image-preview"])
                analyze_btn = gr.Button("Analyze", variant="primary", elem_classes=["primary-button"])
                
                # Example Gallery
                if example_images:
                    with gr.Column():
                        gr.Markdown("### Example Images", elem_classes=["header"])
                        gr.Markdown("Click on an example below to load it:", elem_classes=["description"])
                        example_gallery = gr.Gallery(
                            value=example_images,
                            label=None,
                            columns=3,
                            rows=2,
                            height="auto",
                            object_fit="contain",
                            elem_classes=["gallery"]
                        )
            
            # Function to handle gallery selection
            def select_example(evt: gr.SelectData):
                return example_images[evt.index]
            
            if example_images:
                example_gallery.select(select_example, outputs=image_input)

        # Output Column
        with gr.Column(scale=1, min_width=500):
            with gr.Column(elem_classes=["card"]):
                gr.Markdown("### Analysis Results", elem_classes=["header"])
                
                # Classification Results
                with gr.Column(elem_classes=["label-container"]):
                    gr.Markdown("**Diagnosis Confidence**", elem_classes=["header"])
                    cls_output = gr.Label(label="", show_label=False)
                
                # Segmentation Results
                gr.Markdown("**Segmentation Visualization**", elem_classes=["header"])
                seg_output = gr.Image(label="Lesion highlighted in green", elem_classes=["image-preview"])
                
                # Raw Mask (hidden by default)
                gr.Markdown("**Segmentation Mask**", visible=False)
                mask_output = gr.Image(label="Segmentation Mask", visible=False, elem_classes=["segmentation-mask"])

    # Analysis function
    analyze_btn.click(
        fn=analyze_image,
        inputs=image_input,
        outputs=[cls_output, seg_output, mask_output]
    )

if __name__ == "__main__":
    demo.launch()