import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageEnhance
import cv2
import os
import logging


import traceback
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model

# Set up logging once
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    seg_model_path = hf_hub_download(
        repo_id="jehadcheyi/bc_models",
        filename="unet.h5",
        token=os.environ["jehad"]
    )
    seg_model = load_model(seg_model_path)
    logger.info("Segmentation model loaded successfully!")

    cls_model_path = hf_hub_download(
        repo_id="jehadcheyi/bc_models",
        filename="cnn96.h5",
        token=os.environ["jehad"]
    )
    cls_model = load_model(cls_model_path)
    logger.info("Classification model loaded successfully!")

except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    traceback.print_exc()
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

import gradio as gr

title = "🩺 Breast Ultrasound Diagnostic Assistant"
description = """
<div style='font-size: 16px; line-height: 1.6; font-family: Arial, sans-serif;'>
<b>Welcome to the Breast Ultrasound Diagnostic Assistant</b><br><br>
This intelligent system helps radiologists and researchers analyze ultrasound images of the breast with precision.<br>
Using advanced AI models, it provides:<br>
🟢 <b>Lesion Detection</b>: Highlights potential abnormalities on the scan<br>
📊 <b>Malignancy Prediction</b>: Estimates cancer risk with confidence scores<br>
🖼️ <b>Visual Explanations</b>: Offers clear overlays and segmentation results<br><br>

<b>📌 How to Use:</b>
<ol>
<li>Upload a breast ultrasound image or select one from the gallery</li>
<li>Click <b>'Analyze Image'</b> to let the AI process the scan</li>
<li>Review the diagnostic predictions and visual outputs</li>
</ol>

⚠️ <i><b>Disclaimer:</b> This tool is intended for research and clinical assistance only. It does not replace professional medical diagnosis.</i>
</div>
"""

# Simulated example images list
example_images = []  # <-- replace with your actual examples, e.g., [(img1, "Benign"), (img2, "Malignant")]

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"<h1 style='text-align: center; font-size: 2.5em;'>{title}</h1>")
    gr.Markdown(description)

    with gr.Row():
        # Left Panel: Image Input and Gallery
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("## 📥 Upload Ultrasound Image")
                image_input = gr.Image(label="Choose or drag an image", type="pil")
                analyze_btn = gr.Button("🔍 Analyze Image", variant="primary")

            if example_images:
                gr.Markdown("## 🖼️ Sample Scans")
                gallery = gr.Gallery(
                    value=[(img, name) for img, name in example_images],
                    label="Example Ultrasound Images",
                    columns=3,
                    height="auto",
                    object_fit="contain"
                )

                def select_example(evt: gr.SelectData):
                    return example_images[evt.index][0]

                gallery.select(select_example, outputs=image_input)

        # Right Panel: Results
        with gr.Column(scale=2):
            with gr.Tab("📊 Diagnostic Assessment"):
                gr.Markdown("### AI-Based Diagnosis")
                cls_output = gr.Label(
                    label="Prediction Confidence",
                    num_top_classes=3
                )
                gr.Markdown("""
                **🧠 Interpretation Guide**  
                - <b>Benign</b>: Non-cancerous lesion (usually safe, but monitorable)  
                - <b>Malignant</b>: Suspicious for cancer (requires clinical attention)  
                - <b>Normal</b>: No findings indicative of disease  
                """)

            with gr.Tab("🧬 Lesion Visualization"):
                gr.Markdown("### Segmentation Results")
                seg_output = gr.Image(
                    label="Lesion Region (Green Highlight)",
                    interactive=False
                )
                gr.Markdown("""
                **🧾 Explanation**  
                - Green mask indicates potential lesion zones  
                - Normal cases will show the unaltered image  
                - Enhancements are provided for interpretability  
                """)

    # Event trigger
    analyze_btn.click(
        fn=analyze_image,  # Make sure your function is defined
        inputs=image_input,
        outputs=[cls_output, seg_output]
    )

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch()
