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

import gradio as gr
import numpy as np
import json  # Add this import


def analyze_image(image):
    if seg_model is None or cls_model is None:
        return {"error": "Models failed to load"}, None

    try:
        original_image = np.array(image)

        # ... [keep all your existing analysis code] ...

        # Combine results into a properly formatted dictionary
        combined_output = {
            "classification": cls_result,
            "image": final_display
        }

        return combined_output

    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return {"error": f"Analysis failed: {str(e)}"}, None

def render_results(result):
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except:
            return result
            
    if "error" in result:
        return f"Error: {result['error']}"
    
    # Format classification results as HTML
    classification_html = """
    <div style="
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
    ">
        <h3 style="margin-top: 0;">Classification Results</h3>
        <table style="width: 100%; border-collapse: collapse;">
            <tr style="background: #e9ecef;">
                <th style="padding: 8px; text-align: left; border-bottom: 1px solid #ddd;">Class</th>
                <th style="padding: 8px; text-align: left; border-bottom: 1px solid #ddd;">Confidence</th>
            </tr>
    """
    
    for class_name, confidence in result["classification"].items():
        percentage = f"{confidence*100:.2f}%"
        classification_html += f"""
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;">{class_name.capitalize()}</td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;">{percentage}</td>
            </tr>
        """
    
    classification_html += """
        </table>
    </div>
    """
    
    # Return both the classification and image
    return f"""
    <div style="display: flex; flex-direction: column; gap: 20px;">
        {classification_html}
        <div style="
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
        ">
            <h3 style="margin-top: 0;">Processed Image</h3>
            <img src="data:image/png;base64,{gr.processing_utils.encode_array_to_base64(result['image'])}" 
                 style="max-width: 100%; border-radius: 4px;"/>
        </div>
    </div>
    """

# Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(description)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload Ultrasound Image", type="pil")
            analyze_btn = gr.Button("Analyze", variant="primary")
            
            if example_images:
                gr.Markdown("### Example Images (Click to load)")
                example_gallery = gr.Gallery(
                    value=example_images,
                    label="Click on an example image to load it",
                    columns=3,
                    rows=2,
                    height="auto"
                )
                
                def select_example(evt: gr.SelectData):
                    return example_images[evt.index]
                
                example_gallery.select(select_example, outputs=image_input)

        with gr.Column():
            with gr.Tab("Analysis Results"):
                results_output = gr.HTML(label="Results")

    analyze_btn.click(
        fn=analyze_image,
        inputs=image_input,
        outputs=results_output
    ).then(
        fn=render_results,
        inputs=results_output,
        outputs=results_output
    )

if __name__ == "__main__":
    demo.launch()