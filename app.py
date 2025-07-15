import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageEnhance
import cv2
import os
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
TITLE = "Breast Cancer Ultrasound Analysis"
DESCRIPTION = """
1. Upload an ultrasound image or select from examples
2. Click Analyze to see results
3. Results show classification confidence and processed image
"""
EXAMPLE_FOLDER = "examples"
CLASS_LABELS = ['benign', 'malignant', 'normal']

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

# Get example images
example_images = []
if os.path.exists(EXAMPLE_FOLDER):
    for file in os.listdir(EXAMPLE_FOLDER):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            example_images.append(os.path.join(EXAMPLE_FOLDER, file))
    logger.info(f"Found {len(example_images)} example images")

# ... [keep all your existing functions: enhancement functions, preprocessing, etc.] ...

def analyze_image(image):
    if seg_model is None or cls_model is None:
        return {"error": "Models failed to load"}

    try:
        original_image = np.array(image)

        # Perform segmentation first (as per training workflow)
        seg_input, original_shape = preprocess_for_segmentation(original_image)
        if seg_input is None:
            return {"error": "Segmentation preprocessing failed"}

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
            return {"error": "Overlay creation failed"}

        # Classification with overlay (as trained)
        cls_input = preprocess_for_classification(overlay)
        if cls_input is None:
            return {"error": "Classification preprocessing failed"}

        predictions = cls_model.predict(cls_input, verbose=0)[0]
        cls_result = {
            'benign': float(predictions[0]),
            'malignant': float(predictions[1]),
            'normal': float(predictions[2])
        }

        # Only show overlay if not normal
        final_display = overlay if np.argmax(predictions) != 2 else original_image

        return {
            "classification": cls_result,
            "image": final_display
        }

    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return {"error": f"Analysis failed: {str(e)}"}

def render_results(result):
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except:
            return result
            
    if "error" in result:
        return f"<div style='color: red; padding: 20px;'>{result['error']}</div>"
    
    # Format classification results
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
    
    # Format image display
    image_html = f"""
    <div style="
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
    ">
        <h3 style="margin-top: 0;">Processed Image</h3>
        <img src="data:image/png;base64,{gr.processing_utils.encode_array_to_base64(result['image'])}" 
             style="max-width: 100%; border-radius: 4px;"/>
    </div>
    """
    
    return classification_html + image_html

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# {TITLE}")
    gr.Markdown(DESCRIPTION)

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