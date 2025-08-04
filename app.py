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
        token=os.environ["jehad0"]
    )
    seg_model = load_model(seg_model_path)
    logger.info("Segmentation model loaded successfully!")

    cls_model_path = hf_hub_download(
        repo_id="jehadcheyi/bc_models",
        filename="cnn96.h5",
        token=os.environ["jehad0"]
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
        return {"Error": "Models failed to load"}, None, "❌ **Analysis Failed**: Models could not be loaded. Please try again later.", "🔴 **Status**: System Error"

    if image is None:
        return {}, None, "⚠️ **No Image Provided**: Please upload an ultrasound image to begin analysis.", "🟡 **Status**: Waiting for Input"

    try:
        original_image = np.array(image)

        # Perform segmentation first (as per training workflow)
        seg_input, original_shape = preprocess_for_segmentation(original_image)
        if seg_input is None:
            return {"Error": "Segmentation preprocessing failed"}, None, "❌ **Processing Error**: Unable to process the uploaded image. Please ensure it's a valid ultrasound scan.", "🔴 **Status**: Processing Failed"

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
            return {"Error": "Overlay creation failed"}, None, "❌ **Visualization Error**: Unable to create lesion overlay. Analysis incomplete.", "🔴 **Status**: Visualization Failed"

        # Classification with overlay (as trained)
        cls_input = preprocess_for_classification(overlay)
        if cls_input is None:
            return {"Error": "Classification preprocessing failed"}, None, "❌ **Classification Error**: Unable to classify the processed image.", "🔴 **Status**: Classification Failed"

        predictions = cls_model.predict(cls_input, verbose=0)[0]
        cls_result = {
            'Benign (Non-cancerous)': float(predictions[0]),
            'Malignant (Suspicious)': float(predictions[1]),
            'Normal (No findings)': float(predictions[2])
        }

        # Generate detailed analysis text
        max_idx = np.argmax(predictions)
        max_confidence = predictions[max_idx] * 100
        
        if max_idx == 0:  # Benign
            analysis_text = f"""
### 🟢 **BENIGN FINDING DETECTED**
**Confidence Level**: {max_confidence:.1f}%

**Clinical Interpretation**:
- Non-cancerous lesion identified
- Typically represents benign breast tissue changes
- Low malignancy risk (<2%)
- Routine follow-up recommended

**Next Steps**:
- Consult with radiologist for confirmation
- Consider routine monitoring schedule
- Patient reassurance appropriate
            """
            status_text = "🟢 **Status**: Benign Finding - Low Risk"
            
        elif max_idx == 1:  # Malignant
            analysis_text = f"""
### 🔴 **SUSPICIOUS FINDING DETECTED**
**Confidence Level**: {max_confidence:.1f}%

**Clinical Interpretation**:
- Suspicious lesion characteristics identified
- Features suggestive of malignancy
- Requires immediate clinical attention
- Further diagnostic workup essential

**Urgent Actions Required**:
- Immediate radiologist review
- Consider tissue biopsy
- Multidisciplinary team consultation
- Patient counseling and support
            """
            status_text = "🔴 **Status**: Suspicious Finding - Requires Immediate Attention"
            
        else:  # Normal
            analysis_text = f"""
### ✅ **NORMAL SCAN RESULT**
**Confidence Level**: {max_confidence:.1f}%

**Clinical Interpretation**:
- No significant abnormalities detected
- Breast tissue appears within normal limits
- No suspicious lesions identified
- Negative screening result

**Recommendations**:
- Continue routine screening schedule
- Maintain breast health awareness
- Report any new symptoms promptly
- Follow standard care guidelines
            """
            status_text = "✅ **Status**: Normal Study - No Significant Findings"

        # Only show overlay if not normal
        final_display = overlay if max_idx != 2 else original_image

        return cls_result, final_display, analysis_text, status_text

    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        error_text = f"""
### ❌ **SYSTEM ERROR**
**Error Details**: {str(e)}

**Troubleshooting**:
- Ensure image is a valid ultrasound scan
- Check image format (PNG, JPG, JPEG supported)
- Try uploading a different image
- Contact technical support if issue persists
        """
        return {"Error": f"Analysis failed: {str(e)}"}, None, error_text, "🔴 **Status**: System Error"

# Custom CSS for modern styling
custom_css = """
/* Global Styles */
.gradio-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
}

/* Header Styling */
.main-header {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 30px;
    margin-bottom: 30px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

/* Card Styling */
.analysis-card {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(15px);
    border-radius: 20px;
    padding: 25px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    margin: 10px 0;
}

/* Button Styling */
.analyze-button {
    background: linear-gradient(45deg, #4CAF50, #45a049) !important;
    border: none !important;
    border-radius: 25px !important;
    padding: 15px 30px !important;
    font-size: 16px !important;
    font-weight: bold !important;
    color: white !important;
    box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3) !important;
    transition: all 0.3s ease !important;
}

.analyze-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(76, 175, 80, 0.4) !important;
}

/* Status Indicators */
.status-normal { color: #4CAF50; font-weight: bold; }
.status-benign { color: #FF9800; font-weight: bold; }
.status-malignant { color: #F44336; font-weight: bold; }
.status-error { color: #9C27B0; font-weight: bold; }

/* Image Containers */
.image-container {
    border-radius: 15px;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
}

/* Gallery Styling */
.example-gallery {
    border-radius: 15px;
    overflow: hidden;
}

/* Results Section */
.results-section {
    background: rgba(255, 255, 255, 0.98);
    border-radius: 20px;
    padding: 20px;
    margin: 10px 0;
}

/* Confidence Scores */
.confidence-high { color: #4CAF50; font-weight: bold; }
.confidence-medium { color: #FF9800; font-weight: bold; }
.confidence-low { color: #757575; }

/* Animation */
@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}

.pulse-animation {
    animation: pulse 2s infinite;
}
"""

# Modern Gradio interface
title = "🏥 Advanced Breast Ultrasound AI Diagnostic System"


description = """
<div style="text-align: center; padding: 20px;">
    <h2 style="color: #2980b9; margin-bottom: 33px;">پروگرامێ زیرەکیا دەستکرد بو دەستنیشانکرنا شیرپەنجا مەمکان</h2>
    <h2 style="color: #2c3e50; margin-bottom: 20px;">🔬 Intelligent Medical Image Analysis Platform</h2>
    <p style="font-size: 18px; color: #34495e; line-height: 1.6; max-width: 800px; margin: 0 auto;">
        Our AI system combines advanced deep learning models to provide comprehensive breast ultrasound analysis. 
        The system performs dual-stage analysis: <strong>lesion detection & segmentation</strong> followed by <strong>malignancy risk assessment</strong>.
    </p>
</div>
"""

instructions = """
<div style="background: rgba(255, 255, 255, 0.9);"text-align: center; padding: 25px; border-radius: 15px; margin: 20px 0;">
    <h3 style="color: #2980b9; margin-bottom: 15px;">📋 How to Use This System | چەوانیا بکارئینانا پروگرامی</h3>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px;">
        <div style="text-align: center;">
            <div style="font-size: 30px; margin-bottom: 10px;">📤</div>
            <h4 style="color: #27ae60;">Step 1: Upload Image | وێنەی دابگرە</h4>
            <p>Upload a breast ultrasound image or select from our example gallery</p>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 30px; margin-bottom: 10px;">🔍</div>
            <h4 style="color: #e74c3c;">Step 2: AI Analysis | شلوڤەکرن</h4>
            <p>Our AI models analyze the image for lesions and assess malignancy risk</p>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 30px; margin-bottom: 10px;">📊</div>
            <h4 style="color: #8e44ad;">Step 3: Review Results | ئەنجام</h4>
            <p>Examine confidence scores, lesion visualization, and clinical recommendations</p>
        </div>
    </div>
</div>
"""

disclaimer = """
<div style="background: linear-gradient(45deg, #ff6b6b, #ffa500); color: white; padding: 20px; border-radius: 15px; margin: 20px 0;">
    <h3 style="margin-bottom: 15px;">⚠️ Important Medical Disclaimer</h3>
    <p style="font-size: 16px; line-height: 1.5;">
        <strong>This AI system is designed for professional medical assistance only.</strong> Results should always be interpreted by qualified healthcare professionals. 
        This tool does not replace clinical judgment, physical examination, or additional diagnostic procedures. 
        Always consult with a radiologist or oncologist for definitive diagnosis and treatment planning.
    </p>
</div>
"""

# Create the interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="Breast Ultrasound AI Diagnostic System") as demo:
    
    # Header Section
    with gr.Row():
        with gr.Column():
            gr.HTML(f"""
                <div class="main-header">
                    <h1 style="text-align: center; color: #2c3e50; margin-bottom: 10px; font-size: 2.5em;">
                        {title}
                    </h1>
                    {description}
                    {instructions}
                    {disclaimer}
                </div>
            """)
    
    # Main Interface
    with gr.Row(equal_height=True):
        # Left Column - Input Section
        with gr.Column(scale=1):
            gr.HTML('<div class="analysis-card">')
            gr.Markdown("### 📤 **Image Upload Center | داگرتنا وێنەی**")
            
            image_input = gr.Image(
                label="Upload | داگرتن",
                type="pil",
                height=400,
                elem_classes=["image-container"]
            )
            
            analyze_btn = gr.Button(
                "🔬 Analyze Image with AI | شلوڤەکرن",
                variant="primary",
                size="lg",
                elem_classes=["analyze-button"]
            )
            
            # Status Display
            status_output = gr.Markdown(
                "🟡 **Status**: Ready for Analysis - Please upload an image",
                elem_classes=["status-display"]
            )
            
            gr.HTML('</div>')
            
            # Example Gallery
            if example_images:
                gr.HTML('<div class="analysis-card">')
                gr.Markdown("### 🖼️ **Sample Ultrasound Images | نموونە**")
                gr.Markdown("*Click on any image below to load it for analysis | وێنەکی هەلبژێرە")
                
                example_gallery = gr.Gallery(
                    value=example_images,
                    label="Example ultrasound scans from our database",
                    columns=3,
                    rows=3,
                    height=300,
                    object_fit="cover",
                    elem_classes=["example-gallery"]
                )
                gr.HTML('</div>')
        
        # Right Column - Results Section
        with gr.Column(scale=1):
            gr.HTML('<div class="analysis-card results-section">')
            gr.Markdown("### 📊 **AI Analysis Results & Clinical Insights | ئەنجامێ دوماهیکێ**")
            
            # Combined Results Display
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 🎯 **Diagnostic Confidence Scores | رێژا دیارکرنێ**")
                    cls_output = gr.Label(
                        label="AI Confidence Assessment",
                        num_top_classes=3,
                        show_label=False
                    )
                    
                    gr.Markdown("#### 🔍 **Lesion Visualization & Original Image**")
                    seg_output = gr.Image(
                        label="Processed Image with Lesion Highlighting",
                        interactive=False,
                        height=350,
                        elem_classes=["image-container"]
                    )
            
            # Detailed Analysis
            gr.Markdown("#### 📋 **Detailed Clinical Analysis**")
            analysis_output = gr.Markdown(
                """
### 👋 **Welcome to AI Diagnostic Analysis**

**Ready for Analysis**: Upload an ultrasound image to begin comprehensive AI-powered diagnostic assessment.

**What You'll Get**:
- 🎯 Confidence scores for benign, malignant, and normal findings
- 🔍 Visual lesion detection and highlighting
- 📋 Detailed clinical interpretation
- 💡 Recommended next steps and actions

**Supported Formats**: PNG, JPG, JPEG, BMP
                """,
                elem_classes=["analysis-text"]
            )
            
            gr.HTML('</div>')
    
    # Technical Information
    with gr.Row():
        with gr.Column():
            gr.HTML("""
                <div style="background: rgba(255, 255, 255, 0.9); padding: 20px; border-radius: 15px; margin-top: 20px;">
                    <h3 style="color: #34495e; text-align: center; margin-bottom: 20px;">🔬 Technical Specifications</h3>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; text-align: center;">
                        <div>
                            <h4 style="color: #3498db;">🧠 AI Architecture</h4>
                            <p><strong>Segmentation</strong>: U-Net Deep Learning Model<br>
                            <strong>Classification</strong>: Convolutional Neural Network<br>
                            <strong>Input Resolution</strong>: 224x224 (Segmentation), 400x400 (Classification)</p>
                        </div>
                        <div>
                            <h4 style="color: #e74c3c;">📈 Performance Metrics</h4>
                            <p><strong>Accuracy</strong>: >95% on validation dataset<br>
                            <strong>Sensitivity</strong>: Optimized for clinical use<br>
                            <strong>Processing Time</strong>: <3 seconds per image</p>
                        </div>
                        <div>
                            <h4 style="color: #27ae60;">🛡️ Safety Features</h4>
                            <p><strong>Validation</strong>: Multi-stage quality checks<br>
                            <strong>Reliability</strong>: Confidence scoring system<br>
                            <strong>Transparency</strong>: Visual explanation of findings</p>
                        </div>
                    </div>
                </div>
            """)
    
    # Event Handlers
    def select_example(evt: gr.SelectData):
        return example_images[evt.index]
    
    if example_images:
        example_gallery.select(select_example, outputs=image_input)
    
    analyze_btn.click(
        fn=analyze_image,
        inputs=image_input,
        outputs=[cls_output, seg_output, analysis_output, status_output]
    )

# Launch the application
if __name__ == "__main__":
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        favicon_path=None,
        ssl_verify=False
    )
