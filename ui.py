
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

from config import example_images
from analysis import analyze_image

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
    We developed and trained our own AI model for breast ultrasound analysis, achieving <strong>96% accuracy in lesion detection</strong> and <strong>97% accuracy in segmentation</strong>. 
    The system provides an end-to-end solution, combining <strong>precise lesion detection & segmentation</strong> with <strong>reliable malignancy risk assessment</strong>.
    </p>

</div>
"""

instructions = """
<div style="background: rgba(255, 255, 255, 0.9); text-align: center; padding: 25px; border-radius: 15px; margin: 20px 0;">
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
                    columns=5,
                    rows=5,
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
