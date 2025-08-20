import gradio as gr
import os

from config import example_images
from analysis import analyze_image

# Build sample_images as list of (name, path) tuples
sample_images = [(os.path.basename(path), path) for path in example_images]

# Stub for statistics (replace with real implementation later)
def get_analysis_statistics():
    return """
    <div style="padding: 20px;">
        <h3>📈 Analysis Statistics (Demo)</h3>
        <p>No analysis statistics collected yet.</p>
    </div>
    """

# ---------- Gradio Interface ----------
custom_css = """
body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
"""

with gr.Blocks(css=custom_css, title="BREAST AI PRO - Enhanced Cancer Detection") as demo:
    # Header
    with gr.Row():
        with gr.Column():
            gr.HTML(f"""
            <div class="main-header">
                <h1>🔬 BREAST AI PRO</h1>
                <p>Enhanced Histopathology Analysis System with Uncertainty Quantification</p>
                <p style="font-size: 0.9rem; color: #666;">Version 2.0 | Advanced Diagnostic AI</p>
            </div>
            """)
    
    # Tabs
    with gr.Tabs() as tabs:
        # Main Analysis Tab
        with gr.TabItem("🔬 AI Analysis"):
            with gr.Row(equal_height=True):
                # Left Column - Input
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.HTML('<div class="analysis-card">')
                        gr.Markdown("### 📤 **Enhanced AI Analysis Center**")
                        gr.Markdown("*Upload histopathology images for comprehensive AI analysis with uncertainty quantification*")
                        
                        image_input = gr.Image(
                            label="Upload Histopathology Image",
                            type="numpy",
                            height=400
                        )
                        
                        gr.Markdown("""
                        **Supported formats:** PNG, JPG, JPEG, TIF  
                        **Recommended size:** 500x500 pixels or larger
                        """)
                        
                        with gr.Row():
                            submit_btn = gr.Button("🔍 Analyze Image", variant="primary")
                            clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                        
                        status_output = gr.Markdown(
                            "⚪ **Status**: Ready - Upload an image to begin analysis"
                        )
                        gr.HTML('</div>')
                    
                    # Sample Gallery
                    if sample_images:
                        with gr.Group():
                            gr.HTML('<div class="analysis-card">')
                            gr.Markdown("### 🖼️ **Sample Images**")
                            gr.Markdown("*Click on any sample image to analyze*")
                            
                            example_gallery = gr.Gallery(
                                value=[(path, name) for name, path in sample_images],
                                columns=4,
                                height="auto"
                            )
                            gr.HTML('</div>')
                
                # Right Column - Results
                with gr.Column(scale=2):
                    with gr.Group():
                        gr.HTML('<div class="results-panel">')
                        gr.Markdown("### 📊 **Analysis Results**")
                        
                        diagnosis_output = gr.HTML()
                        
                        with gr.Tabs():
                            with gr.TabItem("Enhanced Visualization"):
                                enhanced_plot_output = gr.Plot()
                            with gr.TabItem("Image Statistics"):
                                stats_plot_output = gr.Plot()
                            with gr.TabItem("Confidence Analysis"):
                                radar_plot_output = gr.Plot()
                            with gr.TabItem("Uncertainty Metrics"):
                                uncertainty_plot_output = gr.Plot()
                            with gr.TabItem("Risk Assessment"):
                                risk_gauge_output = gr.Plot()
                        gr.HTML('</div>')
        
        # Statistics Tab
        with gr.TabItem("📈 Analysis Statistics"):
            with gr.Group():
                gr.HTML('<div class="statistics-container">')
                gr.Markdown("### 📊 **Comprehensive Analysis Statistics**")
                gr.Markdown("*View statistics from all analyses performed in this session*")
                
                statistics_output = gr.HTML()
                
                stats_btn = gr.Button("🔄 Refresh Statistics", variant="primary")
                gr.HTML('</div>')
    
    # ---------- Event Handlers ----------
    def handle_submit(image):
        if image is None:
            return (
                """<div style="text-align: center; padding: 20px; background-color: #f8d7da; border-radius: 10px;">
                <h3 style="color: #721c24;">No image provided!</h3>
                <p>Please upload an image or select from the gallery.</p>
                </div>""",
                None, None, None, None, None,
                "🔴 **Status**: No image provided"
            )
        return analyze_image(image)
    
    def handle_clear():
        return (
            None,
            """<div style="text-align: center; padding: 20px;">
            <p>Upload an image to begin analysis</p>
            </div>""",
            None, None, None, None, None,
            "⚪ **Status**: Ready - Upload an image to begin analysis"
        )
    
    def handle_gallery_select(evt: gr.SelectData):
        return sample_images[evt.index][1]
    
    def handle_statistics():
        return get_analysis_statistics()
    
    # ---------- Event Bindings ----------
    submit_btn.click(
        fn=handle_submit,
        inputs=[image_input],
        outputs=[
            diagnosis_output,
            enhanced_plot_output,
            stats_plot_output,
            radar_plot_output,
            uncertainty_plot_output,
            risk_gauge_output,
            status_output
        ]
    )

    clear_btn.click(
        fn=handle_clear,
        outputs=[
            image_input,
            diagnosis_output,
            enhanced_plot_output,
            stats_plot_output,
            radar_plot_output,
            uncertainty_plot_output,
            risk_gauge_output,
            status_output
        ]
    )

    if sample_images:
        example_gallery.select(
            fn=handle_gallery_select,
            outputs=[image_input]
        )

    stats_btn.click(
        fn=handle_statistics,
        outputs=[statistics_output]
    )
