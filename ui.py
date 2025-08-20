
import os
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt

from config import example_images
from analysis import analyze_core

# --------------------------------------------------------------------------------
# Session stats (in-memory for the current run)
# --------------------------------------------------------------------------------
session_stats = {"total": 0, "benign": 0, "malignant": 0, "normal": 0}

# Build sample gallery list as (display_name, path)
sample_images = [(os.path.basename(p), p) for p in example_images]

custom_css = """
body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
.main-header { text-align:center; padding:20px; }
.analysis-card, .results-panel, .statistics-container {
  background:#fff; border-radius:12px; padding:16px; box-shadow:0 2px 10px rgba(0,0,0,.06);
}
"""

# --------------------------------------------------------------------------------
# Plot helpers (matplotlib only, no styles/colors specified)
# --------------------------------------------------------------------------------
def _fig_image(img_np, title):
    fig, ax = plt.subplots()
    ax.imshow(img_np)
    ax.set_title(title)
    ax.axis("off")
    return fig

def _fig_bar(probs, title):
    labels = list(probs.keys())
    vals = [probs[k] for k in labels]
    fig, ax = plt.subplots()
    ax.bar(labels, vals)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_ylabel("Probability")
    return fig

def _fig_radar(probs, title):
    labels = list(probs.keys())
    vals = [probs[k] for k in labels]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    vals = vals + [vals[0]]
    angles = np.concatenate([angles, angles[:1]])
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, vals)
    ax.fill(angles, vals, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_title(title)
    return fig

def _fig_uncertainty(probs, title):
    u = 1.0 - float(max(probs.values()))
    fig, ax = plt.subplots()
    ax.bar(["Uncertainty"], [u])
    ax.set_ylim(0, 1)
    ax.set_title(title)
    return fig

def _fig_risk_gauge(probs, title):
    risk = float(probs.get("Malignant (Suspicious)", 0.0))
    fig, ax = plt.subplots()
    ax.bar(["Risk (Malignant)"], [risk])
    ax.set_ylim(0, 1)
    ax.set_title(title)
    return fig

def _fig_ultrasound_stats(image_np):
    """Generate a histogram of intensities and an edges count bar on the same axes.
    Since each tab only supports one figure, we visualize histogram only (most informative)."""
    img = np.array(image_np)
    if img.ndim == 3:
        import cv2
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    fig, ax = plt.subplots()
    ax.hist(img.flatten(), bins=32)
    ax.set_title("Ultrasound Intensity Histogram")
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Frequency")
    return fig

def _make_diagnosis_html(probs, analysis_html):
    b = probs.get("Benign (Non-cancerous)", 0.0)
    m = probs.get("Malignant (Suspicious)", 0.0)
    n = probs.get("Normal (No findings)", 0.0)
    return f"""
    <div>
      <h3>🧠 AI Diagnosis (Ultrasound)</h3>
      <ul>
        <li><b>Benign:</b> {b:.3f}</li>
        <li><b>Malignant:</b> {m:.3f}</li>
        <li><b>Normal:</b> {n:.3f}</li>
      </ul>
      <div>{analysis_html}</div>
    </div>
    """

def _update_stats(probs):
    session_stats["total"] += 1
    top = max(probs, key=probs.get)
    if "Benign" in top:
        session_stats["benign"] += 1
    elif "Malignant" in top:
        session_stats["malignant"] += 1
    elif "Normal" in top:
        session_stats["normal"] += 1

def get_analysis_statistics():
    total = session_stats["total"]
    if total == 0:
        return "<div style='padding:10px;'>No analyses yet this session.</div>"
    return (
        "<div style='padding:10px;'>"
        f"<p><b>Total analyses:</b> {total}</p>"
        f"<p>Benign: {session_stats['benign']} | "
        f"Malignant: {session_stats['malignant']} | "
        f"Normal: {session_stats['normal']}</p>"
        "</div>"
    )

# --------------------------------------------------------------------------------
# Gradio Interface
# --------------------------------------------------------------------------------
with gr.Blocks(css=custom_css, title="BREAST AI PRO - Enhanced Cancer Detection") as demo:
    # Header
    with gr.Row():
        with gr.Column():
            gr.HTML(
                """
                <div class="main-header">
                    <h1>🔬 BREAST AI PRO</h1>
                    <p>Advanced Breast Ultrasound Analysis with Uncertainty Quantification</p>
                    <p style="font-size: 0.9rem; color: #666;">Version 2.0 | Diagnostic AI for Ultrasound</p>
                </div>
                """
            )

    # Tabs
    with gr.Tabs() as tabs:
        # Main Analysis Tab
        with gr.TabItem("🔬 AI Analysis"):
            with gr.Row(equal_height=True):
                # Left Column - Input
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.HTML('<div class="analysis-card">')
                        gr.Markdown("### 📤 **Ultrasound Analysis Center**")
                        gr.Markdown("*Upload breast ultrasound images for AI analysis with uncertainty*")

                        image_input = gr.Image(
                            label="Upload Ultrasound Image",
                            type="numpy",
                            height=400
                        )

                        gr.Markdown(
                            "**Supported formats:** PNG, JPG, JPEG, TIF"
                            "**Recommended size:** 500×500 pixels or larger"
                        )

                        with gr.Row():
                            submit_btn = gr.Button("🔍 Analyze Image", variant="primary")
                            clear_btn = gr.Button("🗑️ Clear", variant="secondary")

                        status_output = gr.Markdown(
                            "⚪ **Status**: Ready — Upload an image to begin analysis"
                        )
                        gr.HTML("</div>")

                    # Sample Gallery
                    if sample_images:
                        with gr.Group():
                            gr.HTML('<div class="analysis-card">')
                            gr.Markdown("### 🖼️ **Sample Ultrasound Images**")
                            gr.Markdown("*Click on any sample image to analyze*")

                            example_gallery = gr.Gallery(
                                value=[(path, name) for name, path in sample_images],
                                columns=4,
                                height="auto"
                            )
                            gr.HTML("</div>")

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
                        gr.HTML("</div>")

        # Statistics Tab
        with gr.TabItem("📈 Analysis Statistics"):
            with gr.Group():
                gr.HTML('<div class="statistics-container">')
                gr.Markdown("### 📊 **Session Statistics**")
                gr.Markdown("*View statistics from all analyses performed in this session*")

                statistics_output = gr.HTML()
                stats_btn = gr.Button("🔄 Refresh Statistics", variant="primary")
                gr.HTML("</div>")

    # -------------------- Event Handlers --------------------
    def handle_submit(image):
        if image is None:
            return (
                "<div style='text-align:center; padding:20px; background-color:#f8d7da; border-radius:10px;'>"
                "<h3 style='color:#721c24;'>No image provided!</h3>"
                "<p>Please upload an image or select from the gallery.</p>"
                "</div>",
                None, None, None, None, None,
                "🔴 **Status**: No image provided"
            )

        probs, display_img, analysis_html, status_text = analyze_core(image)
        _update_stats(probs)
        diagnosis_html = _make_diagnosis_html(probs, analysis_html)

        # Build figures for each tab
        if display_img is None:
            fig_display = _fig_bar(probs, "Confidence Scores")
            fig_stats = _fig_bar(probs, "Confidence Scores")
        else:
            fig_display = _fig_image(display_img, "Ultrasound with Lesion Overlay")
            fig_stats = _fig_ultrasound_stats(display_img)

        fig_radar = _fig_radar(probs, "Confidence Radar")
        fig_uncert = _fig_uncertainty(probs, "Uncertainty (1 - max probability)")
        fig_risk = _fig_risk_gauge(probs, "Risk Assessment (Malignant)")

        return (diagnosis_html, fig_display, fig_stats, fig_radar, fig_uncert, fig_risk, status_text)

    def handle_clear():
        return (
            None,
            "<div style='text-align:center; padding:20px;'>"
            "<p>Upload an image to begin analysis</p>"
            "</div>",
            None, None, None, None, None,
            "⚪ **Status**: Ready — Upload an image to begin analysis"
        )

    def handle_gallery_select(evt: gr.SelectData):
        return sample_images[evt.index][1]

    def handle_statistics():
        return get_analysis_statistics()

    # -------------------- Bindings --------------------
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
