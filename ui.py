import os
import json
import time
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Wedge

from config import example_images, REPORTS_FOLDER, STATS_FILE
from analysis import analyze_core

DEFAULT_STATS = {"total": 0, "benign": 0, "malignant": 0, "normal": 0}

def load_stats():
    try:
        if os.path.exists(STATS_FILE):
            with open(STATS_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return DEFAULT_STATS.copy()

def save_stats(stats):
    try:
        with open(STATS_FILE, "w") as f:
            json.dump(stats, f)
    except Exception as e:
        print("Failed to save stats:", e)

session_stats = load_stats()
sample_images = [(os.path.basename(p), p) for p in example_images]

custom_css = """
:root{
  --bg:#f4f6f8; --card:#ffffff; --text:#0f172a; --muted:#6b7280;
}
body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background:var(--bg); color:var(--text); }
.main-header { text-align:center; padding:20px; }
.analysis-card, .results-panel, .statistics-container { background:var(--card); border-radius:12px; padding:16px; box-shadow:0 2px 14px rgba(16,24,40,0.06); color:var(--text); }
.dark-mode { --bg:#0b1220; --card:#071226; --text:#e6eef6; --muted:#9aa6b2; }
.toggle-btn { background:transparent; border:1px solid var(--muted); padding:6px 10px; border-radius:8px; cursor:pointer; color:var(--text); }
"""

def _pro_side_by_side(original, overlay, title_left='Original', title_right='Overlay'):
    fig, axes = plt.subplots(1,2, figsize=(8,4))
    axes[0].imshow(original, cmap='gray' if original.ndim==2 or original.shape[2]==1 else None)
    axes[0].set_title(title_left); axes[0].axis('off')
    im = axes[1].imshow(overlay)
    axes[1].set_title(title_right); axes[1].axis('off')
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Intensity', rotation=270, labelpad=12)
    fig.tight_layout()
    return fig

def _pro_hist_kde(image, title='Intensity Histogram & KDE'):
    img = np.array(image)
    if img.ndim==3:
        import cv2
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    data = img.flatten().astype(np.float32)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(data, bins=60, density=True, alpha=0.6, edgecolor='black')
    counts, bin_edges = np.histogram(data, bins=200, density=True)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    from math import sqrt, pi, exp
    bw = (data.std() if data.std()>0 else 1.0) * 0.8
    kernel_vals = np.array([np.sum(counts * np.exp(-0.5*((centers - c)/bw)**2) / (sqrt(2*pi)*bw)) for c in centers])
    kernel_vals = kernel_vals / (kernel_vals.max() if kernel_vals.max()!=0 else 1.0) * counts.max()
    ax.plot(centers, kernel_vals, linewidth=2)
    mean = data.mean(); median = np.median(data)
    ax.axvline(mean, color='green', linestyle='--', linewidth=1.5, label=f'Mean: {mean:.1f}')
    ax.axvline(median, color='orange', linestyle='-.', linewidth=1.5, label=f'Median: {median:.1f}')
    ax.set_title(title); ax.set_xlabel('Intensity'); ax.set_ylabel('Density')
    ax.legend()
    fig.tight_layout()
    return fig

def _pro_radar(probs, title='Confidence Radar'):
    labels = list(probs.keys())
    vals = [probs[k] for k in labels]
    N = len(labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    vals_closed = vals + vals[:1]
    angles_closed = angles + angles[:1]
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles_closed, vals_closed, linewidth=2)
    ax.fill(angles_closed, vals_closed, alpha=0.25)
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_thetagrids(np.degrees(angles), labels)
    max_idx = int(np.argmax(list(probs.values())))
    angle_max = angles[max_idx]
    ax.scatter([angle_max], [vals[max_idx]], color='red', s=80, zorder=10)
    ax.set_title(title)
    return fig

def _pro_uncertainty_gauge(probs, title='Uncertainty Gauge'):
    u = 1.0 - float(max(probs.values()))
    fig, ax = plt.subplots(figsize=(5,3), subplot_kw={'projection':'polar'})
    ax.set_theta_offset(np.pi); ax.set_theta_direction(-1)
    ax.set_ylim(0,1)
    ax.barh(0, np.pi, left=0, height=1, color='lightgray', alpha=0.3)
    for i in range(5):
        start = np.pi * (i/5); end = np.pi * ((i+1)/5)
        wedge = Wedge((0,0), 1, np.degrees(start)-90, np.degrees(end)-90)
        ax.add_patch(wedge)
    angle = u * np.pi
    ax.plot([angle], [0.5], marker='o', markersize=12, color='black')
    ax.set_yticks([]); ax.set_xticks([])
    ax.set_title(f"{title} — {u:.2f}")
    return fig

def _pro_risk_speedometer(probs, title='Risk Assessment'):
    risk = float(probs.get('Malignant (Suspicious)', 0.0))
    fig, ax = plt.subplots(figsize=(5,3))
    ax.axis('off')
    wedges = [
        Wedge((0,0), 1, 180, 180+180*0.3, facecolor='green', alpha=0.7),
        Wedge((0,0), 1, 180+180*0.3, 180+180*0.6, facecolor='yellow', alpha=0.7),
        Wedge((0,0), 1, 180+180*0.6, 360, facecolor='red', alpha=0.7)
    ]
    for w in wedges: ax.add_patch(w)
    theta = 180 + 180 * risk
    ax.plot([0, 0.9*np.cos(np.radians(theta))], [0, 0.9*np.sin(np.radians(theta))], color='black', linewidth=3)
    ax.text(0, -0.2, f'Risk: {risk*100:.1f}%', ha='center', fontsize=12, fontweight='bold')
    ax.set_xlim(-1.1,1.1); ax.set_ylim(-0.3,1.1)
    ax.set_title(title)
    return fig

def generate_pdf_report(probs, display_img, analysis_html, status_text):
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    filename = f'report_{timestamp}.pdf'
    path = os.path.join(REPORTS_FOLDER, filename)
    try:
        with PdfPages(path) as pdf:
            fig_text = plt.figure(figsize=(8.27, 11.69))
            fig_text.text(0.02, 0.98, 'BREAST AI PRO - Ultrasound Analysis Report', fontsize=16, weight='bold')
            fig_text.text(0.02, 0.94, time.strftime('%Y-%m-%d %H:%M:%S'), fontsize=10)
            import re
            plain = re.sub('<[^<]+?>', '', analysis_html)
            fig_text.text(0.02, 0.86, 'Analysis Summary:', fontsize=12, weight='bold')
            fig_text.text(0.02, 0.84, plain, fontsize=10)
            plt.axis('off')
            pdf.savefig(fig_text); plt.close(fig_text)

            if display_img is not None:
                fig_disp = _pro_side_by_side(display_img, display_img, 'Display Image', 'Display Image (copy)')
                pdf.savefig(fig_disp); plt.close(fig_disp)

            fig_conf = _pro_radar(probs)
            pdf.savefig(fig_conf); plt.close(fig_conf)

            fig_unc = _pro_uncertainty_gauge(probs)
            pdf.savefig(fig_unc); plt.close(fig_unc)

            fig_risk = _pro_risk_speedometer(probs)
            pdf.savefig(fig_risk); plt.close(fig_risk)
        return path
    except Exception as e:
        print('PDF generation error:', e)
        return None

with gr.Blocks(css=custom_css, title='BREAST AI PRO - Ultrasound (Pro UI)') as demo:
    with gr.Row():
        with gr.Column():
            gr.HTML("""
            <div class='main-header'>
                <h1>🔬 BREAST AI PRO — Ultrasound (Pro Dashboard)</h1>
                <p>High-fidelity visualizations: segmentation overlays, uncertainty & risk gauges</p>
                <button class='toggle-btn' onclick="document.body.classList.toggle('dark-mode')">Toggle Dark Mode</button>
            </div>
            """)

    with gr.Tabs():
        with gr.TabItem('🔬 AI Analysis'):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.HTML('<div class="analysis-card">')
                        gr.Markdown('### 📤 Upload Ultrasound Image')
                        image_input = gr.Image(label='Upload Ultrasound Image', type='numpy', height=400')
                        with gr.Row():
                            submit_btn = gr.Button('🔍 Analyze Image', variant='primary')
                            clear_btn = gr.Button('🗑️ Clear', variant='secondary')
                        status_output = gr.Markdown('⚪ **Status**: Ready')
                        gr.HTML('</div>')
                    if sample_images:
                        with gr.Group():
                            gr.HTML('<div class="analysis-card">')
                            gr.Markdown('### 🖼️ Sample Ultrasound Images')
                            example_gallery = gr.Gallery(value=[(p, n) for n,p in sample_images], columns=4, height='auto')
                            gr.HTML('</div>')
                with gr.Column(scale=2):
                    with gr.Group():
                        gr.HTML('<div class="results-panel">')
                        diagnosis_output = gr.HTML()
                        with gr.Tabs():
                            with gr.TabItem('Enhanced Visualization'):
                                enhanced_plot_output = gr.Plot()
                            with gr.TabItem('Image Statistics'):
                                stats_plot_output = gr.Plot()
                            with gr.TabItem('Confidence Analysis'):
                                radar_plot_output = gr.Plot()
                            with gr.TabItem('Uncertainty Metrics'):
                                uncertainty_plot_output = gr.Plot()
                            with gr.TabItem('Risk Assessment'):
                                risk_gauge_output = gr.Plot()
                        gr.HTML('</div>')
        with gr.TabItem('📈 Session Statistics'):
            with gr.Group():
                gr.HTML('<div class="statistics-container">')
                statistics_output = gr.HTML()
                stats_btn = gr.Button('🔄 Refresh Statistics', variant='primary')
                gr.HTML('</div>')

    report_file = gr.File(label='Download PDF Report')

    def _make_diagnosis_html(probs, analysis_html):
        b = probs.get('Benign (Non-cancerous)', 0.0)
        m = probs.get('Malignant (Suspicious)', 0.0)
        n = probs.get('Normal (No findings)', 0.0)
        return f"""
        <div>
          <h3>🧠 AI Diagnosis</h3>
          <ul>
            <li><b>Benign:</b> {b:.3f}</li>
            <li><b>Malignant:</b> {m:.3f}</li>
            <li><b>Normal:</b> {n:.3f}</li>
          </ul>
          <div>{analysis_html}</div>
        </div>
        """

    def _update_stats_and_persist(probs):
        session_stats['total'] += 1
        top = max(probs, key=probs.get)
        if 'Benign' in top: session_stats['benign'] += 1
        elif 'Malignant' in top: session_stats['malignant'] += 1
        elif 'Normal' in top: session_stats['normal'] += 1
        save_stats(session_stats)

    def get_analysis_statistics():
        total = session_stats.get('total', 0)
        if total == 0: return '<div style="padding:10px;">No analyses yet.</div>'
        return (f"<div style='padding:10px;'><p><b>Total:</b> {total}</p>"
                f"<p>Benign: {session_stats['benign']} | Malignant: {session_stats['malignant']} | Normal: {session_stats['normal']}</p></div>")

    def handle_submit(image):
        if image is None:
            return ('<div style="padding:20px; background:#fee; border-radius:8px;'><h3>No image provided</h3></div>', None, None, None, None, None, None, '🔴 Status: No image')
        probs, display_img, analysis_html, status_text = analyze_core(image)
        _update_stats_and_persist(probs)
        diagnosis_html = _make_diagnosis_html(probs, analysis_html)
        if display_img is None:
            fig_vis = _pro_side_by_side(image, image, 'Original', 'Processed')
            fig_stats = _pro_hist_kde(image)
        else:
            fig_vis = _pro_side_by_side(image, display_img, 'Original', 'Overlay')
            fig_stats = _pro_hist_kde(display_img)
        fig_radar = _pro_radar(probs)
        fig_uncert = _pro_uncertainty_gauge(probs)
        fig_risk = _pro_risk_speedometer(probs)
        pdf_path = generate_pdf_report(probs, display_img if display_img is not None else image, analysis_html, status_text)
        return (diagnosis_html, fig_vis, fig_stats, fig_radar, fig_uncert, fig_risk, pdf_path, status_text)

    def handle_clear():
        return (None, '<div style="padding:10px;">Upload an image to begin.</div>', None, None, None, None, None, '⚪ Status: Ready')

    def handle_gallery_select(evt: gr.SelectData):
        return sample_images[evt.index][1]

    def handle_statistics():
        return get_analysis_statistics()

    submit_btn.click(fn=handle_submit, inputs=[image_input], outputs=[diagnosis_output, enhanced_plot_output, stats_plot_output, radar_plot_output, uncertainty_plot_output, risk_gauge_output, report_file, status_output])
    clear_btn.click(fn=handle_clear, outputs=[image_input, diagnosis_output, enhanced_plot_output, stats_plot_output, radar_plot_output, uncertainty_plot_output, risk_gauge_output, report_file, status_output])
    if sample_images:
        example_gallery.select(fn=handle_gallery_select, outputs=[image_input])
    stats_btn.click(fn=handle_statistics, outputs=[statistics_output])
