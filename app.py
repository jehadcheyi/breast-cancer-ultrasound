import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
from PIL import Image
import gradio as gr

# Load models
unet_path = "unet.h5"
classifier_path = "cnn96.h5"

if not os.path.exists(unet_path):
    raise FileNotFoundError(f"UNet model not found at {unet_path}")
if not os.path.exists(classifier_path):
    raise FileNotFoundError(f"Classifier model not found at {classifier_path}")

print("Loading models...")
unet_model = load_model(unet_path)
cnn_model = load_model(classifier_path)
print("Models loaded successfully!")

# Constants
input_size = (224, 224)
seg_size = (256, 256)

class_names = ['Benign', 'Malignant', 'Normal']
class_colors = ['#4CAF50', '#F44336', '#2196F3']

def preprocess_input(image):
    image = cv2.resize(image, seg_size)
    image = image.astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0)

def get_mask(image):
    preprocessed = preprocess_input(image)
    pred_mask = unet_model.predict(preprocessed, verbose=0)[0]
    mask = (pred_mask[..., 0] > 0.5).astype(np.uint8)
    return mask

def extract_region(original, mask):
    masked = cv2.bitwise_and(original, original, mask=mask.astype(np.uint8))
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    cropped = masked[y:y+h, x:x+w]
    if cropped.size == 0:
        raise ValueError("No region detected for classification.")
    resized = cv2.resize(cropped, input_size)
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0), cropped

def analyze_ultrasound(img):
    try:
        img = img.astype('uint8')
        original = cv2.resize(img, seg_size)

        mask = get_mask(original)
        input_for_cnn, cropped = extract_region(original, mask)

        pred = cnn_model.predict(input_for_cnn, verbose=0)[0]
        pred_idx = np.argmax(pred)
        confidence = pred[pred_idx] * 100
        label = class_names[pred_idx]
        color = class_colors[pred_idx]

        # Visualization
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        axs[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Original Ultrasound")
        axs[0].axis('off')

        axs[1].imshow(mask, cmap='gray')
        axs[1].set_title("Segmentation Mask")
        axs[1].axis('off')

        axs[2].imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        axs[2].set_title("Segmented Region")
        axs[2].axis('off')

        plt.tight_layout()

        # Diagnosis summary
        html = f"""
        <div style="border: 2px solid {color}; border-radius: 10px; padding: 20px; font-family: Arial;">
            <h2 style="color: {color};">Diagnosis: {label}</h2>
            <p style="font-size: 16px;">Confidence: <strong>{confidence:.2f}%</strong></p>
        </div>
        """

        return html, fig

    except Exception as e:
        return f"<div style='color:red'><b>Error:</b> {str(e)}</div>", None

# Gradio UI
with gr.Blocks(title="Breast Ultrasound Segmentation & Classification") as demo:
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Upload Breast Ultrasound Image")
            image_input = gr.Image(type="numpy", label="Ultrasound Image")
            submit_btn = gr.Button("Analyze Image", variant="primary")
        with gr.Column():
            diagnosis_output = gr.HTML()
            plot_output = gr.Plot()

    submit_btn.click(fn=analyze_ultrasound, inputs=image_input, outputs=[diagnosis_output, plot_output])

demo.launch(share=True)
