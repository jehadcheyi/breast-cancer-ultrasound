import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import gradio as gr
from tensorflow.keras.models import load_model

# === EncoderBlock for loading UNet ===
class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, pool=True, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size, padding="same", activation="relu")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size, padding="same", activation="relu")
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2)) if pool else None

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.pool:
            return self.pool(x)
        return x

# === Load Models ===
print("Loading models...")
unet_model = load_model("unet.h5", custom_objects={"EncoderBlock": EncoderBlock})
cnn_model = load_model("cnn96.h5")
print("Models loaded successfully!")

# === Constants ===
SEG_SIZE = (224, 224)
CLS_SIZE = (224, 224)
CLASSES = ["Benign", "Malignant", "Normal"]
COLORS = {"Benign": "#4CAF50", "Malignant": "#F44336", "Normal": "#2196F3"}

# === Processing Functions ===
def preprocess_image(image):
    image = cv2.resize(image, SEG_SIZE)
    image = image.astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0)

def get_mask(image):
    preprocessed = preprocess_image(image)
    mask = unet_model.predict(preprocessed, verbose=0)[0]
    return (mask[..., 0] > 0.5).astype(np.uint8)

def extract_lesion(original, mask):
    masked = cv2.bitwise_and(original, original, mask=mask)
    x, y, w, h = cv2.boundingRect(mask)
    cropped = masked[y:y+h, x:x+w]
    if cropped.size == 0:
        raise ValueError("No lesion detected.")
    resized = cv2.resize(cropped, CLS_SIZE)
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0), cropped

def analyze_image(image):
    try:
        image = image.astype('uint8')
        resized = cv2.resize(image, SEG_SIZE)

        mask = get_mask(resized)
        input_for_cls, cropped = extract_lesion(resized, mask)

        pred = cnn_model.predict(input_for_cls, verbose=0)[0]
        pred_idx = np.argmax(pred)
        label = CLASSES[pred_idx]
        confidence = pred[pred_idx] * 100
        color = COLORS[label]

        # === Visualization ===
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        axs[0].imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Original Ultrasound")
        axs[0].axis('off')

        axs[1].imshow(mask, cmap='gray')
        axs[1].set_title("Segmentation Mask")
        axs[1].axis('off')

        axs[2].imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        axs[2].set_title("Segmented Lesion")
        axs[2].axis('off')
        plt.tight_layout()

        # === Diagnosis Box ===
        html = f"""
        <div style="border: 2px solid {color}; border-radius: 10px; padding: 20px; font-family: Arial;">
            <h2 style="color: {color};">Diagnosis: {label}</h2>
            <p style="font-size: 16px;">Confidence: <strong>{confidence:.2f}%</strong></p>
        </div>
        """

        return html, fig

    except Exception as e:
        return f"<div style='color:red'><b>Error:</b> {str(e)}</div>", None

# === Gradio UI ===
with gr.Blocks(title="Breast Ultrasound Analysis") as demo:
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Upload Breast Ultrasound Image")
            image_input = gr.Image(type="numpy", label="Input Ultrasound")
            analyze_btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            diagnosis_html = gr.HTML()
            output_plot = gr.Plot()

    analyze_btn.click(fn=analyze_image, inputs=image_input, outputs=[diagnosis_html, output_plot])

demo.launch(share=True)
