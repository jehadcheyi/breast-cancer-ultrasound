import tensorflow as tf
from tensorflow.keras import layers
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
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

# === Load Model ===
print("Loading segmentation model...")
unet_model = load_model("unet.h5", custom_objects={"EncoderBlock": EncoderBlock})
print("Model loaded successfully!")

# === Constants ===
SEG_SIZE = (224, 224)

# === Processing Functions ===
def preprocess_image(image):
    image = cv2.resize(image, SEG_SIZE)
    image = image.astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0)

def get_mask(image):
    preprocessed = preprocess_image(image)
    mask = unet_model.predict(preprocessed, verbose=0)[0]
    return (mask[..., 0] > 0.5).astype(np.uint8)

def analyze_image(image):
    try:
        image = image.astype('uint8')
        resized = cv2.resize(image, SEG_SIZE)
        mask = get_mask(resized)

        # === Visualization ===
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image
        axs[0].imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Original Ultrasound")
        axs[0].axis('off')
        
        # Segmentation mask
        axs[1].imshow(mask, cmap='gray')
        axs[1].set_title("Segmentation Mask")
        axs[1].axis('off')
        
        plt.tight_layout()

        return "Segmentation completed successfully.", fig

    except Exception as e:
        return f"<div style='color:red'><b>Error:</b> {str(e)}</div>", None

# === Gradio UI ===
with gr.Blocks(title="Breast Ultrasound Segmentation") as demo:
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Upload Breast Ultrasound Image")
            image_input = gr.Image(type="numpy", label="Input Ultrasound")
            analyze_btn = gr.Button("Segment", variant="primary")
        with gr.Column():
            status_html = gr.HTML()
            output_plot = gr.Plot()

    analyze_btn.click(fn=analyze_image, inputs=image_input, outputs=[status_html, output_plot])

demo.launch(share=True)