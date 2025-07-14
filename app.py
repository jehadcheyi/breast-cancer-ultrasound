import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import gradio as gr
import matplotlib.pyplot as plt

# Custom EncoderBlock must match the one used during training
class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, filters, rate=0.0, pooling=True, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv2D(filters, 3, padding='same', activation='relu')
        self.bn1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(rate)
        self.conv2 = layers.Conv2D(filters, 3, padding='same', activation='relu')
        self.bn2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(rate)
        self.pool = layers.MaxPooling2D(pool_size=2) if pooling else None

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.dropout1(x, training=training)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.dropout2(x, training=training)
        if self.pool:
            return self.pool(x)
        return x

# Load model with custom objects
print("Loading U-Net model...")
unet_model = load_model("unet.h5", custom_objects={"EncoderBlock": EncoderBlock})
print("Model loaded.")

# Expected input size for the model
IMG_SIZE = (256, 256)

def segment_image(image):
    try:
        # Resize and normalize input
        img_resized = cv2.resize(image, IMG_SIZE)
        input_arr = img_resized.astype(np.float32) / 255.0
        input_arr = np.expand_dims(input_arr, axis=0)  # Shape (1, H, W, 3)

        # Predict segmentation mask
        pred_mask = unet_model.predict(input_arr)[0]
        # Assuming output mask shape is (H, W, 1) with sigmoid activation
        binary_mask = (pred_mask[..., 0] > 0.5).astype(np.uint8)

        # Resize mask to original image size
        mask_resized = cv2.resize(binary_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Create overlay (red color mask)
        overlay = image.copy()
        overlay[mask_resized == 1] = [255, 0, 0]

        # Plot original, mask, overlay
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        axs[0].imshow(image)
        axs[0].set_title("Original Image")
        axs[0].axis("off")

        axs[1].imshow(mask_resized, cmap="gray")
        axs[1].set_title("Predicted Mask")
        axs[1].axis("off")

        axs[2].imshow(overlay)
        axs[2].set_title("Overlay")
        axs[2].axis("off")

        plt.tight_layout()
        return fig

    except Exception as e:
        return f"<div style='color:red;'><b>Error:</b> {e}</div>"

# Build Gradio interface
with gr.Blocks(title="Breast Ultrasound Segmentation") as demo:
    gr.Markdown("## Upload Breast Ultrasound Image for Segmentation")
    image_input = gr.Image(type="numpy", label="Input Image")
    segment_btn = gr.Button("Segment")
    output_plot = gr.Plot()

    segment_btn.click(fn=segment_image, inputs=image_input, outputs=output_plot)

demo.launch(share=True)
