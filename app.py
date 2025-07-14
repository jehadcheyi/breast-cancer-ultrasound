import cv2
import numpy as np
import tensorflow as tf
import gradio as gr
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load standard UNet model (no custom objects)
unet_model = load_model("unet.h5")
print("UNet model loaded successfully.")

# Image input size expected by the model
IMG_SIZE = (256, 256)

def segment_image(image):
    try:
        # Resize and normalize input
        resized = cv2.resize(image, IMG_SIZE)
        input_arr = resized.astype(np.float32) / 255.0
        input_arr = np.expand_dims(input_arr, axis=0)  # (1, H, W, 3)

        # Predict mask
        pred_mask = unet_model.predict(input_arr, verbose=0)[0]
        mask = (pred_mask[..., 0] > 0.5).astype(np.uint8)

        # Resize mask to match original image size
        mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Overlay: red mask
        overlay = image.copy()
        overlay[mask_resized == 1] = [255, 0, 0]

        # Plot all results
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
        print(f"Error: {e}")
        return f"<div style='color:red'><b>Error:</b> {str(e)}</div>"

# Gradio UI
with gr.Blocks(title="Breast Ultrasound Segmentation") as demo:
    gr.Markdown("## Upload a Breast Ultrasound Image for Segmentation")
    image_input = gr.Image(type="numpy", label="Input Image")
    btn = gr.Button("Segment")
    plot_output = gr.Plot()

    btn.click(fn=segment_image, inputs=image_input, outputs=plot_output)

demo.launch(share=True)
