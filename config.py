
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
        token=os.environ["jehad2"]
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
