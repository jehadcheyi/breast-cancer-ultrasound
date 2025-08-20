
import os
import logging
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model

# -----------------------------------------------------------------------------
# Global configuration & shared resources
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ultrasound_ai_pro")

# Load models (segmentation + classification). Tokens are optional; if absent,
# public access (if allowed) will be attempted.
try:
    seg_model_path = hf_hub_download(
        repo_id="jehadcheyi/bc_models",
        filename="unet.h5",
        token=os.environ.get("jehad2")
    )
    cls_model_path = hf_hub_download(
        repo_id="jehadcheyi/bc_models",
        filename="cnn96.h5",
        token=os.environ.get("jehad2")
    )
    seg_model = load_model(seg_model_path)
    cls_model = load_model(cls_model_path)
    logger.info("✅ Models loaded successfully.")
except Exception as e:
    logger.error(f"❌ Model loading failed: {e}")
    seg_model = None
    cls_model = None

# Example ultrasound images folder
EXAMPLE_FOLDER = "example"
os.makedirs(EXAMPLE_FOLDER, exist_ok=True)

# Gather example images (name, path)
example_images = []
for f in sorted(os.listdir(EXAMPLE_FOLDER)):
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
        example_images.append(os.path.join(EXAMPLE_FOLDER, f))

# Class labels (kept for clarity, not used directly in UI)
CLASS_LABELS = ["benign", "malignant", "normal"]
