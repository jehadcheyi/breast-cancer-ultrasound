import numpy as np
import cv2
from PIL import Image, ImageEnhance
from config import logger

def fuzzy_enhancement(image_pil):
    enhancer = ImageEnhance.Contrast(image_pil)
    return enhancer.enhance(2.0)

def sharpness_enhancement(image_pil):
    enhancer = ImageEnhance.Sharpness(image_pil)
    return enhancer.enhance(1.0)

def brightness_enhancement(image_pil):
    enhancer = ImageEnhance.Brightness(image_pil)
    return enhancer.enhance(0.5)

def preprocess_for_segmentation(image_np):
    try:
        img = np.array(image_np)
        original_shape = img.shape[:2]
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        img = img.astype('float32') / 255.0
        img = img[..., None]          # channel dim
        img = img[None, ...]          # batch dim
        return img, original_shape
    except Exception as e:
        logger.error(f"Segmentation preprocessing error: {e}")
        return None, None

def preprocess_for_classification(image_np):
    try:
        img = np.array(image_np)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        img = Image.fromarray(img).resize((400, 400), Image.BILINEAR)
        img = (np.array(img).astype('float32') / 255.0) - 0.5
        return img[None, ...]
    except Exception as e:
        logger.error(f"Classification preprocessing error: {e}")
        return None

def create_segmentation_overlay(original_image, mask):
    try:
        from PIL import Image
        original_pil = Image.fromarray(original_image)
        mask_u8 = (mask * 255).astype('uint8')

        import cv2
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay_mask = np.zeros_like(mask_u8)
        cv2.drawContours(overlay_mask, contours, -1, color=255, thickness=-1)

        mask_pil = Image.fromarray(overlay_mask)
        green = Image.new("RGB", original_pil.size, (0, 255, 0))
        green.putalpha(mask_pil.convert("L"))
        overlay = Image.alpha_composite(original_pil.convert("RGBA"), green)

        overlay = fuzzy_enhancement(overlay)
        overlay = sharpness_enhancement(overlay)
        overlay = brightness_enhancement(overlay)
        return np.array(overlay)
    except Exception as e:
        logger.error(f"Overlay creation error: {e}")
        return None
