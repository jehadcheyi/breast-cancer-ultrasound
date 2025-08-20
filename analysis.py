
import numpy as np
import cv2

from config import seg_model, cls_model, logger
from preprocessing import (
    preprocess_for_segmentation,
    preprocess_for_classification,
    create_segmentation_overlay
)

# Always return numeric values for the UI's Label/plots (avoid pydantic errors)
SAFE_LABEL_DEFAULT = {
    "Benign (Non-cancerous)": 0.0,
    "Malignant (Suspicious)": 0.0,
    "Normal (No findings)": 0.0,
}

def _entropy(probs):
    p = np.clip(np.array(list(probs.values()), dtype=np.float32), 1e-9, 1.0)
    return float(-(p * np.log(p)).sum() / np.log(len(p)))

def analyze_core(image):
    """Run the full ultrasound pipeline and return:
    (probs_dict, display_image_np, analysis_html, status_text)
    """
    if seg_model is None or cls_model is None:
        return SAFE_LABEL_DEFAULT, None, (
            "❌ <b>Analysis Failed</b>: Models could not be loaded. Check credentials or network."
        ), "🔴 Status: System Error"

    if image is None:
        return SAFE_LABEL_DEFAULT, None, (
            "⚠️ <b>No Image Provided</b>: Please upload an ultrasound image to begin."
        ), "🟡 Status: Waiting for Input"

    try:
        # Ensure numpy array
        original_image = np.array(image)

        # Segmentation
        seg_input, original_shape = preprocess_for_segmentation(original_image)
        if seg_input is None:
            return SAFE_LABEL_DEFAULT, None, (
                "❌ <b>Processing Error</b>: Segmentation preprocessing failed."
            ), "🔴 Status: Processing Failed"

        mask = seg_model.predict(seg_input, verbose=0)[0]
        mask = cv2.resize(mask.squeeze(), (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)

        # Binary refinement
        _, binary_mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask.astype('uint8'), cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        # Overlay for classification
        overlay = create_segmentation_overlay(original_image, binary_mask)
        if overlay is None:
            return SAFE_LABEL_DEFAULT, None, (
                "❌ <b>Visualization Error</b>: Unable to create overlay."
            ), "🔴 Status: Visualization Failed"

        # Classification
        cls_input = preprocess_for_classification(overlay)
        if cls_input is None:
            return SAFE_LABEL_DEFAULT, None, (
                "❌ <b>Classification Error</b>: Preprocessing failed."
            ), "🔴 Status: Classification Failed"

        preds = cls_model.predict(cls_input, verbose=0)[0]
        probs = {
            "Benign (Non-cancerous)": float(preds[0]),
            "Malignant (Suspicious)": float(preds[1]),
            "Normal (No findings)": float(preds[2]),
        }

        # Analysis summary
        top_idx = int(np.argmax(preds))
        top_conf = float(preds[top_idx]) * 100.0
        if top_idx == 0:
            interpretation = "🟢 <b>Benign pattern</b> — low malignancy risk, consider routine follow-up."
            status_text = "🟢 Status: Benign - Low Risk"
        elif top_idx == 1:
            interpretation = "🔴 <b>Suspicious lesion</b> — prompt clinical work-up recommended."
            status_text = "🔴 Status: Suspicious - Immediate Attention"
        else:
            interpretation = "✅ <b>Normal study</b> — no significant suspicious findings."
            status_text = "✅ Status: Normal - No Significant Findings"

        ent = _entropy(probs)
        analysis_html = f"""
        <h3>🩻 Ultrasound Analysis Summary</h3>
        <p><b>Top class confidence:</b> {top_conf:.1f}%</p>
        <p><b>Uncertainty (entropy):</b> {ent:.3f}</p>
        <p>{interpretation}</p>
        """

        display_img = overlay if top_idx != 2 else original_image
        return probs, display_img, analysis_html, status_text

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return SAFE_LABEL_DEFAULT, None, (
            f"<h3>❌ SYSTEM ERROR</h3><p><b>Error:</b> {e}</p>"
        ), "🔴 Status: System Error"
