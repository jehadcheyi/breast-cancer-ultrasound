
import numpy as np
import cv2

from config import seg_model, cls_model, logger
from preprocessing import (
    preprocess_for_segmentation,
    preprocess_for_classification,
    create_segmentation_overlay
)

SAFE_LABEL_DEFAULT = {
    'Benign (Non-cancerous)': 0.0,
    'Malignant (Suspicious)': 0.0,
    'Normal (No findings)': 0.0
}

def analyze_core(image):
    """Core analysis returning probabilities, display image, analysis html, and status."""
    if seg_model is None or cls_model is None:
        return SAFE_LABEL_DEFAULT, None, (
            "❌ <b>Analysis Failed</b>: Models could not be loaded. "
            "Please check your configuration."
        ), "🔴 Status: System Error"

    if image is None:
        return SAFE_LABEL_DEFAULT, None, (
            "⚠️ <b>No Image Provided</b>: Please upload an image to begin."
        ), "🟡 Status: Waiting for Input"

    try:
        original_image = np.array(image)
        seg_input, original_shape = preprocess_for_segmentation(original_image)
        if seg_input is None:
            return SAFE_LABEL_DEFAULT, None, (
                "❌ <b>Processing Error</b>: Segmentation preprocessing failed."
            ), "🔴 Status: Processing Failed"

        mask = seg_model.predict(seg_input, verbose=0)[0]
        mask = cv2.resize(mask.squeeze(), (original_shape[1], original_shape[0]))
        _, binary_mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask.astype('uint8'), cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        overlay = create_segmentation_overlay(original_image, binary_mask)
        if overlay is None:
            return SAFE_LABEL_DEFAULT, None, (
                "❌ <b>Visualization Error</b>: Unable to create overlay."
            ), "🔴 Status: Visualization Failed"

        cls_input = preprocess_for_classification(overlay)
        if cls_input is None:
            return SAFE_LABEL_DEFAULT, None, (
                "❌ <b>Classification Error</b>: Preprocessing failed."
            ), "🔴 Status: Classification Failed"

        preds = cls_model.predict(cls_input, verbose=0)[0]
        cls_result = {
            'Benign (Non-cancerous)': float(preds[0]),
            'Malignant (Suspicious)': float(preds[1]),
            'Normal (No findings)': float(preds[2])
        }

        max_idx = int(np.argmax(preds))
        max_conf = float(preds[max_idx]) * 100.0

        if max_idx == 0:
            analysis_html = f"""
            <h3>🟢 BENIGN FINDING</h3>
            <p><b>Confidence:</b> {max_conf:.1f}%</p>
            <ul>
              <li>Non-cancerous pattern detected</li>
              <li>Low malignancy risk</li>
              <li>Consider routine follow-up</li>
            </ul>
            """
            status_text = "🟢 Status: Benign - Low Risk"
        elif max_idx == 1:
            analysis_html = f"""
            <h3>🔴 SUSPICIOUS FINDING</h3>
            <p><b>Confidence:</b> {max_conf:.1f}%</p>
            <ul>
              <li>Findings suggest malignancy</li>
              <li>Prompt clinical follow-up recommended</li>
              <li>Consider additional diagnostics</li>
            </ul>
            """
            status_text = "🔴 Status: Suspicious - Immediate Attention"
        else:
            analysis_html = f"""
            <h3>✅ NORMAL RESULT</h3>
            <p><b>Confidence:</b> {max_conf:.1f}%</p>
            <ul>
              <li>No suspicious abnormalities detected</li>
              <li>Continue routine screening</li>
            </ul>
            """
            status_text = "✅ Status: Normal - No Significant Findings"

        display_img = overlay if max_idx != 2 else original_image
        return cls_result, display_img, analysis_html, status_text

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return SAFE_LABEL_DEFAULT, None, (
            f"""<h3>❌ SYSTEM ERROR</h3>
            <p><b>Error:</b> {e}</p>"""
        ), "🔴 Status: System Error"
