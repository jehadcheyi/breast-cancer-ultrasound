
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

from config import seg_model, cls_model, logger
from preprocessing import preprocess_for_segmentation, preprocess_for_classification, create_segmentation_overlay

def analyze_image(image):
    if seg_model is None or cls_model is None:
        return {"Error": "Models failed to load"}, None, "❌ **Analysis Failed**: Models could not be loaded. Please try again later.", "🔴 **Status**: System Error"

    if image is None:
        return {}, None, "⚠️ **No Image Provided**: Please upload an ultrasound image to begin analysis.", "🟡 **Status**: Waiting for Input"

    try:
        original_image = np.array(image)

        # Perform segmentation first (as per training workflow)
        seg_input, original_shape = preprocess_for_segmentation(original_image)
        if seg_input is None:
            return {"Error": "Segmentation preprocessing failed"}, None, "❌ **Processing Error**: Unable to process the uploaded image. Please ensure it's a valid ultrasound scan.", "🔴 **Status**: Processing Failed"

        mask = seg_model.predict(seg_input, verbose=0)[0]
        mask = cv2.resize(mask.squeeze(), (original_shape[1], original_shape[0]))

        # Threshold + morphological refinement
        _, binary_mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask.astype('uint8'), cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        mask = binary_mask

        # Create green overlay (always create it for classification)
        overlay = create_segmentation_overlay(original_image, mask)
        if overlay is None:
            return {"Error": "Overlay creation failed"}, None, "❌ **Visualization Error**: Unable to create lesion overlay. Analysis incomplete.", "🔴 **Status**: Visualization Failed"

        # Classification with overlay (as trained)
        cls_input = preprocess_for_classification(overlay)
        if cls_input is None:
            return {"Error": "Classification preprocessing failed"}, None, "❌ **Classification Error**: Unable to classify the processed image.", "🔴 **Status**: Classification Failed"

        predictions = cls_model.predict(cls_input, verbose=0)[0]
        cls_result = {
            'Benign (Non-cancerous)': float(predictions[0]),
            'Malignant (Suspicious)': float(predictions[1]),
            'Normal (No findings)': float(predictions[2])
        }

        # Generate detailed analysis text
        max_idx = np.argmax(predictions)
        max_confidence = predictions[max_idx] * 100

        if max_idx == 0:  # Benign
            analysis_text = f"""
### 🟢 **BENIGN FINDING DETECTED**
**Confidence Level**: {max_confidence:.1f}%

**Clinical Interpretation**:
- Non-cancerous lesion identified
- Typically represents benign breast tissue changes
- Low malignancy risk (<2%)
- Routine follow-up recommended

**Next Steps**:
- Consult with radiologist for confirmation
- Consider routine monitoring schedule
- Patient reassurance appropriate
            """
            status_text = "🟢 **Status**: Benign Finding - Low Risk"

        elif max_idx == 1:  # Malignant
            analysis_text = f"""
### 🔴 **SUSPICIOUS FINDING DETECTED**
**Confidence Level**: {max_confidence:.1f}%

**Clinical Interpretation**:
- Suspicious lesion characteristics identified
- Features suggestive of malignancy
- Requires immediate clinical attention
- Further diagnostic workup essential

**Urgent Actions Required**:
- Immediate radiologist review
- Consider tissue biopsy
- Multidisciplinary team consultation
- Patient counseling and support
            """
            status_text = "🔴 **Status**: Suspicious Finding - Requires Immediate Attention"

        else:  # Normal
            analysis_text = f"""
### ✅ **NORMAL SCAN RESULT**
**Confidence Level**: {max_confidence:.1f}%

**Clinical Interpretation**:
- No significant abnormalities detected
- Breast tissue appears within normal limits
- No suspicious lesions identified
- Negative screening result

**Recommendations**:
- Continue routine screening schedule
- Maintain breast health awareness
- Report any new symptoms promptly
- Follow standard care guidelines
            """
            status_text = "✅ **Status**: Normal Study - No Significant Findings"

        # Only show overlay if not normal
        final_display = overlay if max_idx != 2 else original_image

        return cls_result, final_display, analysis_text, status_text

    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        error_text = f"""
### ❌ **SYSTEM ERROR**
**Error Details**: {str(e)}

**Troubleshooting**:
- Ensure image is a valid ultrasound scan
- Check image format (PNG, JPG, JPEG supported)
- Try uploading a different image
- Contact technical support if issue persists
        """
        return {{"Error": f"Analysis failed: {str(e)}"}}, None, error_text, "🔴 **Status**: System Error"
