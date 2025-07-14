import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import gradio as gr

# Load Models
model_paths = {
    'classification': 'cnn96.h5',
    'segmentation': 'unet.h5'
}

img_size = (224, 224)  # Classification input size
seg_size = (224, 224)  # Segmentation input size
target_size = (500, 500)  # Display size

classes = {
    'classification': ['benign', 'malignant','normal'],
    'classification_names': {
    
        0: 'Benign',
        1: 'Malignant',
        2: 'Normal',
    }
}

# Verify and load models
print("Checking model files...")
for name, path in model_paths.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    print(f"Found {name} model at: {path}")

print("Loading models...")
try:
    models = {
        'classification': load_model(model_paths['classification']),
        'segmentation': load_model(model_paths['segmentation'])
    }
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    raise

def prep_img(img, model_type='classification'):
    """Prepare image for model input"""
    img = Image.fromarray(img)
    if model_type == 'classification':
        img = img.resize(img_size)
    else:  # segmentation
        img = img.resize(seg_size)
    arr = img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def predict(img):
    try:
        # Classification
        arr = prep_img(img, 'classification')
        class_pred = models['classification'].predict(arr, verbose=0)
        pred_idx = np.argmax(class_pred[0])
        class_conf = 100 * np.max(class_pred[0])
        label = classes['classification'][pred_idx]
        label_name = classes['classification_names'][pred_idx]
        
        # Get probabilities for all classes
        class_probs = {}
        for i, prob in enumerate(class_pred[0]):
            class_probs[classes['classification_names'][i]] = prob*100

        # Segmentation
        seg_arr = prep_img(img, 'segmentation')
        seg_mask = models['segmentation'].predict(seg_arr, verbose=0)[0]
        seg_mask = (seg_mask > 0.5).astype(np.uint8)  # Threshold to binary mask
        
        return pred_idx, class_conf, label_name, class_probs, seg_mask
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise

def apply_segmentation(img, mask):
    """Apply segmentation mask to image"""
    # Resize mask to match original image size
    img = Image.fromarray(img)
    original_size = img.size
    mask = Image.fromarray((mask * 255).astype(np.uint8))
    mask = mask.resize(original_size, Image.NEAREST)
    mask = np.array(mask)
    
    # Create colored overlay (red for segmentation)
    overlay = np.zeros_like(img)
    overlay[mask == 1] = [255, 0, 0]  # Red color for segmented areas
    
    # Blend with original image
    blended = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
    return blended, mask

def create_visualization(img, pred_idx, class_conf, label_name, seg_mask):
    """Create visualization with classification and segmentation results"""
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Apply segmentation
    blended_img, mask = apply_segmentation(img, seg_mask)
    
    # Calculate segmented area percentage
    total_pixels = mask.size
    lesion_pixels = np.sum(mask)
    lesion_percentage = (lesion_pixels / total_pixels) * 100
    
    # Convert images back to RGB for display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    blended_rgb = cv2.cvtColor(blended_img, cv2.COLOR_BGR2RGB)
    mask_rgb = cv2.cvtColor(mask*255, cv2.COLOR_GRAY2RGB)
    
    # Create figure
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    ax[0].imshow(img_rgb)
    ax[0].set_title("Original Ultrasound Image")
    ax[0].axis('off')
    
    # Segmentation mask
    ax[1].imshow(mask_rgb, cmap='gray')
    ax[1].set_title(f"Lesion Segmentation Mask\n({lesion_percentage:.1f}% of image)")
    ax[1].axis('off')
    
    # Overlay
    ax[2].imshow(blended_rgb)
    ax[2].set_title("Segmentation Overlay\n(Red areas indicate lesions)")
    ax[2].axis('off')
    
    # Main title with diagnosis
    diagnosis_color = "#4CAF50" if label_name.lower() != 'malignant' else "#F44336"
    plt.suptitle(
        f"BREAST ULTRASOUND ANALYSIS\nDiagnosis: {label_name} ({class_conf:.1f}% confidence)",
        fontsize=16,
        y=1.05,
        color=diagnosis_color,
        fontweight='bold'
    )
    
    plt.tight_layout()
    return fig

def analyze_image(input_img):
    try:
        img = input_img.astype('uint8')
        pred_idx, class_conf, label_name, class_probs, seg_mask = predict(img)
        
        diagnosis = label_name.upper()
        diagnosis_color = "#4CAF50" if diagnosis != "MALIGNANT" else "#F44336"
        
        # Create table with class probabilities
        prob_table = """
        <table style='width:100%; border-collapse: collapse; margin: 15px 0; font-family: Arial, sans-serif;'>
            <tr style='background-color: #f2f2f2;'>
                <th style='color: #0D47A1; padding: 8px; text-align: left;'>Class</th>
                <th style='color: #0D47A1; padding: 8px; text-align: left;'>Probability</th>
            </tr>
        """
        for name, prob in class_probs.items():
            prob_table += f"""
            <tr>
                <td style='padding: 8px;'>{name}</td>
                <td style='padding: 8px; font-weight: bold;'>{prob:.1f}%</td>
            </tr>
            """
        prob_table += "</table>"
        
        # Diagnosis summary box
        diagnosis_html = f"""
        <div style="border: 2px solid {diagnosis_color}; border-radius: 8px; padding: 20px; margin: 20px 0; font-family: Arial, sans-serif;">
            <h2 style="color: {diagnosis_color}; margin: 0;">Diagnosis: {diagnosis}</h2>
            <div style="background-color: {diagnosis_color}; color: white; padding: 5px 15px; border-radius: 20px; display: inline-block; margin: 10px 0;">
                {class_conf:.1f}% Confidence
            </div>
            <div style="margin-top: 15px;">
                <h3>Class Probabilities:</h3>
                {prob_table}
            </div>
            <p style="margin-top: 15px; font-style: italic;">
                Note: This analysis is for research purposes only. Please consult a medical professional for clinical diagnosis.
            </p>
        </div>
        """
        
        # Create visualization
        fig = create_visualization(img, pred_idx, class_conf, label_name, seg_mask)
        
        return diagnosis_html, fig

    except Exception as e:
        error_html = f"""
        <div style="border: 2px solid #FF5722; border-radius: 8px; padding: 20px; margin: 20px 0; background-color: #fff3e0;">
            <h2 style="color: #FF5722; margin: 0;">Error Processing Image</h2>
            <p>{str(e)}</p>
        </div>
        """
        return error_html, None

def get_sample_images():
    sample_images = []
    sample_folder = "example"
    if os.path.exists(sample_folder):
        for file in os.listdir(sample_folder):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    img_path = os.path.join(sample_folder, file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        sample_images.append((img, file))
                except Exception as e:
                    print(f"Error loading sample image {file}: {e}")
    return sample_images

sample_images = get_sample_images()

# Gradio UI
with gr.Blocks(title="Breast Ultrasound Analysis") as demo:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Upload Breast Ultrasound Image")
            image_input = gr.Image(type="numpy")
            submit_btn = gr.Button("Analyze Image", variant="primary")
            
            if sample_images:
                gr.Markdown("## Example Images")
                gallery_items = [(img, name) for img, name in sample_images]
                gallery = gr.Gallery(
                    value=gallery_items,
                    label="Click on an example image to load it",
                    columns=3,
                    height="auto",
                    object_fit="contain",
                    show_label=True
                )
                
                def select_sample_image(evt: gr.SelectData):
                    return sample_images[evt.index][0]
                
                gallery.select(select_sample_image, None, image_input)
            else:
                gr.Markdown("No example images found in the 'example' folder")
        
        with gr.Column(scale=2):
            diagnosis_output = gr.HTML(
                value="<div style='text-align: center; padding: 40px;'>Analysis results will appear here after processing an image.</div>"
            )
            plot_output = gr.Plot()
    
    submit_btn.click(
        fn=analyze_image,
        inputs=image_input,
        outputs=[diagnosis_output, plot_output]
    )

demo.launch(share=True)