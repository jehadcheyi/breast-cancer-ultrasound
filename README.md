# 🧠 Breast Ultrasound Diagnostic Assistant

> An AI-powered web application for the **detection and classification** of breast lesions using **ultrasound imaging**, powered by deep learning (U-Net for segmentation and CNN for classification).

---

## 📸 Overview

The **Breast Ultrasound Diagnostic Assistant** is a medical imaging tool that:

- ✅ Automatically **segments suspicious lesions** in ultrasound scans  
- 🧪 Provides **diagnostic confidence scores** for:
  - **Benign**
  - **Malignant**
  - **Normal** cases
- 🖼️ Returns a **visual overlay** showing highlighted lesion areas in green

Paper:
Advanced CNN-Based Classification and Segmentation for Enhanced Breast Cancer Ultrasound Imaging
J Cheyi, YÇ Kaya - Gazi University Journal of Science Part A: Engineering, 2024

---

## 🚀 Features

- 🔍 **Automatic Lesion Segmentation** using a trained U-Net model
- 🧠 **Classification** using a robust Convolutional Neural Network (CNN)
- 💡 Interactive UI built with **Gradio**
- 🧪 Real-time predictions with confidence scores
- 🌱 Includes enhancement filters to improve lesion visibility (contrast, sharpness, brightness)
- 📁 Supports drag-and-drop and sample image gallery

---

## 📁 Example Output

![Example Interface](https://user-images.githubusercontent.com/your-image-here.png)

---

## 📦 Installation

1. **Clone this repository**

```bash
git clone https://github.com/yourusername/breast-ultrasound-assistant.git
cd breast-ultrasound-assistant
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Set your Hugging Face token

Export your Hugging Face token (for loading models):

bash
Copy
Edit
export jehad=your_huggingface_token_here
💡 You can generate a token from https://huggingface.co/settings/tokens

🧠 Model Information
Model Type	Architecture	Purpose	Source
Segmentation	U-Net	Lesion Highlight	unet.h5
Classification	Custom CNN	Diagnosis	cnn96.h5

🧪 How It Works
Input an ultrasound image

The U-Net model segments the breast region and highlights potential lesions

The CNN model analyzes the segmented region and outputs diagnostic probabilities

Results are displayed with:

Classification labels and confidence

Enhanced lesion visualization overlay

🔧 File Structure
bash
Copy
Edit
├── app.py                     # Main Gradio app script
├── example/                   # Folder containing example ultrasound scans
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
🖥️ Run Locally
bash
Copy
Edit
python app.py
This will open the Gradio interface in your browser.

🧪 Sample Usage
Upload your ultrasound image or click on a sample

Click “Analyze Image”

View classification results and lesion overlay

💻 Dependencies
tensorflow

gradio

opencv-python

Pillow

numpy

huggingface_hub

Install all dependencies via:

bash
Copy
Edit
pip install -r requirements.txt
🛠️ Customization
You can customize:

Input size (currently 224x224 for U-Net, 400x400 for CNN)

Model weights via Hugging Face Hub

Enhancement techniques (fuzzy_enhancement, sharpness_enhancement, etc.)

🙋 FAQ
Q: What if models fail to load?
A: Make sure your Hugging Face token is valid and you have access to the model repository.

Q: Can I use my own models?
A: Yes. Replace the .h5 files with your trained models and update the code accordingly.

🤝 Acknowledgements
BUSI Dataset (Breast Ultrasound Images)

Hugging Face Model Hosting

TensorFlow & Gradio open-source communities

📜 License
This project is licensed under the MIT License.

👤 Author
Developed by Jehad Cheyi
Hugging Face Profile

⭐️ Show Your Support
If you found this project useful or interesting:

🌟 Star the repository

🍴 Fork it

🧠 Use it in your research or classroome
