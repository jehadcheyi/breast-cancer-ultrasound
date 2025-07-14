
# Breast Ultrasound Image Classification Using Deep Learning

This repository contains implementations of deep learning models for classifying breast ultrasound images into malignant, benign, and normal categories. It compares several transfer learning architectures and a custom CNN, highlighting the impact of image enhancement and overlay techniques on performance. Grad-CAM visualizations are included for model interpretability.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architectures](#model-architectures)
- [Performance Summary](#performance-summary)
- [Grad-CAM Visualization](#grad-cam-visualization)
- [Results and Comparison](#results-and-comparison)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

---

## Project Overview

Breast cancer diagnosis using ultrasound imaging is a crucial clinical task. This project explores multiple CNN architectures trained on enhanced and raw datasets to classify breast ultrasound images. The enhanced dataset uses image preprocessing techniques such as segmentation, enhancement, and overlaying, which significantly improve model performance. The study also includes model explainability with Grad-CAM heatmaps.

---

## Dataset

- The primary dataset used is the **Breast Ultrasound Images (BUSI)** dataset containing 780 images categorized as benign, malignant, or normal.
- Dataset Link: [BUSI Dataset on Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)

---

## Requirements

- Python 3.8 or higher
- TensorFlow 2.x
- Keras
- OpenCV
- NumPy
- Matplotlib
- scikit-learn

---

## Installation

1. Clone this repository:

```bash
git clone https://github.com/your-username/breast-ultrasound-classification.git
cd breast-ultrasound-classification
```

2. (Optional) Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. Install required Python packages:

```bash
pip install -r requirements.txt
```

---

## Usage

### Train a Model

Run the training script specifying the model and options:

```bash
python train.py --model vgg19 --fine_tune_layers 4 --enhance True
```

- `--model`: Choose one from `[vgg16, vgg19, densenet121, efficientnetb0, mobilenet, inceptionv3, xception, resnet50, customcnn]`
- `--fine_tune_layers`: Number of last layers to fine-tune (default: all layers)
- `--enhance`: Use enhanced and overlayed images (`True` or `False`)

### Evaluate a Model

```bash
python evaluate.py --model_path models/vgg19_best.h5 --test_dir data/test/
```

### Generate Grad-CAM Visualization

```bash
python gradcam.py --model_path models/customcnn_best.h5 --image_path data/samples/malignant_01.jpg
```

---

## Model Architectures

- **Transfer Learning Models:**
  - VGG16
  - VGG19
  - DenseNet121
  - EfficientNetB0
  - MobileNet
  - InceptionV3
  - Xception
  - ResNet50

- **Custom CNN Model:**
  - A lightweight CNN tailored for breast ultrasound image classification, with approximately 3 million parameters, achieving high accuracy and efficiency.

---

## Performance Summary

| Model          | Accuracy (Enhanced Dataset) | Precision | Recall | F1-Score |
|----------------|-----------------------------|-----------|--------|----------|
| EfficientNetB0 | 0.99                        | 0.99      | 0.98   | 0.99     |
| VGG19          | 0.98                        | 0.99      | 0.98   | 0.98     |
| DenseNet121    | 0.98                        | 0.99      | 0.98   | 0.98     |
| MobileNet      | 0.98                        | 0.98      | 0.98   | 0.98     |
| InceptionV3    | 0.98                        | 0.99      | 0.98   | 0.98     |
| Xception       | 0.98                        | 0.99      | 0.98   | 0.98     |
| VGG16          | 0.96                        | 0.98      | 0.95   | 0.96     |
| ResNet50       | 0.91                        | 0.91      | 0.93   | 0.92     |
| **Custom CNN** | 0.97                        | 0.97      | 0.97   | 0.97     |

> Dataset enhancement and preprocessing significantly improved model results.

---

## Grad-CAM Visualization

Grad-CAM heatmaps provide visual explanations by highlighting important regions influencing the model's prediction, aiding clinical interpretability and trust.

Example heatmaps are generated for malignant, benign, and normal cases, overlayed on the original ultrasound images.

---

## Results and Comparison

- The custom CNN achieves competitive accuracy (97%) with fewer parameters compared to most transfer learning models.
- EfficientNetB0 achieved the highest accuracy (99%) on the enhanced dataset.
- Dataset enhancement (segmentation, overlay) leads to marked performance improvements.
- The proposed model outperforms or matches state-of-the-art approaches with better computational efficiency.

---

## Contributing

Contributions, issues, and feature requests are welcome! Please feel free to check the issues page or submit a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## References

Please refer to the published paper:

- J. Cheyi, Y. Çetin-Kaya, *GU J Sci, Part A*, 11(4), 647-667 (2024). DOI: 10.54287/gujsa.1529857

For more background and similar studies, check the references within the paper.

---

*This README was generated to assist users in running and understanding the breast ultrasound classification project.*
