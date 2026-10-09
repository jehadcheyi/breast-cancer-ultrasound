---
title: Breast Cancer Ultrasound V2
emoji: 📉
colorFrom: pink
colorTo: gray
sdk: gradio
sdk_version: 5.37.0
app_file: app.py
pinned: false
---

# Breast Ultrasound Diagnostic Assistant

**Live demo:** [huggingface.co/spaces/jehadcheyi/bc-ultrasound](https://huggingface.co/spaces/jehadcheyi/bc-ultrasound)

A web application for detecting and classifying breast lesions in ultrasound images. A U-Net locates the suspicious region, and a CNN classifies the image as benign, malignant or normal.

I built it as part of my work on medical imaging AI, with the aim of making a research model easy for clinicians and students to try without installing anything.

> **Research use only.** This tool is not a medical device and does not replace expert clinical evaluation.

## What it does

Given an ultrasound image, the app:

1. Segments the suspicious lesion with a U-Net.
2. Classifies the image with a CNN and reports a confidence score for each class: benign, malignant and normal.
3. Returns an overlay with the lesion area highlighted in green, so the result can be checked visually.

The interface also applies optional enhancement filters (contrast, sharpness and brightness) to make lesion boundaries easier to see.

## Models

| Task           | Architecture | Weights      | Input size |
|----------------|--------------|--------------|------------|
| Segmentation   | U-Net        | `unet.h5`    | 224 × 224  |
| Classification | Custom CNN   | `cnn96.h5`   | 400 × 400  |

Both models are loaded from the Hugging Face Hub. The classes follow the BUSI (Breast Ultrasound Images) dataset.

## Using the app

1. Upload an ultrasound image, or choose one from the example gallery.
2. Click **Analyze Image**.
3. Read the class probabilities and inspect the lesion overlay.

## Running it locally

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies are TensorFlow, Gradio, OpenCV, Pillow, NumPy and `huggingface_hub`.

The model weights are fetched from the Hub, so the app needs a Hugging Face access token. You can create one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and expose it as an environment variable named `jehad`, which is the name `app.py` reads. Then start the app with `python app.py` and open the local address that Gradio prints.

## Project structure

```
.
├── app.py             # Gradio app and inference code
├── example/           # Sample ultrasound images
├── requirements.txt   # Python dependencies
└── README.md
```

## Customization

- **Models:** replace the `.h5` files with your own trained weights and update the loading code in `app.py`.
- **Input sizes:** currently 224 × 224 for the U-Net and 400 × 400 for the CNN.
- **Enhancement:** the filters (for example `fuzzy_enhancement` and `sharpness_enhancement`) can be changed or removed in `app.py`.

## Troubleshooting

**The models fail to load.** Check that your Hugging Face token is valid and that it has access to the model repository.

## Limitations

- Trained and evaluated on BUSI, a relatively small public dataset; external validation on images from other scanners and centres is still needed.
- Image quality, probe settings and operator technique can all affect results.
- The overlay shows where the model found a lesion, not a clinical delineation.

## License

MIT.

## Acknowledgments

The BUSI dataset, the Hugging Face Hub, and the TensorFlow and Gradio open-source communities.

## Author

Jehad Cheyi, Duhok, Kurdistan Region of Iraq.
[Hugging Face profile](https://huggingface.co/jehadcheyi)
