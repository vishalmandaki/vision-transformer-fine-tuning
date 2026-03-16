# Vision Transformer (ViT) Fine-Tuning Pipeline 👁️🚀

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/get-started/locally/)

An advanced Computer Vision pipeline for fine-tuning Vision Transformer (ViT) models on custom datasets. This repository provides a robust, production-ready implementation utilizing HuggingFace Transformers and PyTorch, optimized for high-performance training and easy deployment.

## 🌟 Key Features
- **Modern Architecture**: Leverages `google/vit-base-patch16-224-in21k` and other state-of-the-art ViT variants.
- **Efficient Training**: Support for mixed-precision training (FP16/BF16) and gradient accumulation.
- **Automated Preprocessing**: Scalable image augmentation and normalization pipelines.
- **Comprehensive Metrics**: Integrated with `evaluate` for accuracy, F1-score, and confusion matrix analysis.
- **Experiment Tracking**: Built-in support for TensorBoard and Weights & Biases.

## 🛠️ Installation

```bash
git clone https://github.com/dirk-kuijprs/vision-transformer-fine-tuning.git
cd vision-transformer-fine-tuning
pip install -r requirements.txt
```

## 🚀 Usage

Fine-tune a ViT model on the `beans` dataset:

```bash
python train_vit.py \
    --model_name "google/vit-base-patch16-224-in21k" \
    --dataset_name "beans" \
    --output_dir "./vit-beans-finetuned" \
    --batch_size 32 \
    --epochs 10 \
    --lr 5e-5
```

## 📈 Architecture Overview
The pipeline utilizes the Vision Transformer (ViT) architecture, which treats images as sequences of patches. By applying self-attention mechanisms directly to these patches, the model captures global dependencies more effectively than traditional Convolutional Neural Networks (CNNs).

## 👨‍💻 Author
**Dirk Kuijprs**  
Data Scientist at G42

Special thanks to **Muhammad Ajmal Siddiqui** for his mentorship and guidance. Connect with him on [LinkedIn](https://www.linkedin.com/in/muhammadajmalsiddiqi/).

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
