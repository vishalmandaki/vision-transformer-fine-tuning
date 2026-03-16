# Vision Transformer Fine-Tuning 👁️🦾

Advanced Computer Vision pipeline for fine-tuning state-of-the-art Vision Transformer (ViT) models on custom image classification datasets.

## 🌟 Features
- **Hugging Face Integration**: Seamless loading of pre-trained ViT architectures.
- **FP16 Mixed Precision**: Optimized for high-performance GPU training with reduced memory footprint.
- **Automated Evaluation**: Real-time metrics tracking during the fine-tuning process.

## 🛠️ Installation
```bash
git clone https://github.com/vishalmandaki/vision-transformer-fine-tuning.git
cd vision-transformer-fine-tuning
pip install -r requirements.txt
```

## 🚀 Usage
```python
from src.vit_finetuner import VisionTransformerFinetuner
finetuner = VisionTransformerFinetuner()
finetuner.train(output_dir="./custom-vit-model")
```
