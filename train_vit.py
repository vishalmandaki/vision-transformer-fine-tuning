import os
import torch
import argparse
import logging
from datasets import load_dataset
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer,
    DefaultDataCollator
)
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor
)
import numpy as np
import evaluate

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Vision Transformer (ViT) on custom datasets.")
    parser.add_argument("--model_name", type=str, default="google/vit-base-patch16-224-in21k", help="Pre-trained ViT model from HuggingFace.")
    parser.add_argument("--dataset_name", type=str, default="beans", help="Dataset to fine-tune on (from HF or local).")
    parser.add_argument("--output_dir", type=str, default="./vit-fine-tuned", help="Output directory for the fine-tuned model.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load dataset
    dataset = load_dataset(args.dataset_name)
    labels = dataset["train"].features["labels"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # Image processing
    image_processor = ViTImageProcessor.from_pretrained(args.model_name)
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    size = (image_processor.size["height"], image_processor.size["width"])

    _train_transforms = Compose([
        RandomResizedCrop(size),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ])

    _val_transforms = Compose([
        Resize(size),
        CenterCrop(size),
        ToTensor(),
        normalize,
    ])

    def train_transforms(examples):
        examples["pixel_values"] = [_train_transforms(img.convert("RGB")) for img in examples["image"]]
        del examples["image"]
        return examples

    def val_transforms(examples):
        examples["pixel_values"] = [_val_transforms(img.convert("RGB")) for img in examples["image"]]
        del examples["image"]
        return examples

    dataset["train"].set_transform(train_transforms)
    dataset["validation"].set_transform(val_transforms)

    # Load model
    model = ViTForImageClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    ).to(device)

    # Metrics
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DefaultDataCollator(),
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
    )

    # Train
    logger.info("Starting fine-tuning...")
    trainer.train()
    
    # Save model
    logger.info(f"Saving fine-tuned model to {args.output_dir}")
    trainer.save_model()
    image_processor.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
