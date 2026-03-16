import torch
from transformers import ViTForImageClassification, ViTImageProcessor, TrainingArguments, Trainer
from datasets import load_dataset
import numpy as np

class VisionTransformerFinetuner:
    def __init__(self, model_id: str = "google/vit-base-patch16-224"):
        """Advanced CV pipeline for fine-tuning ViT models."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = ViTImageProcessor.from_pretrained(model_id)
        self.model = ViTForImageClassification.from_pretrained(model_id).to(self.device)
        print(f"ViT Finetuner initialized with {model_id} on {self.device}")

    def load_and_preprocess(self, dataset_name: str = "beans"):
        """Load and preprocess dataset for ViT."""
        ds = load_dataset(dataset_name)
        
        def transform(example_batch):
            inputs = self.processor([x for x in example_batch['image']], return_tensors='pt')
            inputs['labels'] = example_batch['labels']
            return inputs

        prepared_ds = ds.with_transform(transform)
        return prepared_ds

    def train(self, output_dir: str = "./vit-finetuned"):
        """Execute the fine-tuning process."""
        dataset = self.load_and_preprocess()
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=8,
            evaluation_strategy="steps",
            num_train_epochs=3,
            fp16=True if self.device == "cuda" else False,
            save_steps=100,
            eval_steps=100,
            logging_steps=10,
            learning_rate=2e-5,
            save_total_limit=2,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to="none",
            load_best_model_at_end=True,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            tokenizer=self.processor,
        )
        
        print("Starting training...")
        trainer.train()
        print(f"Fine-tuning completed. Best model saved to {output_dir}")

if __name__ == "__main__":
    finetuner = VisionTransformerFinetuner()
    # finetuner.train() # Requires GPU and Beans dataset for full execution
    print("ViT Finetuner module ready.")
