"""
5_transformers_nlp_master.py
Comprehensive Hugging Face Transformers Pipeline with Trainer & TrainingArguments
"""

import torch
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification,
    AutoModelForCausalLM, AutoModelForQuestionAnswering,
    Trainer, TrainingArguments, DefaultDataCollator, DataCollatorForTokenClassification,
    DataCollatorForLanguageModeling, pipeline,
    EarlyStoppingCallback, TrainerCallback
)
from transformers.trainer_utils import EvalPrediction
from datasets import Dataset, load_dataset, load_metric
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import train_test_split
import evaluate
import warnings
warnings.filterwarnings('ignore')

class TransformersNLPipeline:
    """Comprehensive Hugging Face Transformers pipeline for all NLP tasks"""
    
    def __init__(self, model_name="bert-base-uncased", task="classification", num_labels=2):
        self.model_name = model_name
        self.task = task
        self.num_labels = num_labels
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
    def load_model_and_tokenizer(self):
        """Load appropriate model and tokenizer for the task"""
        print(f"Loading model and tokenizer for {self.task}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            if self.task == "classification":
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name, num_labels=self.num_labels
                )
            elif self.task == "token_classification":
                self.model = AutoModelForTokenClassification.from_pretrained(
                    self.model_name, num_labels=self.num_labels
                )
            elif self.task == "question_answering":
                self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
            elif self.task == "text_generation":
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            else:
                raise ValueError(f"Unsupported task: {self.task}")
                
            print(f"Successfully loaded {self.model_name} for {self.task}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to distilbert for faster testing
            self.model_name = "distilbert-base-uncased"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=self.num_labels
            )
    
    def preprocess_data(self, texts, labels=None, max_length=128):
        """Preprocess data for transformer models"""
        print("Preprocessing data...")
        
        # Tokenize texts
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Create dataset
        if labels is not None:
            encodings['labels'] = labels
            
        return encodings
    
    def create_dataset(self, texts, labels=None, max_length=128):
        """Create Hugging Face dataset"""
        encodings = self.preprocess_data(texts, labels, max_length)
        return Dataset.from_dict(encodings)
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        if self.task == "classification":
            return self._compute_classification_metrics(eval_pred)
        elif self.task == "token_classification":
            return self._compute_token_classification_metrics(eval_pred)
        else:
            return {}
    
    def _compute_classification_metrics(self, eval_pred):
        """Compute metrics for classification tasks"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        acc = accuracy_score(labels, predictions)
        
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def _compute_token_classification_metrics(self, eval_pred):
        """Compute metrics for token classification"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        # Remove padding tokens (label = -100)
        true_predictions = [
            [p for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [l for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        # Flatten
        true_predictions_flat = [p for sublist in true_predictions for p in sublist]
        true_labels_flat = [l for sublist in true_labels for l in sublist]
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels_flat, true_predictions_flat, average='weighted'
        )
        acc = accuracy_score(true_labels_flat, true_predictions_flat)
        
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

class AdvancedTrainingConfig:
    """Advanced training configuration with multiple presets"""
    
    @staticmethod
    def get_preset(preset_name, output_dir="./results"):
        """Get training arguments preset"""
        common_args = {
            "output_dir": output_dir,
            "save_strategy": "epoch",
            "evaluation_strategy": "epoch",
            "learning_rate": 2e-5,
            "per_device_train_batch_size": 16,
            "per_device_eval_batch_size": 16,
            "num_train_epochs": 3,
            "weight_decay": 0.01,
            "logging_dir": "./logs",
            "logging_steps": 50,
            "report_to": "none"
        }
        
        presets = {
            "fast": {
                **common_args,
                "num_train_epochs": 1,
                "per_device_train_batch_size": 8,
                "learning_rate": 5e-5,
                "max_steps": 1000
            },
            "standard": {
                **common_args,
                "num_train_epochs": 3,
                "per_device_train_batch_size": 16,
                "learning_rate": 2e-5,
                "warmup_steps": 500,
                "logging_steps": 100
            },
            "advanced": {
                **common_args,
                "num_train_epochs": 5,
                "per_device_train_batch_size": 8,
                "gradient_accumulation_steps": 2,
                "learning_rate": 1e-5,
                "warmup_ratio": 0.1,
                "lr_scheduler_type": "cosine",
                "weight_decay": 0.01,
                "fp16": True,
                "load_best_model_at_end": True,
                "metric_for_best_model": "f1",
                "greater_is_better": True
            },
            "production": {
                **common_args,
                "num_train_epochs": 10,
                "per_device_train_batch_size": 32,
                "learning_rate": 3e-5,
                "warmup_steps": 1000,
                "lr_scheduler_type": "linear",
                "save_total_limit": 3,
                "load_best_model_at_end": True,
                "metric_for_best_model": "accuracy",
                "greater_is_better": True,
                "fp16": True,
                "dataloader_num_workers": 4,
                "group_by_length": True
            }
        }
        
        return TrainingArguments(**presets.get(preset_name, presets["standard"]))

class CustomTrainer(Trainer):
    """Custom trainer with additional functionality"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom loss computation if needed
        """
        labels = inputs.get("labels")
        outputs = model(**inputs)
        loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss

class TransformersTextClassification:
    """Complete text classification pipeline"""
    
    def __init__(self, model_name="distilbert-base-uncased", num_labels=2):
        self.pipeline = TransformersNLPipeline(model_name, "classification", num_labels)
        self.pipeline.load_model_and_tokenizer()
        
    def train(self, train_texts, train_labels, val_texts=None, val_labels=None, 
              preset="standard", **kwargs):
        """Train the classification model"""
        
        # Create datasets
        train_dataset = self.pipeline.create_dataset(train_texts, train_labels)
        
        if val_texts is not None and val_labels is not None:
            val_dataset = self.pipeline.create_dataset(val_texts, val_labels)
        else:
            # Split training data for validation
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                train_texts, train_labels, test_size=0.2, random_state=42
            )
            train_dataset = self.pipeline.create_dataset(train_texts, train_labels)
            val_dataset = self.pipeline.create_dataset(val_texts, val_labels)
        
        # Get training arguments
        training_args = AdvancedTrainingConfig.get_preset(preset)
        
        # Update with any custom arguments
        for key, value in kwargs.items():
            if hasattr(training_args, key):
                setattr(training_args, key, value)
        
        # Create trainer
        self.trainer = CustomTrainer(
            model=self.pipeline.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.pipeline.compute_metrics,
            tokenizer=self.pipeline.tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train model
        print("Starting training...")
        train_result = self.trainer.train()
        
        # Save model
        self.trainer.save_model()
        self.pipeline.tokenizer.save_pretrained(training_args.output_dir)
        
        return train_result
    
    def predict(self, texts):
        """Make predictions on new texts"""
        if not hasattr(self, 'trainer'):
            raise ValueError("Model must be trained first!")
        
        # Create dataset
        dataset = self.pipeline.create_dataset(texts)
        
        # Get predictions
        predictions = self.trainer.predict(dataset)
        pred_labels = np.argmax(predictions.predictions, axis=1)
        
        return pred_labels, predictions.predictions
    
    def evaluate(self, test_texts, test_labels):
        """Evaluate model on test data"""
        test_dataset = self.pipeline.create_dataset(test_texts, test_labels)
        results = self.trainer.evaluate(test_dataset)
        return results

class TransformersNamedEntityRecognition:
    """Named Entity Recognition pipeline"""
    
    def __init__(self, model_name="dslim/bert-base-NER", num_labels=9):
        self.pipeline = TransformersNLPipeline(model_name, "token_classification", num_labels)
        self.pipeline.load_model_and_tokenizer()
        
    def train(self, tokens_list, labels_list, val_tokens=None, val_labels=None, preset="standard"):
        """Train NER model"""
        
        def tokenize_and_align_labels(examples):
            tokenized_inputs = self.pipeline.tokenizer(
                examples["tokens"],
                truncation=True,
                is_split_into_words=True,
                padding=True,
                max_length=128
            )
            
            labels = []
            for i, label in enumerate(examples["labels"]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(-100)
                    previous_word_idx = word_idx
                
                labels.append(label_ids)
            
            tokenized_inputs["labels"] = labels
            return tokenized_inputs
        
        # Create datasets
        train_data = {"tokens": tokens_list, "labels": labels_list}
        train_dataset = Dataset.from_dict(train_data)
        train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
        
        if val_tokens is not None and val_labels is not None:
            val_data = {"tokens": val_tokens, "labels": val_labels}
            val_dataset = Dataset.from_dict(val_data)
            val_dataset = val_dataset.map(tokenize_and_align_labels, batched=True)
        else:
            # Split for validation
            split_idx = int(0.8 * len(tokens_list))
            train_dataset = Dataset.from_dict({
                "tokens": tokens_list[:split_idx],
                "labels": labels_list[:split_idx]
            }).map(tokenize_and_align_labels, batched=True)
            
            val_dataset = Dataset.from_dict({
                "tokens": tokens_list[split_idx:],
                "labels": labels_list[split_idx:]
            }).map(tokenize_and_align_labels, batched=True)
        
        # Training arguments
        training_args = AdvancedTrainingConfig.get_preset(preset)
        
        # Data collator
        data_collator = DataCollatorForTokenClassification(tokenizer=self.pipeline.tokenizer)
        
        # Create trainer
        self.trainer = Trainer(
            model=self.pipeline.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=self.pipeline.compute_metrics,
            tokenizer=self.pipeline.tokenizer,
        )
        
        # Train
        self.trainer.train()
        return self.trainer
    
    def predict_entities(self, text):
        """Predict named entities in text"""
        nlp_pipeline = pipeline(
            "ner",
            model=self.pipeline.model,
            tokenizer=self.pipeline.tokenizer,
            aggregation_strategy="simple"
        )
        return nlp_pipeline(text)

class TransformersTextGeneration:
    """Text generation pipeline"""
    
    def __init__(self, model_name="gpt2"):
        self.pipeline = TransformersNLPipeline(model_name, "text_generation")
        self.pipeline.load_model_and_tokenizer()
        
    def train(self, texts, preset="standard"):
        """Train language model (causal LM)"""
        
        def tokenize_function(examples):
            return self.pipeline.tokenizer(
                examples["text"],
                truncation=True,
                max_length=512,
                padding=False
            )
        
        # Create dataset
        dataset = Dataset.from_dict({"text": texts})
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Training arguments
        training_args = AdvancedTrainingConfig.get_preset(preset)
        
        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.pipeline.tokenizer,
            mlm=False  # Causal LM, not masked LM
        )
        
        # Create trainer
        self.trainer = Trainer(
            model=self.pipeline.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            tokenizer=self.pipeline.tokenizer,
        )
        
        # Train
        self.trainer.train()
        return self.trainer
    
    def generate_text(self, prompt, max_length=100, num_return_sequences=1, temperature=0.7):
        """Generate text from prompt"""
        inputs = self.pipeline.tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.pipeline.model.generate(
                inputs.input_ids,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.pipeline.tokenizer.eos_token_id
            )
        
        generated_texts = []
        for output in outputs:
            text = self.pipeline.tokenizer.decode(output, skip_special_tokens=True)
            generated_texts.append(text)
        
        return generated_texts

class MultiTaskTransformersPipeline:
    """Multi-task pipeline for different NLP tasks"""
    
    def __init__(self):
        self.classification_pipeline = None
        self.ner_pipeline = None
        self.generation_pipeline = None
        
    def run_text_classification(self, texts, labels, model_name="distilbert-base-uncased"):
        """Run complete text classification pipeline"""
        print("Starting Text Classification Pipeline...")
        
        self.classification_pipeline = TransformersTextClassification(model_name, len(set(labels)))
        
        # Train model
        train_result = self.classification_pipeline.train(texts, labels)
        
        # Evaluate
        pred_labels, probabilities = self.classification_pipeline.predict(texts[:10])  # Sample prediction
        
        return {
            "train_result": train_result,
            "sample_predictions": pred_labels,
            "model": self.classification_pipeline
        }
    
    def run_ner(self, tokens_list, labels_list, model_name="dslim/bert-base-NER"):
        """Run complete NER pipeline"""
        print("Starting NER Pipeline...")
        
        self.ner_pipeline = TransformersNamedEntityRecognition(model_name)
        
        # Train model
        trainer = self.ner_pipeline.train(tokens_list, labels_list)
        
        # Sample prediction
        sample_text = "My name is John Doe and I live in New York."
        entities = self.ner_pipeline.predict_entities(sample_text)
        
        return {
            "trainer": trainer,
            "sample_entities": entities,
            "model": self.ner_pipeline
        }
    
    def run_text_generation(self, texts, model_name="gpt2"):
        """Run complete text generation pipeline"""
        print("Starting Text Generation Pipeline...")
        
        self.generation_pipeline = TransformersTextGeneration(model_name)
        
        # Train model
        trainer = self.generation_pipeline.train(texts)
        
        # Sample generation
        prompt = "The future of artificial intelligence"
        generated_texts = self.generation_pipeline.generate_text(prompt)
        
        return {
            "trainer": trainer,
            "sample_generation": generated_texts,
            "model": self.generation_pipeline
        }

# Example usage
if __name__ == "__main__":
    # Sample data for demonstration
    sample_texts = [
        "This movie is absolutely fantastic!",
        "Terrible acting and boring plot.",
        "One of the best films I've ever seen.",
        "Waste of time, don't watch it.",
        "Brilliant cinematography and great performances.",
        "Mediocre at best, nothing special.",
        "A masterpiece of modern cinema.",
        "Poor script and weak characters."
    ]
    
    sample_labels = [1, 0, 1, 0, 1, 0, 1, 0]  # 1: positive, 0: negative
    
    # Initialize multi-task pipeline
    nlp_pipeline = MultiTaskTransformersPipeline()
    
    # Run text classification
    print("=" * 60)
    clf_results = nlp_pipeline.run_text_classification(sample_texts, sample_labels)
    print("Text classification completed!")
    
    # For NER (would need proper token-label pairs)
    print("\n" + "=" * 60)
    print("NER pipeline ready (would need proper training data)")
    
    # For text generation (would need larger text corpus)
    print("\n" + "=" * 60)
    print("Text generation pipeline ready (would need larger corpus)")
    
    print("\nAll Transformers pipelines are ready to use!")