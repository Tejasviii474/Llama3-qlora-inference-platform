import os
from typing import Dict, Any, Tuple
from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd

class FinanceDatasetLoader:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dataset_name = config.get("name", "gbharti/finance-alpaca")
        self.text_column = config.get("text_column", "text")
        self.train_split = config.get("train_split", "train")
        self.val_size = config.get("val_size", 0.1)

    def format_llama3_instruct(self, example: Dict[str, Any]) -> Dict[str, str]:
        """
        Formats a single example using the Meta Llama 3 Instruct prompt template.
        Assumes the dataset has 'instruction', 'input', and 'output' (or similar) fields.
        """
        # Default alpaca structure
        instruction = example.get('instruction', '')
        input_text = example.get('input', '')
        output_text = example.get('output', '')
        
        system_prompt = "You are a highly knowledgeable financial AI assistant. Your goal is to provide accurate, clear, and helpful answers to financial and economic questions."
        
        # Build the Llama 3 prompt structure
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
        
        user_content = instruction
        if input_text and input_text.strip() != "":
            user_content += f"\n\nContext:\n{input_text}"
            
        prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{user_content}<|eot_id|>"
        prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{output_text}<|eot_id|>"
        
        return {self.text_column: prompt}

    def load_and_prepare(self) -> DatasetDict:
        """
        Loads the dataset from Hugging Face hub, applies the formatting, and splits into train/val.
        """
        print(f"Loading dataset: {self.dataset_name}")
        raw_dataset = load_dataset(self.dataset_name)
        
        # Use the specified training split, or default to train
        if self.train_split not in raw_dataset:
            raise ValueError(f"Split {self.train_split} not found in dataset {self.dataset_name}")
            
        dataset = raw_dataset[self.train_split]
        
        # Apply formatting
        print("Formatting dataset for Llama 3 Instruct...")
        formatted_dataset = dataset.map(
            self.format_llama3_instruct,
            remove_columns=dataset.column_names, # Remove original columns to keep it clean
            desc="Formatting prompts"
        )
        
        # Create train/val split
        print(f"Creating train/val split (val_size={self.val_size})...")
        split_dataset = formatted_dataset.train_test_split(test_size=self.val_size, seed=42)
        
        dataset_dict = DatasetDict({
            'train': split_dataset['train'],
            'validation': split_dataset['test']
        })
        
        print(f"Dataset prepared! Train size: {len(dataset_dict['train'])}, Validation size: {len(dataset_dict['validation'])}")
        return dataset_dict

if __name__ == "__main__":
    # Test the loader
    config = {
        "name": "gbharti/finance-alpaca",
        "text_column": "text",
        "train_split": "train",
        "val_size": 0.05
    }
    loader = FinanceDatasetLoader(config)
    dataset = loader.load_and_prepare()
    print("Sample prompt:")
    print(dataset['train'][0]['text'])
