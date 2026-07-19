print(f"Top of train_lora.py reached! __name__ is {__name__}")
import os
import yaml
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import wandb

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_prep.dataset_loader import FinanceDatasetLoader

def load_config(config_path="configs/training_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    print("Script started! Loading config...")
    # 1. Load Configuration
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs", "training_config.yaml")
    config = load_config(config_path)
    
    set_seed(config['training']['seed'])

    if config['tracking']['wandb']:
        wandb.init(project=config['tracking']['wandb_project'], name=config['training']['run_name'])

    # 2. Prepare Dataset
    dataset_loader = FinanceDatasetLoader(config['dataset'])
    dataset_dict = dataset_loader.load_and_prepare()

    # 3. Setup Quantization (QLoRA)
    quant_config = config['quantization']
    bnb_config = None
    if quant_config['load_in_4bit']:
        compute_dtype = getattr(torch, quant_config['bnb_4bit_compute_dtype'])
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_config['bnb_4bit_quant_type'],
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=quant_config['bnb_4bit_use_double_quant'],
        )
    elif quant_config['load_in_8bit']:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # 4. Load Model and Tokenizer
    model_name = config['model']['base_model']
    print(f"Loading base model: {model_name}")
    
    # We require a token to be present for Llama 3
    hf_token = os.environ.get("HF_TOKEN")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    # Llama 3 does not have a pad token by default, we use eos_token
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
        trust_remote_code=config['model']['trust_remote_code']
    )
    
    # 5. Prepare for PEFT
    if config['training']['gradient_checkpointing']:
        model.gradient_checkpointing_enable()
        
    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model)

    peft_cfg = config['peft']
    lora_config = LoraConfig(
        r=peft_cfg['lora_r'],
        lora_alpha=peft_cfg['lora_alpha'],
        lora_dropout=peft_cfg['lora_dropout'],
        bias=peft_cfg['bias'],
        task_type=peft_cfg['task_type'],
        target_modules=peft_cfg['target_modules']
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 6. Setup Training Arguments
    t_cfg = config['training']
    training_args = SFTConfig(
        output_dir=t_cfg['output_dir'],
        per_device_train_batch_size=t_cfg['per_device_train_batch_size'],
        per_device_eval_batch_size=t_cfg['per_device_eval_batch_size'],
        gradient_accumulation_steps=t_cfg['gradient_accumulation_steps'],
        learning_rate=float(t_cfg['learning_rate']),
        num_train_epochs=t_cfg['num_train_epochs'],
        weight_decay=t_cfg['weight_decay'],
        warmup_ratio=t_cfg['warmup_ratio'],
        lr_scheduler_type=t_cfg['lr_scheduler_type'],
        optim=t_cfg['optim'],
        fp16=t_cfg['fp16'],
        bf16=t_cfg['bf16'],
        max_grad_norm=t_cfg['max_grad_norm'],
        logging_steps=t_cfg['logging_steps'],
        eval_strategy=t_cfg['evaluation_strategy'],
        eval_steps=t_cfg['eval_steps'],
        save_steps=t_cfg['save_steps'],
        save_total_limit=t_cfg['save_total_limit'],
        report_to="wandb" if config['tracking']['wandb'] else "none",
        run_name=t_cfg['run_name'],
        dataset_text_field=config['dataset']['text_column'],
        max_seq_length=config['dataset']['max_length'],
        packing=False
    )

    # 7. Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset_dict['train'],
        eval_dataset=dataset_dict['validation'],
        peft_config=lora_config,
        tokenizer=tokenizer,
        args=training_args,
    )

    # 8. Train
    print("Starting training...")
    trainer.train()

    # 9. Save Model
    print(f"Saving fine-tuned model to {t_cfg['output_dir']}")
    trainer.model.save_pretrained(t_cfg['output_dir'])
    tokenizer.save_pretrained(t_cfg['output_dir'])
    
    if config['tracking']['wandb']:
        wandb.finish()
        
    print("Training complete!")

if __name__ == "__main__":
    main()
