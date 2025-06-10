# -*- coding: utf-8 -*-
"""Step_3_Model_Building.py

Adapted for running on V100-server2 server with paths matching Step1.py.
"""

import os
import h5py
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM, # Use this instead
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)
# os.environ['TORCH_USE_CUDA_DSA'] = '1'
# Define file paths matching Step1.py
BASE_PATH = os.path.join('btp', 'btp')
ROOT_DIR = "/home/garvit22185/Python-3.8.18/Python-3.8.18"
FULL_BASE_PATH = os.path.join(ROOT_DIR, BASE_PATH)

# Input and output file paths
train_temp_path = os.path.join(FULL_BASE_PATH, "train_temp.txt")
test_temp_path = os.path.join(FULL_BASE_PATH, "test_temp.txt")
data_h5_path = os.path.join(FULL_BASE_PATH, "data_temp.h5")
output_dir = os.path.join(FULL_BASE_PATH, "outputs")

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Select GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

# Define PyTorch-compatible Dataset class
class H5Dataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size=512):
        cached_features_file = data_h5_path
        print(f"Loading features from cached file {cached_features_file}")
        with h5py.File(cached_features_file, 'r') as f:
            if file_path == test_temp_path:
                self.samples = f[file_path][:]  # Test set
            else:
                self.samples = f[file_path][:]  # Train set

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return torch.tensor(self.samples[item])

def get_dataset(tokenizer, evaluate=False, local_rank=-1):
    file_path = test_temp_path if evaluate else train_temp_path
    return H5Dataset(tokenizer=tokenizer, file_path=file_path)

# Perform Transformer Configuration
config = AutoConfig.from_pretrained('gpt2', cache_dir=os.path.join(ROOT_DIR, 'cache'))
set_seed(20)

# Define the Tokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2', cache_dir=os.path.join(ROOT_DIR, 'cache'))

# Initialize the GPT-2 Model
model = AutoModelForCausalLM.from_pretrained('gpt2', config=config, cache_dir=os.path.join(ROOT_DIR, 'cache')) # New
# model = model.to('cuda:1')
print("Model loaded successfully!")

# Add special recipe tokens to the tokenizer
special_tokens = {
    "additional_special_tokens": ['<RECIPE_START>', '<INPUT_START>', '<NEXT_INPUT>', '<INPUT_END>',
                                  '<INGR_START>', '<NEXT_INGR>', '<INGR_END>', '<INSTR_START>',
                                  '<NEXT_INSTR>', '<INSTR_END>', '<TITLE_START>', '<TITLE_END>',
                                  '<RECIPE_END>', '<CF_START>', '<CF_END>']
}
tokenizer.add_special_tokens(special_tokens)

# Resize the model to fit the tokenizer with special tokens
model.resize_token_embeddings(len(tokenizer))

# Convert datasets to PyTorch format
train_dataset = get_dataset(tokenizer=tokenizer)
eval_dataset = get_dataset(tokenizer=tokenizer, evaluate=True)

# Define data collator for batching
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False, mlm_probability=0.15
)

# Define training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    optim="adafactor",
    num_train_epochs=2,
    per_device_train_batch_size=1,  # Keep original for now, reduce if OOM occurs on GPU 0
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=32,
    # evaluation_strategy="steps", # Deprecated
    eval_strategy="steps", # Use this instead
    fp16=True,
    # fp16_opt_level='O1',
    warmup_steps=100,
    learning_rate=5e-4,
    adam_epsilon=1e-8,
    weight_decay=0.01,
    save_total_limit=1,
    load_best_model_at_end=True,
    gradient_checkpointing=True, # Consider adding if OOM persists on GPU 0
    # report_to="none"
)

# Initialize PyTorch Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Save the tokenizer and train the model
print("Saving tokenizer to:", output_dir)
tokenizer.save_pretrained(output_dir)
print("Starting model training...")
trainer.train()
print("Training complete. Saving model to:", output_dir)
trainer.save_model()
tokenizer.save_pretrained(output_dir)