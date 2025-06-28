import os
import yaml
import pandas as pd
import torch
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

# Disable wandb
os.environ["WANDB_DISABLED"] = "true"

# Check GPU
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()

# Load params
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

# Set paths
TRAIN_DATA_PATH = "./data/processed/train.csv"
TEST_DATA_PATH = "./data/processed/test.csv"
MODEL_OUTPUT_DIR = params["model"]["output_dir"]
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# Load data in chunks to save memory
print("Loading data in chunks...")
CHUNK_SIZE = 10000  # Adjust based on your RAM

# Read only the first chunk for training
train_df = pd.read_csv(TRAIN_DATA_PATH, nrows=CHUNK_SIZE)
test_df = pd.read_csv(TEST_DATA_PATH, nrows=CHUNK_SIZE//10)

print(f"Using {len(train_df)} training examples and {len(test_df)} test examples")

# Create datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Load tokenizer
tokenizer = T5Tokenizer.from_pretrained(params["model"]["name"])

# Tokenize with reduced max lengths
def tokenize_data(example):
    # Reduced lengths to save memory
    input_max_length = 64  # Smaller than default
    target_max_length = 256  # Smaller than default
    
    inputs = tokenizer(
        example['input_text'],
        truncation=True,
        padding='max_length',
        max_length=input_max_length
    )
    
    with tokenizer.as_target_tokenizer():
        targets = tokenizer(
            example['target_text'],
            truncation=True,
            padding='max_length',
            max_length=target_max_length
        )
    
    inputs['labels'] = targets['input_ids']
    return inputs

# Process in small batches
print("Tokenizing data (small batches to prevent memory issues)...")
train_tokenized = train_dataset.map(
    tokenize_data, 
    batched=True,
    batch_size=16,  # Small batch size
    remove_columns=train_dataset.column_names  # Remove original columns to save RAM
)

test_tokenized = test_dataset.map(
    tokenize_data, 
    batched=True,
    batch_size=16,
    remove_columns=test_dataset.column_names
)

print("Tokenization complete!")

# Load model
print("Loading model...")
model = T5ForConditionalGeneration.from_pretrained(params["model"]["name"])

# Move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Model moved to {device}")

# Training arguments - REMOVED fp16 and half-precision
training_args = TrainingArguments(
    output_dir=MODEL_OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=200,
    weight_decay=0.01,
    logging_dir=os.path.join(MODEL_OUTPUT_DIR, 'logs'),
    logging_steps=50,
    report_to="none",  # Disable wandb reporting
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    # Removed fp16=True to avoid the error
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
)

# Train
print("Starting training...")
trainer.train()

# Save model
model.save_pretrained(MODEL_OUTPUT_DIR)
tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
print("Training complete!")