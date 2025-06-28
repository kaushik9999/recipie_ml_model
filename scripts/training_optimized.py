import os
import yaml
import pandas as pd
import torch
import glob
import shutil
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

# Configuration
MODEL_NAME = "google/flan-t5-base"
TRAIN_DATA_PATH = "./data/processed/train.csv"
TEST_DATA_PATH = "./data/processed/test.csv"
TOTAL_EXAMPLES = 100000
CHUNK_SIZE = 20000
NUM_EPOCHS = 3
MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 512

# IMPORTANT: Store checkpoints on Google Drive to persist between sessions
LOCAL_CHECKPOINT_DIR = "/content/drive/MyDrive/recipe_model_checkpoints"
FINAL_MODEL_DIR = params["model"]["output_dir"]  # On Google Drive

# Create directories
os.makedirs(LOCAL_CHECKPOINT_DIR, exist_ok=True)
os.makedirs(FINAL_MODEL_DIR, exist_ok=True)

# Mount Google Drive if in Colab
try:
    from google.colab import drive
    print("Mounting Google Drive...")
    drive.mount('/content/drive')
    print("Google Drive mounted successfully")
except ImportError:
    print("Not running in Colab, skipping drive mount")
except:
    print("Error mounting Google Drive. Make sure you're running in Colab and authorize access")
    
# Find latest checkpoint
def find_latest_checkpoint():
    checkpoints = glob.glob(f"{LOCAL_CHECKPOINT_DIR}/checkpoint-*")
    if not checkpoints:
        return None
    
    # Extract checkpoint numbers and find the highest
    checkpoint_nums = [int(cp.split("-")[-1]) for cp in checkpoints]
    if not checkpoint_nums:
        return None
    
    latest_checkpoint = f"{LOCAL_CHECKPOINT_DIR}/checkpoint-{max(checkpoint_nums)}"
    return latest_checkpoint if os.path.exists(latest_checkpoint) else None

# Clean up old checkpoints to save space
def cleanup_checkpoints(keep_latest=2):
    checkpoints = glob.glob(f"{LOCAL_CHECKPOINT_DIR}/checkpoint-*")
    if len(checkpoints) <= keep_latest:
        return
    
    # Sort checkpoints by number
    checkpoint_nums = [(cp, int(cp.split("-")[-1])) for cp in checkpoints]
    checkpoint_nums.sort(key=lambda x: x[1])
    
    # Delete all but the latest ones
    for cp, _ in checkpoint_nums[:-keep_latest]:
        print(f"Removing old checkpoint: {cp}")
        shutil.rmtree(cp)

# Load tokenizer
print(f"Loading tokenizer from {MODEL_NAME}...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

# Tokenization function
def tokenize_data(example):
    inputs = tokenizer(
        example['input_text'],
        truncation=True,
        padding='max_length',
        max_length=MAX_INPUT_LENGTH
    )
    
    with tokenizer.as_target_tokenizer():
        targets = tokenizer(
            example['target_text'],
            truncation=True,
            padding='max_length',
            max_length=MAX_TARGET_LENGTH
        )
    
    inputs['labels'] = targets['input_ids']
    return inputs

# Determine the starting chunk and model
latest_checkpoint = find_latest_checkpoint()
start_chunk = 0
model = None

# Check if we should resume from a checkpoint
if latest_checkpoint:
    print(f"Found checkpoint at {latest_checkpoint}")
    try:
        if os.path.exists(f"{LOCAL_CHECKPOINT_DIR}/current_chunk.txt"):
            with open(f"{LOCAL_CHECKPOINT_DIR}/current_chunk.txt", 'r') as f:
                start_chunk = int(f.read().strip())
                print(f"Resuming from chunk {start_chunk}")
    except:
        print("Could not determine chunk number, starting from beginning")
        start_chunk = 0
    
    # Load model from checkpoint
    print(f"Loading model from checkpoint {latest_checkpoint}")
    model = T5ForConditionalGeneration.from_pretrained(latest_checkpoint)
else:
    # Start fresh
    print(f"Starting fresh training with model {MODEL_NAME}")
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Create training arguments - saving to LOCAL storage
# FIX: Added eval_strategy="steps" to match save_strategy
training_args = TrainingArguments(
    output_dir=LOCAL_CHECKPOINT_DIR,  # LOCAL storage, not Drive
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=200,
    weight_decay=0.01,
    logging_dir=os.path.join(LOCAL_CHECKPOINT_DIR, 'logs'),
    logging_steps=50,
    report_to="none",
    save_strategy="steps",
    eval_strategy="steps",  # ADDED THIS LINE - must match save_strategy
    save_steps=500,
    eval_steps=500,  # ADDED THIS LINE - should match save_steps
    save_total_limit=2,
    learning_rate=5e-5,
    load_best_model_at_end=True,
    metric_for_best_model="loss",  # ADDED THIS LINE
)

# Load test data once
print("Loading test data...")
test_df = pd.read_csv(TEST_DATA_PATH, nrows=CHUNK_SIZE//10)
test_dataset = Dataset.from_pandas(test_df)
test_tokenized = test_dataset.map(
    tokenize_data, 
    batched=True,
    batch_size=16,
    remove_columns=test_dataset.column_names
)
print(f"Test dataset prepared with {len(test_tokenized)} examples")

# Process data in chunks
total_chunks = (TOTAL_EXAMPLES + CHUNK_SIZE - 1) // CHUNK_SIZE
for chunk_idx in range(start_chunk, total_chunks):
    chunk_start = chunk_idx * CHUNK_SIZE
    chunk_end = min((chunk_idx + 1) * CHUNK_SIZE, TOTAL_EXAMPLES)
    
    print(f"\n{'='*50}")
    print(f"PROCESSING CHUNK {chunk_idx+1}/{total_chunks}: examples {chunk_start} to {chunk_end}")
    print(f"{'='*50}")
    
    # Load chunk of training data
    print(f"Loading training data chunk {chunk_idx+1}...")
    
    # Read only the chunk we need
    if chunk_start == 0:
        # First chunk is simple
        train_df = pd.read_csv(TRAIN_DATA_PATH, nrows=CHUNK_SIZE)
    else:
        # Skip to the right position for later chunks
        train_df = pd.read_csv(
            TRAIN_DATA_PATH, 
            skiprows=range(1, chunk_start+1),  # Skip header + previous chunks
            nrows=CHUNK_SIZE,
            header=None,
            names=['input_text', 'target_text']  # Specify column names since we skip header
        )
    
    # Save current chunk number for resuming
    with open(f"{LOCAL_CHECKPOINT_DIR}/current_chunk.txt", 'w') as f:
        f.write(str(chunk_idx))
    
    print(f"Loaded {len(train_df)} examples for this chunk")
    
    # Convert to dataset and tokenize
    train_dataset = Dataset.from_pandas(train_df)
    
    # Force garbage collection before tokenization
    del train_df
    import gc
    gc.collect()
    
    train_tokenized = train_dataset.map(
        tokenize_data, 
        batched=True,
        batch_size=16,
        remove_columns=train_dataset.column_names
    )
    
    # More cleanup
    del train_dataset
    gc.collect()
    
    print(f"Tokenized {len(train_tokenized)} training examples")
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=test_tokenized,
    )
    
    # Resume from checkpoint if needed
    resume_checkpoint = latest_checkpoint if chunk_idx == start_chunk and latest_checkpoint else None
    
    # Train on this chunk
    print(f"Training on chunk {chunk_idx+1}...")
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    
    # Save intermediate model to local storage only
    # (we'll only save the final model to Drive)
    if chunk_idx < total_chunks - 1:
        chunk_model_path = f"{LOCAL_CHECKPOINT_DIR}/after_chunk_{chunk_idx+1}"
        model.save_pretrained(chunk_model_path)
        print(f"Saved intermediate model after chunk {chunk_idx+1}")
        
        # Clean up old checkpoints
        cleanup_checkpoints(keep_latest=1)
    
    # Update latest_checkpoint
    latest_checkpoint = find_latest_checkpoint()
    
    # Clean up to save memory
    del train_tokenized
    gc.collect()
    torch.cuda.empty_cache()
    
    # Delete all but the latest checkpoint to save space
    cleanup_checkpoints(keep_latest=1)

# Save final model to Google Drive
print(f"Training complete! Saving final model to Google Drive at {FINAL_MODEL_DIR}")
model.save_pretrained(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)

# Keep checkpoints for potential resumption, but clean up older ones
print("Cleaning up older checkpoints but keeping the latest...")
cleanup_checkpoints(keep_latest=1)
print(f"Checkpoints are saved at {LOCAL_CHECKPOINT_DIR}")
print("If training was interrupted, you can resume by running this script again.")

print("Training complete!")