import sys
import os
import torch
import pandas as pd
import yaml
from transformers import T5Tokenizer, T5ForConditionalGeneration
from metrics import compute_metrics

def main():
    # Load parameters
    try:
        with open("params.yaml", "r") as f:
            params = yaml.safe_load(f)
    except FileNotFoundError:
        print("Warning: params.yaml not found. Using default values where possible.")
        params = {}

    # Model and test data paths
    MODEL_DIR = params["model"]["output_dir"]
    TEST_DATA_PATH = params["data"]["test_file"]

    # Check if model directory exists
    if not os.path.exists(MODEL_DIR):
        print(f"Error: Model directory not found at {MODEL_DIR}")
        sys.exit(1)

    # Check if test data exists
    if not os.path.exists(TEST_DATA_PATH):
        print(f"Error: Test data file not found at {TEST_DATA_PATH}")
        sys.exit(1)

    # Load model and tokenizer
    print(f"Loading model and tokenizer from {MODEL_DIR}...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    # Load test data
    test_df = pd.read_csv(TEST_DATA_PATH)
    if not {'input_text', 'target_text'}.issubset(test_df.columns):
        print("Test data must contain 'input_text' and 'target_text' columns.")
        sys.exit(1)

    # How many samples to evaluate
    eval_limit = params.get('evaluate', {}).get('eval_limit', 100)
    eval_limit = min(eval_limit, len(test_df))
    print(f"Evaluating on {eval_limit} test samples...")

    predictions = []
    references = []

    # Improved generation parameters
    gen_kwargs = dict(
        max_length=params.get("training", {}).get("target_max_length", 512),
        num_beams=5,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

    for i, row in test_df.iloc[:eval_limit].iterrows():
        input_text = row['input_text']
        reference = row['target_text']
        inputs = tokenizer(input_text, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model.generate(inputs.input_ids, **gen_kwargs)
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(pred)
        references.append(reference)
        if i < 3:
            print(f"\n--- Example {i+1} ---")
            print(f"Input: {input_text}")
            print(f"Predicted: {pred[:300]}...")
            print(f"Reference: {reference[:300]}...")

    # Compute metrics
    print("\nComputing BLEU and ROUGE metrics...")
    metrics = compute_metrics(predictions, references)
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()