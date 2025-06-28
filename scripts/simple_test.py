import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import sys
import os

def main():
    # Get recipe name from command line or use default
    recipe_name = sys.argv[1] if len(sys.argv) > 1 else "Chocolate Chip Cookies"
    
    # Paths to try for finding the model
    model_paths = [
        "/content/drive/MyDrive/models/flan_t5_base",
        "/content/drive/MyDrive/recipe_model_checkpoints/after_chunk_5",
        "./models/flan_t5_base",
        os.path.abspath("./models/flan_t5_base")
    ]
    
    # Find a valid model path
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            print(f"Using model at: {path}")
            break
    
    if model_path is None:
        print("Could not find model in any expected location.")
        print("Please specify a path: python simple_test.py 'Recipe Name' /path/to/model")
        return
    
    # If a path was provided as the second argument, use it
    if len(sys.argv) > 2:
        model_path = sys.argv[2]
        if not os.path.exists(model_path):
            print(f"Error: Provided model path {model_path} does not exist")
            return
    
    try:
        # Load model and tokenizer
        print(f"Loading model from {model_path}...")
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        
        # Use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print(f"Using device: {device}")
        
        # Prepare input
        input_text = f"Generate recipe for: {recipe_name}"
        print(f"Input: {input_text}")
        
        # Tokenize and generate
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        
        # Generate with reasonable parameters
        outputs = model.generate(
            input_ids,
            max_length=512,
            num_beams=4,
            temperature=0.7,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
        
        # Decode and print result
        recipe = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print("\n" + "="*50)
        print(f"RECIPE FOR: {recipe_name}")
        print("="*50)
        print(recipe)
        print("="*50)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 