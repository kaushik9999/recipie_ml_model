import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import sys
import os
import re

def format_recipe(recipe_text):
    """Format and clean up the generated recipe text."""
    # Remove source/website references
    recipe_text = re.sub(r'Source:.*$', '', recipe_text, flags=re.MULTILINE)
    recipe_text = re.sub(r'Website:.*$', '', recipe_text, flags=re.MULTILINE)
    
    # Try to split ingredients and directions if they exist
    if "Ingredients:" in recipe_text and "Directions:" in recipe_text:
        parts = recipe_text.split("Directions:")
        ingredients = parts[0].strip()
        directions = "Directions:" + parts[1].strip()
        
        # Format ingredients
        ingredients = ingredients.replace("Ingredients:", "**Ingredients:**\n")
        ingredients = re.sub(r'•\s*', '• ', ingredients)
        
        # Format directions
        directions = directions.replace("Directions:", "\n**Directions:**\n")
        
        # Combine
        recipe_text = ingredients + directions
    
    # Clean up bullet points
    recipe_text = recipe_text.replace("• ", "\n• ")
    
    # Remove duplicate newlines
    recipe_text = re.sub(r'\n+', '\n', recipe_text)
    
    return recipe_text

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
        print("Please specify a path: python simple_test_improved.py 'Recipe Name' /path/to/model")
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
        
        # Create more structured prompt
        input_text = f"Generate detailed ingredients and step-by-step directions for: {recipe_name}"
        print(f"Input: {input_text}")
        
        # Tokenize and generate
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        
        # Generate with improved parameters
        outputs = model.generate(
            input_ids,
            max_length=512,
            num_beams=5,           # Increased from 4
            do_sample=True,        # Enable sampling
            temperature=0.6,       # Slightly lower temperature for more focus
            top_p=0.9,             # Add nucleus sampling
            no_repeat_ngram_size=3,  # Increased from 2
            early_stopping=True,
            num_return_sequences=1  # Generate just one sequence
        )
        
        # Decode and format result
        recipe = tokenizer.decode(outputs[0], skip_special_tokens=True)
        formatted_recipe = format_recipe(recipe)
        
        print("\n" + "="*50)
        print(f"RECIPE FOR: {recipe_name}")
        print("="*50)
        print(formatted_recipe)
        print("="*50)
        
        # Show generation parameters for reference
        print("\nGeneration Parameters:")
        print("- Beams: 5")
        print("- Temperature: 0.6")
        print("- Do Sample: True")
        print("- Top-p: 0.9")
        print("- No Repeat Ngram Size: 3")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 