import os
import argparse
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

def generate_recipe(model, tokenizer, recipe_name, max_length=512, num_beams=4, 
                   no_repeat_ngram_size=3, temperature=0.7):
    """
    Generate a recipe using the trained model
    
    Args:
        model: The trained T5 model
        tokenizer: The T5 tokenizer
        recipe_name: The name of the recipe to generate
        max_length: Maximum length of the generated recipe
        num_beams: Number of beams for beam search
        no_repeat_ngram_size: Size of n-grams to avoid repeating
        temperature: Sampling temperature
        
    Returns:
        Generated recipe text
    """
    # Format input for T5
    input_text = f"Generate recipe for: {recipe_name}"
    
    # Tokenize input
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    
    # Move to same device as model
    input_ids = input_ids.to(model.device)
    
    # Generate output
    outputs = model.generate(
        input_ids,
        max_length=max_length,
        num_beams=num_beams,
        no_repeat_ngram_size=no_repeat_ngram_size,
        early_stopping=True,
        temperature=temperature
    )
    
    # Decode and return
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser(description="Generate recipes using the trained model")
    parser.add_argument("--model_path", type=str, default="models/flan_t5_base", 
                        help="Path to the trained model directory")
    parser.add_argument("--recipe", type=str, required=True, 
                        help="Recipe name to generate (e.g., 'Chocolate Chip Cookies')")
    parser.add_argument("--max_length", type=int, default=512, 
                        help="Maximum length of generated recipe")
    parser.add_argument("--temperature", type=float, default=0.7, 
                        help="Sampling temperature (higher = more creative, lower = more focused)")
    parser.add_argument("--num_beams", type=int, default=4, 
                        help="Number of beams for beam search")
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        print("Please check the path or ensure training has completed")
        return
    
    # Load model and tokenizer
    print(f"Loading model from {args.model_path}...")
    tokenizer = T5Tokenizer.from_pretrained(args.model_path)
    model = T5ForConditionalGeneration.from_pretrained(args.model_path)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Generate recipe
    print(f"\nGenerating recipe for: {args.recipe}")
    print("-" * 50)
    
    recipe = generate_recipe(
        model, 
        tokenizer, 
        args.recipe,
        max_length=args.max_length,
        num_beams=args.num_beams,
        temperature=args.temperature
    )
    
    print(recipe)
    print("-" * 50)

if __name__ == "__main__":
    main() 