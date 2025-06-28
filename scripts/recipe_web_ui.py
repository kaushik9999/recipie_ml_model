import torch
import gradio as gr
import os
import re
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Recipe formatting function
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
    raise ValueError("Could not find model in any expected location. Please specify a valid model path.")

# Load model and tokenizer
print(f"Loading model from {model_path}...")
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Using device: {device}")

def generate_recipe(recipe_name, temperature=0.7, max_length=512, detailed=True, creative=False):
    """Generate a recipe based on user inputs."""
    # Create prompt based on detail level
    if detailed:
        input_text = f"Generate detailed ingredients and step-by-step directions for: {recipe_name}"
    else:
        input_text = f"Generate recipe for: {recipe_name}"
    
    # Adjust temperature based on creativity
    if creative:
        actual_temp = 0.9
    else:
        actual_temp = temperature
    
    # Tokenize input
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    
    # Generate with parameters
    outputs = model.generate(
        input_ids,
        max_length=max_length,
        num_beams=5,
        do_sample=True,
        temperature=actual_temp,
        top_p=0.9,
        no_repeat_ngram_size=3,
        early_stopping=True
    )
    
    # Decode and format
    recipe = tokenizer.decode(outputs[0], skip_special_tokens=True)
    formatted_recipe = format_recipe(recipe)
    
    # Add generation info
    gen_info = f"\n\n---\n*Generated with temperature: {actual_temp}, detailed: {detailed}, creative: {creative}*"
    
    return formatted_recipe + gen_info

# Create Gradio interface
with gr.Blocks(title="Recipe Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🍳 Recipe Generator")
    gr.Markdown("Enter a dish name and get a generated recipe!")
    
    with gr.Row():
        with gr.Column(scale=3):
            recipe_input = gr.Textbox(
                label="Recipe Name",
                placeholder="Enter a dish name (e.g., Chocolate Chip Cookies)",
                lines=1
            )
            
            with gr.Row():
                temp_slider = gr.Slider(
                    minimum=0.1, 
                    maximum=1.0, 
                    value=0.6, 
                    step=0.1, 
                    label="Temperature"
                )
                
                max_length = gr.Slider(
                    minimum=256, 
                    maximum=768, 
                    value=512, 
                    step=32, 
                    label="Max Length"
                )
            
            with gr.Row():
                detailed = gr.Checkbox(label="Detailed Instructions", value=True)
                creative = gr.Checkbox(label="Creative Mode", value=False)
            
            generate_btn = gr.Button("Generate Recipe", variant="primary")
        
        with gr.Column(scale=5):
            recipe_output = gr.Markdown(label="Generated Recipe")
    
    # Connect components
    generate_btn.click(
        generate_recipe,
        inputs=[recipe_input, temp_slider, max_length, detailed, creative],
        outputs=recipe_output
    )
    
    # Add examples
    gr.Examples(
        examples=[
            ["Chocolate Chip Cookies"],
            ["Chicken Curry"],
            ["Banana Bread"],
            ["Vegetable Lasagna"],
            ["Strawberry Cheesecake"]
        ],
        inputs=recipe_input
    )
    
    gr.Markdown("## How to use")
    gr.Markdown("""
    - Enter a recipe name in the text box
    - Adjust the temperature slider (higher = more creative, lower = more focused)
    - Click "Generate Recipe"
    - Toggle "Detailed Instructions" for more comprehensive steps
    - Enable "Creative Mode" for more unique recipes
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True) 