# Test the trained model
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the trained model
model_path = "models/flan_t5_base"  # Or your specified output path
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Move to GPU for faster inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Test with example inputs
test_inputs = [
    "Generate ingredients and directions for: Chocolate Cake",
    "Generate ingredients and directions for: Chicken Curry",
    "Generate ingredients and directions for: Vegetable Soup"
]

for test_input in test_inputs:
    print("\n" + "="*50)
    print(f"Input: {test_input}")
    print("="*50)
    
    # Generate output
    inputs = tokenizer(test_input, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs.input_ids, 
        max_length=512,
        num_beams=4,
        temperature=0.7
    )
    
    # Decode and print
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Output:\n{generated_text}")
    print("="*50)