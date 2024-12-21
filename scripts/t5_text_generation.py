from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the pre-trained T5 model and tokenizer
model_name = "t5-small"  # You can also try 't5-base', 't5-large' for better results
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Function to generate text based on an input prompt
def generate_text(input_text):
    # Tokenize the input text
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    
    # Generate the output text
    output_ids = model.generate(input_ids, max_length=150, num_beams=5, early_stopping=True)
    
    # Decode the generated output back into text
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return output_text

# Example input text
input_text = "Translate English to French: How are you?"

# Generate text based on the input
output_text = generate_text(input_text)

# Print the generated output
print(f"Input: {input_text}")
print(f"Generated: {output_text}")
