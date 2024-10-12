import gradio as gr
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig, GenerationConfig
import torch
from PIL import Image

# Load the processor
processor = AutoProcessor.from_pretrained(
    'allenai/Molmo-7B-O-0924',
    trust_remote_code=True
)

# Configure 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    'allenai/Molmo-7B-O-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto',
    quantization_config=quantization_config,
)

# Define the prediction function
def molmo_inference(image, text_prompt):
    # Process the image and text
    inputs = processor.process(images=[Image.fromarray(image)], text=text_prompt)
    
    # Move inputs to the correct device and make a batch of size 1
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
    
    # Generate output
    with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer
        )
    
    # Only get generated tokens; decode them to text
    generated_tokens = output[0, inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return generated_text

# Create the Gradio interface
interface = gr.Interface(
    fn=molmo_inference,
    inputs=[
        gr.Image(type="numpy"),  # Input for image
        gr.Textbox(lines=2, placeholder="Enter your text prompt here...")  # Input for text
    ],
    outputs="text",  # Output will be the generated text
    title="Molmo 7B - Image Description",
    description="Upload an image and provide a text prompt for description. The model uses mixed precision for efficient inference."
)

# Launch the interface
interface.launch()