import gradio as gr
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig, GenerationConfig
import torch
from PIL import Image

# Load the processor
processor = AutoProcessor.from_pretrained(
    'allenai/MolmoE-1B-0924',
    trust_remote_code=True
)

# Configure 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

# Load the model with 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    'allenai/MolmoE-1B-0924',
    trust_remote_code=True,
    device_map='auto',
    quantization_config=quantization_config,
    torch_dtype=torch.float16
)

# Create a GenerationConfig object for greedy search
generation_config = GenerationConfig(
    max_new_tokens=200,
    num_beams=1,  # Use 1 for greedy search
    do_sample=False,
    eos_token_id=processor.tokenizer.eos_token_id,
    pad_token_id=processor.tokenizer.pad_token_id,
)

# Define the prediction function
def molmo_inference(image, text_prompt):
    # Process the image and text
    inputs = processor.process(images=[image], text=text_prompt)
    # Move inputs to the correct device
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
    # Generate output with 4-bit quantized model using greedy search
    with torch.no_grad():  # Disable gradient calculation for inference
        output = model.generate(
            inputs['input_ids'],
            generation_config=generation_config
        )
    # Decode the generated tokens to text
    generated_text = processor.tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Create the Gradio interface
interface = gr.Interface(
    fn=molmo_inference,
    inputs=[
        gr.Image(type="pil"),  # Input for image
        gr.Textbox(lines=2, placeholder="Enter your text prompt here...")  # Input for text
    ],
    outputs="text",  # Output will be the generated text
    title="Molmo AI - 4-bit Quantized Inference with Greedy Search",
    description="Upload an image and provide a text prompt for description using 4-bit quantization and greedy search."
)

# Launch the interface
interface.launch()