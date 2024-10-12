import gradio as gr
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig, GenerationConfig
import torch
from PIL import Image, ImageDraw
import re
import numpy as np

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

# Function to draw the point based on the artifact
def draw_point(image_array, artifact):
    # Convert the NumPy array to a PIL Image
    image = Image.fromarray(image_array)

    # Ensure no padding (crop to actual content)
    image = image.crop(image.getbbox())

    # Size of the image
    width, height = image.size

    # Parse artifact using regex to extract x and y coordinates
    match = re.search(r'<point x="([\d.]+)" y="([\d.]+)"', artifact)
    if match:
        x = int(float(match.group(1)) / 100 * width)  # Extract x
        y = int(float(match.group(2)) / 100 * height)  # Extract y

        # Draw the point
        draw = ImageDraw.Draw(image)
        radius = 10
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="lightblue", outline="lightblue")

        return image  # Return the modified image
    else:
        return None  # Return None if artifact is invalid

# Create the Gradio interface
def interface_function(image_array, text_prompt):
    # First, crop the input image to remove any padding
    cropped_image = Image.fromarray(image_array).crop(Image.fromarray(image_array).getbbox())

    # Convert back to NumPy for processing
    cropped_image_array = np.array(cropped_image)

    generated_text = molmo_inference(cropped_image_array, text_prompt)
    
    # Extract the artifact from the generated text
    artifact_match = re.search(r'(<point.*?</point>)', generated_text, re.DOTALL)
    
    if artifact_match:
        artifact = artifact_match.group(1)  # Get the matched artifact
        modified_image = draw_point(cropped_image_array, artifact)
        return generated_text, modified_image  # Return text and modified image
    else:
        return generated_text, None  # Return text and no image if no valid artifact found

interface = gr.Interface(
    fn=interface_function,
    inputs=[
        gr.Image(type="numpy"),  # Input for image
        gr.Textbox(lines=2, placeholder="Enter your text prompt here...")  # Input for text
    ],
    outputs=["text", gr.Image(type="pil")],  # Output will be the generated text and modified image
    title="Molmo 7B-O - Image Description",
    description="Upload an image and provide a text prompt for description. The model uses mixed precision for efficient inference."
)

# Launch the interface
interface.launch()
