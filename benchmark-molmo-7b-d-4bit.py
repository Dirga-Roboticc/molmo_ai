import os
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig, GenerationConfig
import torch
from PIL import Image
import re
import numpy as np
import time

# Load the processor
processor = AutoProcessor.from_pretrained(
    'allenai/Molmo-7B-D-0924',
    trust_remote_code=True
)

# Configure 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    'allenai/Molmo-7B-D-0924',
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

def benchmark_inference(image_array, text_prompt, runs=5):
    # First, crop the input image to remove any padding
    cropped_image = Image.fromarray(image_array).crop(Image.fromarray(image_array).getbbox())
    cropped_image_array = np.array(cropped_image)

    results = []
    times = []

    # Perform 6 runs, but only save results from the last 5
    for _ in range(runs + 1):
        start_time = time.time()
        generated_text = molmo_inference(cropped_image_array, text_prompt)
        elapsed_time = time.time() - start_time
        
        if _ > 0:  # Skip the first inference
            results.append(generated_text)
            times.append(elapsed_time)
            print(f"Inference run: {_}, Time taken: {elapsed_time:.4f} seconds")
        else:
            print(f"Skipped the first inference (run: {_}), Time taken: {elapsed_time:.4f} seconds")

    # Calculate average time taken for the last 5 runs
    avg_time = sum(times) / len(times) if times else 0
    return results, times, avg_time

# Example usage
if __name__ == "__main__":
    # Load all images from the test folder
    test_folder = './test'  # Path to the folder containing images
    text_prompt = "point to search icon on left sidebar of vs code "  # Your text prompt

    # Iterate over all images in the folder
    for filename in os.listdir(test_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Adjust based on your image formats
            image_path = os.path.join(test_folder, filename)
            image_array = np.array(Image.open(image_path))

            print(f"\nProcessing image: {filename}")
            # Run the benchmark
            results, times, avg_time = benchmark_inference(image_array, text_prompt)

            # Print the results
            for i, result in enumerate(results):
                print(f"Result {i + 1} for {filename}:\n{result}\n")

            # Print the average time
            print(f"Average time for {filename}: {avg_time:.4f} seconds")
