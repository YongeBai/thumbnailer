import get_data
from accelerate import Accelerator
import os

# TODO: this while shit, load in difussors with train_data not dataset_name

accelerator = Accelerator()
model_name = "stabilityai/stable-diffusion-xl-base-1.0"  # Replace with your model name
dataset_path = get_data.load_dataset(channel_url="https://www.youtube.com/@Fireship", max_results=5)

# Define your training arguments here
training_args = {
    'pretrained_model_name_or_path': model_name,
    'train_data_dir': dataset_path,
    'validation_epochs' = 0,
    'num_train_epochs' = 10,
    'mixed_precision': 'fp16',
    'image_column': 'file_name'
    # Add other training arguments as needed
}

# Use Accelerator to launch the training script
accelerator.launch(
    command="train_text_to_image_lora_sdxl.py",
    args=training_args
)


from diffusers import StableDiffusionXLPipeline

# Load the pretrained SDXL model
pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")

# Define the text prompt for the image generation
text_prompt = "A beautiful sunset over the ocean"

# Set the target size to  16:9 aspect ratio
target_size = (1920,  1080)  #  16:9 resolution of  1920x1080 pixels

# Generate the image
output = pipeline(text_prompt, target_size=target_size)

# The output.images[0] will contain the generated image
