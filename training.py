import get_data
from accelerate import Accelerator
import os

# TODO: this while shit, load in difussors with train_data not dataset_name

accelerator = Accelerator()
model_name = "stabilityai/stable-diffusion-xl-base-1.0"  # Replace with your model name
# dataset_name = get_data.load_dataset(channel_url="https://www.youtube.com/@Fireship", max_results=5)

# Define your training arguments here
training_args = {
    'pretrained_model_name_or_path': model_name,
    'dataset_name': dataset_name,
    # Add other training arguments as needed
}

# Use Accelerator to launch the training script
accelerator.launch(
    command="train_text_to_image_lora_sdxl.py",
    args=training_args
)
