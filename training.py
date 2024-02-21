import get_data
import os
import subprocess
from accelerate.utils import write_basic_config
# from accelerate import Accelerator

write_basic_config()

model_name = "stabilityai/stable-diffusion-xl-base-1.0"  # Replace with your model name
dataset_path = get_data.create_dataset(channel_url="https://www.youtube.com/@Fireship", max_results=5)


training_args = {
    'pretrained_model_name_or_path': model_name,
    'train_data_dir': dataset_path,
    'validation_epochs': 0,
    'num_train_epochs': 10,
    'image_column': 'file_name',
    'output_dir': 'finetunes',
}

# accel = Accelerator(mixed_precision='fp16')

def convert_training_args_to_command_line_args(training_args):
    command_line_args = []
    for key, value in training_args.items():
        if isinstance(value, bool):
            value = 'true' if value else 'false'
        command_line_args.append(f'--{key}={value}')
    return command_line_args

# I FUCKING HATE THIS
command_line_args = convert_training_args_to_command_line_args(training_args)
command = ['accelerate',
            'launch',
            "--mixed_precision=fp16",
            './diffusers/examples/text_to_image/train_text_to_image_lora_sdxl.py', 
        ] + command_line_args

subprocess.run(command)




# from diffusers import StableDiffusionXLPipeline

# # Load the pretrained SDXL model
# pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")

# # Define the text prompt for the image generation
# text_prompt = "A beautiful sunset over the ocean"

# # Set the target size to  16:9 aspect ratio
# target_size = (1920,  1080)  #  16:9 resolution of  1920x1080 pixels

# # Generate the image
# output = pipeline(text_prompt, target_size=target_size)

# # The output.images[0] will contain the generated image
