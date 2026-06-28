import subprocess
import sys

# Check if necessary libraries are installed, install them if not
required_packages = ['transformers', 'torch', 'pillow','bitsandbytes']

# def install(package):
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# for package in required_packages:
#     try:
#         __import__(package)
#     except ImportError:
#         install(package)

# from transformers import BitsAndBytesConfig
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image


# Model configuration
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16,  # Set compute dtype to match input dtype
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
# )

# Load processor and model (will use cache if already downloaded)
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    torch_dtype=torch.float16,  # Ensure input dtype is still float16
    low_cpu_mem_usage=True,
    # load_in_4bit=True,
    # bnb_4bit_compute_dtype=torch.float16,
    # quantization_config=bnb_config,
)

# Function for generating image descriptions
def Llava_generation(input_text, image):
    conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": input_text},
            {"type": "image", "image": image},
        ]
    }
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(prompt, images=[image], return_tensors="pt").to(0)

    output = model.generate(**inputs, max_new_tokens=1000)
    result = processor.decode(output[0], skip_special_tokens=True).split("[/INST]")[1]
    return result.strip()

# Function for generating descriptions for a list of images
def getImageDescriptions(images_list):
    img_desc_prompt = """Describe the image with a focus on navigational elements such as hallways, doorways, stairs, and open spaces.
                Pay attention to the positions of these elements and their relationships to each other.
                Use directional cues like 'left,' 'right,' 'ahead,' or 'behind' to provide spatial context. 
                Include details such as the size of the spaces, distances between objects, and any potential obstacles.
                Prioritize information that would assist in understanding how to move through or navigate the environment depicted in the image."""

    image_description_list = []

    for image in images_list:
        scene_description = Llava_generation(img_desc_prompt, image)
        image_description_list.append(scene_description)

    return image_description_list
