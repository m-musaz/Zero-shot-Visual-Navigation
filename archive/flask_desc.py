import logging
from flask import Flask, request, jsonify
import subprocess
import sys
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import io
import numpy as np
import cv2
import base64
import os
import shutil
from langchain_groq import ChatGroq


# Set up logging
logging.basicConfig(filename='llm_log.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
model.to("cuda:0")


# Function for generating image descriptions
def Llava_generation(input_text, image):
    logging.info(f"Llava_generation called with input_text: {input_text}")
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": input_text},
                
            ]
        }
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(image, prompt, return_tensors="pt").to(0)

    output = model.generate(**inputs, max_new_tokens=1000)
    result = processor.decode(output[0], skip_special_tokens=True).split("[/INST]")[1].strip()
    
    logging.info(f"Llava_generation output: {result}")
    return result


llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


def get_image_from_desc(Descriptions, Images, Goal):
    logging.info(f"get_image_from_desc called with Goal: {Goal}")
    prompt_for_hlp = """
        You are a step by step navigator. You are given a list of descriptions you must chose which image contains a path that best matches the description.
        OUTPUT ONLY THE NUMBER OF THE IMAGE THAT BEST MATCHES THE DESCRIPTION.
        YOUR INPUT MUST BE AN INTEGER ONLY DONOT INCLUDE ANY TEXT UNDER ANY CIRCUMSTANCES
    """
    Description_prompts = [f"Image Description-{i}: {desc}" for i, desc in enumerate(Descriptions)]
    prompt_for_hlp += "\n".join(Description_prompts) + f"\nGoal: {Goal}"

    Chosen_Image_num = llm.invoke(prompt_for_hlp).content
    logging.info(f"get_image_from_desc LLM output: {Chosen_Image_num}")

    Chosen_Image_num = int(Chosen_Image_num)
    return Images[Chosen_Image_num], Chosen_Image_num


def get_nav_point(Image, Goal):
    prompt = """
     You are an image describer. You must describe images to help with navigation. The given image contains numbers. 
     These numbers can be used for navigation. You need to describe the scene right next to each number.
     Lets say there are 2 numbers you must format like
     nav-1: scene description about 1
     nav-2: scene description about 2
     Note: Donot give descriptions for non-existent numbers, and include descriptions for all present numbers.
    """

    desc = Llava_generation(prompt, Image)
    logging.info(f"get_nav_point Llava_generation output: {desc}")

    nav_prompt = f"""You are a navigator. You are given a text with navigation numbers.
    These represent the points which you can navigate to in the image. You must pick a navigation number description 
    that best aligns with the goal description.
    Goal: {Goal}
    {desc}
    RESPOND WITH ONLY A NUMBER THAT CAN BE CONVERTED TO AN INTEGER.
    """
    nav_point = llm.invoke(nav_prompt).content
    logging.info(f"get_nav_point LLM output: {nav_point}")

    nav_point = int(nav_point)
    return nav_point


# Function for generating descriptions for a list of images
def getImageDescriptions(images_list):
    img_desc_prompt = """Describe the image with a focus on navigational elements such as hallways, doorways, stairs, and open spaces.
                Pay attention to the positions of these elements and their relationships to each other.
                Use directional cues like 'left,' 'right,' 'ahead,' or 'behind' to provide spatial context. 
                Include details such as the size of the spaces, distances between objects, and any potential obstacles.
                Donot pay any attention to the numbers in the image and donot include them in the description."""

    image_description_list = []

    for image in images_list:
        scene_description = Llava_generation(img_desc_prompt, image)
        image_description_list.append(scene_description)
        logging.info(f"Generated description for image: {scene_description}")

    return image_description_list


def reached_goal(goal: str, descriptions: list):
    logging.info(f"reached_goal called with Goal: {goal} and Descriptions: {descriptions}")
    prompt = f"Given a set of image descriptions and the goal description, determine if the goal has been reached. Answer 'yes' or 'no'.\nGoal: {goal}\n"
    for i, desc in enumerate(descriptions):
        prompt += f"Image Description-{i}: {desc}"
    prompt += "\nHas the goal been reached? (yes/no)"

    response = llm.invoke(prompt).content
    logging.info(f"reached_goal LLM output: {response}")

    return response.strip().lower() == "yes"


@app.route('/llava', methods=['POST'])
def process_images():
    logging.info("Received POST request")
    data = request.json
    images_data = data.get('arrays', [])
    logging.info(f"Received {len(images_data)} images")

    images_list = []
    for idx, img_str in enumerate(images_data):
        img_data = base64.b64decode(img_str)
        img_array = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        images_list.append(image)

    folder_name = "images"
    os.makedirs(folder_name, exist_ok=True)
    for idx, img in enumerate(images_list):
        cv2.imwrite(f"{folder_name}/image_{idx}.jpg", img)

    Goal = "Walk to the foot of the bed and to a little sitting area.  Go through the door on the left side.  Stop once you have stepped into the bathroom."
    descriptions = getImageDescriptions(images_list)

    if reached_goal(Goal, descriptions):
        logging.info("Goal Reached")
        return jsonify({"image_num": -1, "nav_point": -1})

    logging.info("Goal not reached")
    img, img_num = get_image_from_desc(descriptions, images_list, Goal)
    nav_point = get_nav_point(img, Goal)

    return jsonify({"image_num": img_num, "nav_point": nav_point})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
