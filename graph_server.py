#!/usr/bin/env python
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
from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks.manager import CallbackManager
import pytesseract
from dotenv import load_dotenv
import json
import re
from ollama import chat
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

deep_seek_client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"),base_url="https://api.deepseek.com")

tracer = LangChainTracer()
callback_manager = CallbackManager(handlers=[tracer])
app = Flask(__name__)
model_reasoning = None

# ---------------------------
# LLM Wrapper and Model Loading
# ---------------------------
class LLMResponse:
    def __init__(self, content):
        self.content = content

class LLMWrapper:
    def __init__(self, model_name, api_key):
        self.model_name = model_name

    def invoke(self, prompt, system_prompt=None):
        messages = []
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [
                {"role": "user", "content": prompt},
            ]
        
        if(self.model_name =="deepseek-reasoner"):
            completion = deep_seek_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_completion_tokens=200,
            )
            generated_text = completion.choices[0].message.content

            print("Generated Deepseek=", generated_text)
            reasoning_content = completion.choices[0].message.reasoning_content
            print("Deepseek reasoning = ",reasoning_content)
            return LLMResponse(content=generated_text)
        
        else:

            completion = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_completion_tokens=200,
            )
            generated_text = completion.choices[0].message.content
            print("Generated GPT=", generated_text)
            return LLMResponse(content=generated_text)

llm = LLMWrapper("gpt-4o", api_key="YOUR_API_KEY_HERE")


deepseek = LLMWrapper("deepseek-reasoner",api_key="YOUR_API_KEY_HERE")

# Load Llava models
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
model.to("cuda:0")

# ---------------------------
# Decision Making Function
# ---------------------------
def select_viewpoint(curr_img_desc, viewpoints_dict, goal, history, graph_info,current_VP):
    """
    Build a minimal prompt using the current image description, available viewpoint info,
    goal, history, and navigation graph.
    """
    descriptions = "\n".join([
        f"Viewpoint ID: {v_id}, Description: {desc.get('unified_description', '')}"
        for v_id, desc in viewpoints_dict.items()
    ])
    viewpoint_list = "\n".join(viewpoints_dict.keys())
    system_prompt = """You are an AI assistant tasked with a navigation problem. Your goal is to analyze a given objective and a set of viewpoint descriptions, each corresponding to a unique location. 

    Your task:
    1. Identify the viewpoint description that most closely aligns with the given goal or a relevant subgoal.
    2. Output the corresponding viewpoint ID along with a clear and concise reasoning.

    Output Format:
    You must strictly return your answer in JSON format, following this exact structure:

    ```json
    {
        "reasoning": "<Brief but precise explanation of why this viewpoint best aligns with the goal>",
        "viewpointID": "<Selected Viewpoint ID>"
    }
    ```
    DO NOT include any additional text or formatting outside of this JSON structure."""

    prompt = f""" Navigation Task Goal: {goal}

    You must use all this to give your answer

    Current Location: {current_VP}

    Your Navigation History: {history}

    Available Viewpoints: \n{viewpoint_list}\n

    Navigation Graph: {graph_info}

    Instructions
    1. Carefully analyze the goal, current location, available viewpoints, and the navigation graph.
    2. Use the navigation graph to understand how each viewpoint is connected and interpret their descriptions.
    3. Refer to the history of viewpoints you have navigated to ensure logical and consistent decision-making.
    4. Select the best viewpoint ID that aligns with the goal or its subgoals.
    5. Provide reasoning as to what viewpoint or path makes the most sense compared to the goal.
    6. If you find yourself going to a viewpoint more than twice, consider that maybe the goal is not there and take another fresh route, away from that viewpoint. Use the navigation history to guide you.
    7. Closed Door must not be considered in reasoning, they are not accessible and donot lead anywhere meaningful.
    you are also given your descriptions of the following:
    - your current location
    - the available viewpoints you can navigate to (available viewpoints list, and their description is in the navigation graph)

    Output your answer strictly in the following JSON format:
    ```
    json
    {{
        "reasoning": "<Why this viewpoint best aligns with the goal>",
        "viewpointID": "<Selected Viewpoint ID>"
    }}
    ```
    Ensure the output is valid JSON. """
    print("\n\n------Prompt: ",prompt,"\n---------------\n\n")
    response = deepseek.invoke(prompt, system_prompt).content
    markdown_match = re.search(r"```json\s*({.*?})\s*```", response, re.DOTALL)
    if markdown_match:
        response = markdown_match.group(1).strip()
    try:
        response_data = json.loads(response)
        viewpoint_id = response_data.get("viewpointID", "Unknown")
    except json.JSONDecodeError:
        print("Error: Unable to parse response.")
        viewpoint_id = None
    return viewpoint_id

# ---------------------------
# Llava-based Image Description Generation
# ---------------------------
def Llava_generation(input_text, image):
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
    return result

def getImageDescriptions(images_list, simple):
    if simple:
        img_desc_prompt = """Analyze the provided image and describe the exact location where the agent is standing. Use very simple and clear language, focusing only on key landmarks or surroundings.""" 
    else:
        img_desc_prompt = """Describe the image with a focus on navigational elements such as hallways, doorways, stairs, and open spaces. Pay attention to the positions of these elements and their relationships to each other. Use directional cues like 'left,' 'right,' 'ahead,' or 'behind' to provide spatial context. Include details such as the size of the spaces, distances between objects, and any potential obstacles. Do not include extraneous numbers. Keep the description concise."""
    image_description_list = []
    for image in images_list:
        scene_description = Llava_generation(img_desc_prompt, image)
        image_description_list.append(scene_description)
    return image_description_list

def unify_descriptions(descriptions: list, simple=False):
    descriptions_joined = "\n".join(descriptions)
    if not simple:
        system_prompt = """You are given a list of image descriptions that capture various angles of a scene around a central point. Please analyze these descriptions and generate a single, cohesive summary that combines the information from all angles. Ensure your response is concise and no more than one brief paragraph, highlighting the main landmarks and their spatial relationships."""
        prompt =  f""" Here is the list of image descriptions: {descriptions_joined}

Please generate a unified, brief paragraph describing the scene."""
    else:
        system_prompt = """You are provided with a list of image descriptions capturing a 360-degree view around a central point. Your task is to generate a single short sentence indicating the agent's exact current location using very simple language."""
        prompt =  f""" Here are the image descriptions: {descriptions_joined}

Please generate a single short sentence describing the agent's current location."""
    response = llm.invoke(prompt, system_prompt).content
    return response

def reached_goal(goal: str, descriptions: list):
    system_prompt = """Given a set of image descriptions and the goal description, determine if the goal has been reached. Answer 'yes' or 'no'."""
    prompt = f"Goal: {goal}\n" 
    for i, desc in enumerate(descriptions): 
        prompt += f"Image Description-{i}: {desc}\n" 
    prompt += "Has the goal been reached? (yes/no)"
    response = llm.invoke(prompt, system_prompt).content
    return response.strip().lower() == "yes"

def reached_goal_unified(goal: str, description):
    prompt = f"""Given the scene description below, and the goal provided, determine if the goal has been achieved based on the description. Goal: {goal} Scene Description: {description} Has the goal been reached? (yes/no)"""
    response = llm.invoke(prompt).content
    return response.strip().lower() == "yes"

def update_history(current_history: str, current_img_desc: str):
    if not current_history:
        prompt = f"""Summarize the following scene description in a brief sentence, giving a sense of where the agent is standing currently:

Scene Description: {current_img_desc}""" 
    else:
        prompt = f"""Update the agent's movement history based on the new scene description.

Current History: {current_history}

New Scene Description: {current_img_desc}

Instructions:

Analyze the new scene description and determine the agent's movement.
Append the new location to the current history, maintaining chronological order.
Mention both the previous and new locations, highlighting the transition.
Use very simple and clear language.
Provide the updated history in a single short sentence.
Ensure the updated history includes all prior locations without unnecessary repetition.
Example:

Current History: "The agent is next to a dining table in the kitchen."

New Scene Description: "The agent is now beside a sofa in the living room."

Updated History Output: "The agent moved from sitting next to a dining table in the kitchen to standing beside a sofa in the living room."

Provide the updated history below:"""
    updated_history = llm.invoke(prompt).content.strip()
    return updated_history

def get_goal_decision_single_image(goal: str, image, current_history: str):
    img_desc_prompt =  f""" You are an intelligent assistant tasked with determining whether the final room specified in the goal has been reached based on the provided image and the history of visited rooms. Instructions:

Analyze the provided image carefully.
Use the current history to confirm if the preceding rooms mentioned in the goal have been visited.
Ignore minor sub-location details. For example, if the goal is "foot of the bed," interpret it as "the bedroom."
Return a JSON response indicating if the goal has been achieved. Example:
```
json
{{
    "goal_achieved": "Yes"
}}
```
Now, evaluate the following: Goal: {goal} Current History: {current_history} Response:"""
    response = Llava_generation(img_desc_prompt, image)
    print("\nIsGoal LLava response:", response)
    try:
        markdown_match = re.search(r"```json\s*({.*?})\s*```", response, re.DOTALL)
        if markdown_match:
            response = markdown_match.group(1).strip()
        data = json.loads(response)
        goal_achieved = data["goal_achieved"].strip()
        print("\nGoal achieved:", goal_achieved)
        return goal_achieved == "Yes"
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        raise ValueError(f"Failed to parse or validate response: {e}")

# ---------------------------
# Flask Endpoints
# ---------------------------

@app.route('/detect_cycle', methods=['POST'])
def detect_cycle():
    try:
        data = request.json
        current_vp = data.get("current_viewpoint", "")
        chosen_vp = data.get("chosen_viewpoint", "")
        history = data.get("history", "")
        graph_info = data.get("graph_info", "")
        goal = data.get("goal", "")

        system_prompt = """You are an AI navigation assistant analyzing a pathfinding agent.
        The agent is stuck in a cycle, repeatedly visiting the same viewpoint. Your task:
        1. Explain why this cycle is occurring based on the history and navigation graph.
        2. Suggest a different viewpoint (if possible) to break the cycle.
        
        Output format:
        ```json
        {
            "reasoning": "<Brief explanation of why the cycle is happening>",
            "alternative_viewpoint": "<Viewpoint ID if an alternative exists, otherwise return the same chosen viewpoint>"
        }
        ```
        """

        prompt = f"""Navigation Cycle Analysis:
        
        - Current Viewpoint: {current_vp}
        - Chosen Viewpoint (Loop Detected): {chosen_vp}
        - Navigation History: {history}
        - Goal: {goal}
        - Navigation Graph: {graph_info}

        Why is this loop occurring? Can we suggest a better viewpoint?
        """

        response = llm.invoke(prompt, system_prompt).content
        markdown_match = re.search(r"```json\s*({.*?})\s*```", response, re.DOTALL)
        if markdown_match:
            response = markdown_match.group(1).strip()

        response_data = json.loads(response)

        return jsonify(response_data)

    except Exception as e:
        print(f"Error in cycle detection: {e}")
        return jsonify({"error": str(e)})
@app.route('/save_images', methods=['POST'])
def save_images():
    try:
        data = request.json  # Expecting JSON data
        images_list = data.get('images', [])  # List of images (base64 or arrays)
        viewpoint_id = str(data.get('viewpoint_id', 'default'))  # Ensure string format
        
        # Validate inputs
        if not images_list:
            return jsonify({'error': 'No images provided'}), 400
        
        # Define the directory
        save_dir = os.path.join("output_images", viewpoint_id)
        os.makedirs(save_dir, exist_ok=True)  # Create directory if not exists
        
        for idx, img_array in enumerate(images_list):
            # Convert list to NumPy array if necessary
            img = np.array(img_array, dtype=np.uint8)
            image_path = os.path.join(save_dir, f"image_{idx+1}.png")
            cv2.imwrite(image_path, img)
        
        return jsonify({'message': f'Saved {len(images_list)} images in {save_dir}'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/make_decision', methods=['POST'])
def make_decision():
    try:
        data = request.json
        nav_locs = data.get('viewpoint_descs', {})
        current_img_desc = data.get('current_img_desc', '')
        goal = data.get('goal', '')
        history = data.get('history','')
        graph_info = data.get('graph_info', '')
        current_VP = data.get('current_viewpoint',"")
      
        if len(nav_locs) > 1:
            v_id = select_viewpoint(current_img_desc, nav_locs, goal,history,graph_info,current_VP)
        else:
            v_id = list(nav_locs.keys())[0]
     
        return jsonify({"v_id": v_id})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)})

@app.route('/get_desc', methods=['POST'])
def get_unified_description():
    try:
        data = request.json
        images_data = data.get('arrays', [])
        simple = data.get('simple', False)

        images_list = []
        for idx, img_str in enumerate(images_data):
            img_data = base64.b64decode(img_str)
            img_array = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            images_list.append(image)

        descriptions = getImageDescriptions(images_list, simple)
        unified_description = unify_descriptions(descriptions, simple)
        return jsonify({"unified_description": unified_description})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)})

@app.route('/is_goal', methods=['POST'])
def get_goal_decision():
    try:
        data = request.json
        Goal = data.get('goal', "")
        images_data = data.get('pano_images', [])
        current_history = data.get('current_history', "")

        pos_count = 0
        for idx, img_str in enumerate(images_data):
            img_data = base64.b64decode(img_str)
            img_array = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            decision = get_goal_decision_single_image(Goal, image, current_history)
            if decision:
                pos_count += 1

        if pos_count >= 2:
            return jsonify({"decision": 1})
        return jsonify({"decision": 0})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)})

@app.route('/get_history', methods=['POST'])
def get_updated_history():
    try:
        data = request.json
        current_img_desc = data.get('current_img_desc', "")
        current_history = data.get('current_history', "")
        updated_history = update_history(current_history, current_img_desc)
        print("\nupdated_history:", updated_history)
        return jsonify({"updated_history": updated_history})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)})

@app.route('/summarize_desc', methods=['POST'])
def summarize_desc():
    try:
        print("\nReceived request at /summarize_desc")  # Check if request is reaching the endpoint

        data = request.json
        # print("Received Data:", data)  # Debug: Print request payload

        viewpoint_id = data.get('viewpoint_id', '')
        description = data.get('description', '')

        if not viewpoint_id or not description:
            print("Error: Missing viewpoint_id or description")  # Debug: Check for missing data
            return jsonify({"error": "Missing viewpoint_id or description"}), 400

        # Debug: Print before making LLM call
        # print(f"Summarizing viewpoint {viewpoint_id} with description: {description}")

        # LLM Prompt
        system_prompt = """You are given a description of a viewpoint from a navigation graph.
        Your task is to summarize it into 1-2 lines while keeping the most relevant details.
        
        - Focus on key landmarks, objects, and spatial positioning.
        - Avoid excessive details but retain navigational significance.
        - The output should be clear, concise, and in plain text.
        """

        prompt = f"Original Description:\n{description}\n\nSummarized Description:"
        response = llm.invoke(prompt, system_prompt).content.strip()

        # print(f"LLM Response: {response}")  # Debug: Print LLM output

        return jsonify({"summarized_description": response})

    except Exception as e:
        print(f"Error in summarize_desc: {e}")  # Debug: Catch unexpected errors
        return jsonify({"error": str(e)})


# ---------------------------
# Main Entry Point
# ---------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)

# print("Registered Routes:")
# for rule in app.url_map.iter_rules():
#     print(rule)
