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
import os
from dotenv import load_dotenv
load_dotenv()
import json
import re
from ollama import chat
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


#set env varaibles


tracer = LangChainTracer()

# Create a CallbackManager and add the tracer
callback_manager = CallbackManager(handlers=[tracer])
app = Flask(__name__)
model_reasoning = None
# processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
# model = LlavaNextForConditionalGeneration.from_pretrained(
#     "llava-hf/llava-v1.6-mistral-7b-hf",
#     torch_dtype=torch.float16,
#     low_cpu_mem_usage=True,
# )
# Define a wrapper to simulate the `invoke` method
# class LLMResponse:
#     def __init__(self, content):
#         self.content = content

# class LLMWrapper:
#     def __init__(self, pipeline):
#         self.pipeline = pipeline

#     def invoke(self, prompt,system_prompt=None ):
#         messages=[]
#         if(system_prompt):
#             messages = [
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": prompt},
#             ]
#         else:
#             messages = prompt

#         response = self.pipeline(messages, max_new_tokens=200,return_full_text=False)
#         print("LLM response plain",response)
#         print("\n\n\nLLM response",response[0]["generated_text"])
#         return LLMResponse(content=response[0]["generated_text"])  # Return response with .content

class LLMResponse:
    def __init__(self, content):
        self.content = content

class LLMWrapper:
    def __init__(self, model_name, api_key):
        self.model_name = model_name

    def invoke(self, prompt,system_prompt=None):
        messages=[]
        if(system_prompt):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [
                {"role": "user", "content": prompt},
            ]

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_completion_tokens=200,  
        )
        generated_text = completion.choices[0].message.content
        print("Generated GPT=",generated_text)
        return LLMResponse(content=generated_text)
    

def load_phi(api_key=os.getenv("OPENAI_API_KEY")):
    model_name = "gpt-4o-mini"  # Use "gpt-3.5-turbo" for a faster, cheaper alternative if needed
    llm = LLMWrapper(model_name, api_key)
    return llm

llm = load_phi()


# Function to load LLaVA
def load_llava():
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    model.to("cuda:0")  # Load onto GPU
    return processor, model

processor,model = load_llava()
# Function to load PHI-4
# def load_phi():
#     # phi_pipeline = pipeline(
#     #     "text-generation",
#     #     model="microsoft/phi-4",
#     #     torch_dtype=torch.bfloat16,
#     #     device=0  # Use GPU
#     # )

#     model_name = "Satwik11/Microsoft-phi-4-Instruct-AutoRound-GPTQ-4bit"

#     tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         device_map="auto",
#         torch_dtype=torch.float16,
#         trust_remote_code=True
#     )

#     phi_pipeline = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer
#     )

#     llm = LLMWrapper(phi_pipeline)
#     return llm

# Function to unload model
def unload_model(model):
    if model:
        # Handle LLMWrapper for phi_pipeline
        try:
            if isinstance(model, LLMWrapper):
                del model.pipeline  # Delete the pipeline instance
            del model  # Delete the model object
            torch.cuda.empty_cache()  # Clear CUDA cache
        except:
            pass

# llm = ChatGroq(
#     model="llama-3.1-70b-versatile",
#     temperature=0,
#     api_key="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",
#     callback_manager=callback_manager,
    
# )

# Initialize the pipeline for text generation
# model_id = "microsoft/phi-4"
# phi_pipeline = pipeline(
#     "text-generation",
#     model=model_id,
#     torch_dtype=torch.bfloat16,
#     device=0,  # Use RTX 4090 (cuda:0)
# )

# Wrap the pipeline
# llm = LLMWrapper(phi_pipeline)

# model.to("cuda:0")
#idhr json output krana, reasoning wala, phir history
def select_viewpoint(curr_img_desc,viewpoints_dict, goal,history): #TODO: Give example in prompt or output should be in JSON
    # Create the prompt with the dictionary of viewpoint descriptions and goal
    

    descriptions = "\n".join([f"Viewpoint ID: {v_id}, Description: {desc}" for v_id, desc in viewpoints_dict.items()])
    system_prompt = """You are an AI assistant tasked with a navigation problem. Your goal is to analyze a given objective and a set of viewpoint descriptions, each corresponding to a unique location. 

    Your task:
    1. Identify the viewpoint description that most closely aligns with the given goal or a relevant subgoal.
    2. Output the corresponding viewpoint ID along with a clear and concise reasoning.

    Output Format:
    You must strictly return your answer in JSON format, following this exact structure:

    ```json
    {{
        "viewpointID": "<Selected Viewpoint ID>",
        "reasoning": "<Brief but precise explanation of why this viewpoint best aligns with the goal>"
    }}
    ```
    DO NOT include any additional text or formatting outside of this JSON structure. """
    prompt = f"""

    Navigation Task
    Goal: {goal}

    Available Viewpoints: {descriptions}

    Current Location Description: {curr_img_desc}

    Instructions:
    Carefully analyze the goal and the provided viewpoint descriptions.
    Closed Doors must not be considered while assesing navigation, since they are closed permanently.
    Determine which viewpoint ID best matches the goal or its subgoals.
    Justify your selection with a brief, logical reasoning.
    Output your answer strictly in the following JSON format:
    ```json
    {{
        "viewpointID": "<Selected Viewpoint ID>",
        "reasoning": "<Why this viewpoint best aligns with the goal>"
    }}
    ```
    ⚠️ Important:

    Ensure the output is valid JSON (no missing brackets, commas, or syntax errors).
    Return only the JSON response—no extra text, explanations, or formatting. """

    
    # Assuming llm.invoke() is the method to call the language model
    response = llm.invoke(system_prompt=system_prompt,prompt=prompt).content
    
    # Extract and return the viewpoint ID from the response
    # print("Listing all viewpoints IDs and descriptions\n")
    # for v_id, desc in viewpoints_dict.items():
    #     print(f"Viewpoint ID: {v_id} \n")
    #     print(f"Description: {desc} \n\n")
    # print("\n\n")
    markdown_match = re.search(r"```json\s*({.*?})\s*```", response, re.DOTALL)
    
    if markdown_match:
        response = markdown_match.group(1).strip()  # Extract the JSON content

    print(f"Response that should have viewpoint ID: {response} \n")
    # viewpoint_id = response.strip()  # Assuming response directly returns the viewpoint ID
    try:
        # Parse JSON response
        response_data = json.loads(response)

        # Extract viewpointID and reasoning
        viewpoint_id = response_data.get("viewpointID", "Unknown")
        reasoning = response_data.get("reasoning", "No reasoning provided.")

        # Print reasoning
        print(f"Reasoning: {reasoning}")
        model_reasoning = reasoning

        # Return viewpoint_id
    except json.JSONDecodeError:
        print("Error: Unable to parse response.")
        viewpoint_id = None

    return viewpoint_id

# Function for generating image descriptions
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



# Function for generating descriptions for a list of images
def getImageDescriptions(images_list,simple):
    if simple:
        img_desc_prompt = f"""Analyze the provided image and describe the exact location where the agent is standing. Use very simple and clear language, focusing only on key landmarks or surroundings to indicate the precise position.
        *Indoor Scenario:*

        *When provided with an image of a bookshelf in a library:*

        **Output:**
        "The agent is standing next to a bookshelf in the library.
        Your response 
        """
    else:
        img_desc_prompt = f"""Describe the image with a focus on navigational elements such as hallways, doorways, stairs, and open spaces.
                Pay attention to the positions of these elements and their relationships to each other.
                Use directional cues like 'left,' 'right,' 'ahead,' or 'behind' to provide spatial context. 
                Include details such as the size of the spaces, distances between objects, and any potential obstacles.
                Donot pay any attention to the numbers in the image and donot include them in the description.
                Make sure you description is concise and short donot answer in more than one paragraph."""
    
    

    image_description_list = []

    for image in images_list:
        scene_description = Llava_generation(img_desc_prompt, image)
        image_description_list.append(scene_description)
       

    return image_description_list


def unify_descriptions(descriptions: list,simple=False):
    

    print("\nCurrently in unify_descriptions() function and simple= ",simple," \n")
    descriptions_joined = "\n".join(descriptions)
    if simple==False:
        system_prompt = """You are given a list of image descriptions that capture various angles of a scene in a  F0-degree view around a central point. 
        Please analyze these descriptions and generate a single, cohesive summary that combines the information from all angles.
        NOTE: Make sure your response is concise and no more than one brief paragraph.
        
        Consider the following when creating the unified description:
        Highlight any consistent or notable landmarks, objects, and their relative positions.
        Describe the surroundings in a continuous flow that logically follows the 360-degree perspective, mentioning any major shifts in scenery 
        as you move around."""
        prompt = f"""
        Focus on spatial relationships, distances, and directional cues where possible, so that the summary represents a clear overall layout of the scene.
        Here is the list of image descriptions: {descriptions_joined}
        
        Please generate a single brief paragraph wiht a unified scene description that conveys the main elements and spatial layout clearly.
        Make sure your response is concise and no more than one brief paragraph.
        """
    else:
        system_prompt = f"""
        You are provided with a list of image descriptions capturing a 360-degree view around a central point.
        Your task is to analyze these descriptions and generate a single, cohesive summary that clearly indicates the agent's exact current location.

        **Instructions:**
        - Use very simple and clear language.
        - Keep the summary as a single short sentence.
        - Highlight consistent or notable landmarks and their relative positions.
        - Focus on describing the exact location of the agent based on the combined information.
        """

        prompt = f"""
        Here are the image descriptions capturing the 360-degree view:

        {descriptions_joined}

        Please generate a single short sentence that provides a unified description of the agent's exact current location.
        - Use simple and clear language.
        - Ensure the summary is a single short sentence.
        - Focus on the main landmarks and their positions to clearly indicate where the agent is standing.
        """


    response = llm.invoke(system_prompt,prompt).content

    return response


def reached_goal(goal: str, descriptions: list):
    

    print("\nCurrently in reached_goal() function\n")
    system_prompt = """Given a set of image descriptions and the goal description, determine if the goal has been reached. Answer 'yes' or 'no'."""
    prompt = f"Goal: {goal}\n"
    for i, desc in enumerate(descriptions):
        prompt += f"Image Description-{i}: {desc}"
    prompt += "\nHas the goal been reached? (yes/no)"

    response = llm.invoke(system_prompt,prompt).content
   
    return response.strip().lower() == "yes"

def reached_goal_unified(goal: str, description):
    
    print("\nCurrently in reached_goal_unified() function\n")
    prompt = f"""Given the scene description below, and the goal provided, determine if the goal has been achieved based on the description. 
    Provide a clear answer with 'yes' or 'no' based on whether the goal has been reached.
    Goal: {goal}
    Scene Description: {description}
    Has the goal been reached? (yes/no)"""

    response = llm.invoke(prompt).content
   
    return response.strip().lower() == "yes"

# def update_history(current_history: str, current_img_desc: str):
#     
#     print("\nCurrently in update_history() function\n")
#     prompt = f"""You are a summarization assistant. Your task is to create a concise, movement-based summary of a person's actions and movements based on detailed scene descriptions. The summary should focus on key movements and transitions, avoiding unnecessary detail.

# If no previous summary is provided, create a new summary starting from the given scene description. If a previous summary is provided, integrate the new movement into it to update the history.
# **Note:** Always Keep the final summary concise, it should not be longer than one paragaph.
# **Instructions:**
# - Analyze the scene description carefully.
# - Identify the key movements and actions performed by the person.
# - Focus on what changed or what new locations you entered, mentioning your location around furniture like a bed or a dining table for example.
# - Avoid re-describing details/locations that remain unchanged from the previous summary unless essential for clarity.
# - Based on this, provide the final updated summary (or start a new one if no previous summary exists)
# - Always Keep the final updated summary concise, it should not be longer than one paragaph.
# - Ensure the JSON is properly formatted with double quotes and no trailing commas.
# - Do not include any additional information beyond the JSON object.

# **Examples:**
# **Previous Summary(if any):** ""
# **latest scene description:** "The scene is set in a modern interior space with a narrow hallway that connects various rooms. The hallway is characterized by decorative walls, including a textured wall and a wall with a blue geometric pattern, and is bordered by a dark-colored floor. It leads to multiple doors, some of which are slightly ajar, providing access to rooms with large windows that allow natural light to enter. The rooms are well-lit, well-organized, and uncluttered, featuring furniture such as beds, nightstands, and flat-screen TVs. As you move around the space, the scenery shifts to reveal a series of connected rooms, including a bedroom and a living area, with the hallway serving as a central pathway that ties the different areas together, creating a sense of continuity and flow throughout the space."
# **Response:**
# ```json
# {{
#     "updated_history": "You are currently in a bedroom with a bed in the center. The room is well-organized, featuring a bed, nightstand, and flat-screen TV, with an uncluttered and modern interior."
# }}
# ```

# ---

# **Next Step Example (demonstrating the previous summary is the current history of the last step):**
# **Previous Summary:** "You are currently in a bedroom with large windows letting in natural light. The room is well-organized, featuring a bed, nightstand, and flat-screen TV, with an uncluttered and modern interior."
# **latest scene description:** "As you leave the bedroom, you step into the same narrow hallway illuminated by recessed lighting. The walls continue their decorative patterns—one textured and one with a blue geometric design—and a dark-colored floor stretches ahead. To your left, a slightly ajar door reveals a modern bathroom featuring a sleek sink and a large mirror. Faint sounds of dripping water echo from within, and the cooler air contrasts with the warmth of the bedroom you just left."
# **Response:**
# ```json
# {{
#     "updated_history": "You started in the well-lit bedroom, stepped into the hallway with decorative walls, then approached the slightly ajar bathroom door to the left."
# }}
# ```

# **Now, evaluate the current history (if any) and the latest scene description and return the JSON object with updated history:**

# **Previous Summary:** {current_history}
# **Latest Scene Description:** {current_img_desc}
# **Response:**
# """
#     response = llm.invoke(prompt=prompt).content
    
#     markdown_match = re.search(r"```json\s*({.*?})\s*```", response, re.DOTALL)
    
#     if markdown_match:
#         response = markdown_match.group(1).strip()  # Extract the JSON content

#     # Parse the JSON response
#     data = json.loads(response)
    
#     # Check for the "goal_achieved" key
#     if "updated_history" not in data:
#         raise ValueError("Response JSON does not contain 'updated_history' field.")
    
#     # Extract the value of "goal_achieved"
#     updated_history = data["updated_history"].strip()

#     unload_model(llm)
#     return updated_history.lower()

def update_history(current_history: str, current_img_desc: str):
    
    print("\nCurrently in update_history() function\n")
    
    if not current_history:
        prompt = f"""Summarize the following scene description in a brief sentence, giving a sense of where the agent is standing currently:

Scene Description:
{current_img_desc}"""
    else:
        prompt = f"""Update the agent's movement history based on the new scene description.

        **Current History:**
        {current_history}

        **New Scene Description:**
        {current_img_desc}

        **Instructions:**
        - Analyze the new scene description and determine the agent's movement.
        - Append the new location to the current history, maintaining chronological order.
        - Mention both the previous and new locations, highlighting the transition.
        - Avoid re-describing details or locations that remain unchanged from the previous summary unless essential for clarity.
        - Use very simple and clear language.
        - Provide the updated history in a single short sentence.
        - Ensure the updated history includes all prior locations without unnecessary repetition.

        **Example:**

        *Current History:*
        "The agent is next to a dining table in the kitchen."

        *New Scene Description:*
        "The agent is now beside a sofa in the living room."

        *Updated History Output:*
        "The agent moved from sitting next to a dining table in the kitchen to standing beside a sofa in the living room."

        **Provide the updated history below:**
        """
    
    updated_history = llm.invoke(prompt=prompt).content.strip()
    return updated_history




# def get_goal_decision_single_image(goal: str, image):
    
#     img_desc_prompt = f"""Given the Image, and the goal provided, determine if the goal has been achieved based on the Image. 
#     Provide a clear answer with 'yes' or 'no' based on whether the goal has been reached.
#     Goal: {goal}
#     Has the goal been reached? (yes/no)"""

#     response = Llava_generation(img_desc_prompt, image)

#     return response.strip().lower() == "yes"

def get_goal_decision_single_image(goal: str, image,current_history:str):
    img_desc_prompt = f"""
    You are an intelligent assistant tasked with determining whether the room given in the final location in the goal statement has been reached based on the image provided and the past history of rooms that have been visited.
    **Note:** Focus on the room (bedroom, hallway, kitchen, outside, etc.) instead of minor sub-locations (like the foot of the bed).
    For example, if the goal is to go to the foot of the bed, visiting the bedroom fulfills that requirement, because being in the bedroom implies you could stand at the foot of the bed.

    **Instructions:**
    - Analyze the provided image carefully.
    - You must use the current image only to deduce if the last location in the goal statement has been reached.
    - You must use the history to deduce if the other rooms mentioned in the goal statement have been visited, regardless of sequence or positon inside room.
    - Ignore minor sub-location details in the goal statement. If the goal is “foot of the bed,” interpret that as “the bedroom.” If you have visited the bedroom (or the image shows you are in it), you can consider that goal achieved.
    - Compare the image with the final location in goal and current history of visited rooms against the specified goal rooms before the last one.
    - If the goal room(s) excluding the final room mentioned have already been visited based on the history and the image shows a location consistent with that last room in the goal, respond with a yes
    - Give a reason as to why you think the goal is reached or not. 
    - Ensure the JSON is properly formatted with double quotes and no trailing commas.
    - Do not include any additional information beyond the JSON object.

    **Examples:**

    **Goal:** "Head straight throguht the hallway. Turn right past the room door and stop at the the kitchen counter"
    **Image:** "A grey kitchen counter with a sink and a faucet. There is a window above the counter with a view of the garden."
    **Current History:** "The person have walked through the hallway."
    **Reason:** "The image shows that you are in proximity of the kitchen counter, and the history tells that hallway has been visited hence indicating that the goal has been achieved."
    **Response:**
    ```json
    {{
        "goal_achieved": "Yes"
    }}
    ```

    **Now, evaluate the following goal and the provided image:**

    **Goal:** {goal}
    **Current History:** {current_history}
    **Reason**:
    **Response:**
    """
    
    # print("Prompt = ",img_desc_prompt)
    response = Llava_generation(img_desc_prompt, image)
    print("\n IsGoal LLava response: ",response)
    
    try:
        markdown_match = re.search(r"```json\s*({.*?})\s*```", response, re.DOTALL)
        if markdown_match:
            response = markdown_match.group(1).strip()  # Extract the JSON content

        # Parse the JSON response
        data = json.loads(response)
        
        # Check for the "goal_achieved" key
        if "goal_achieved" not in data:
            raise ValueError("Response JSON does not contain 'goal_achieved' field.")
        
        # Extract the value of "goal_achieved"
        goal_achieved = data["goal_achieved"].strip()
        
        # Validate the value of "goal_achieved"
        if goal_achieved not in {"Yes", "No"}:
            raise ValueError(f"Invalid value for 'goal_achieved': {goal_achieved}")
        
        # Return True if "Yes", otherwise False
        print("\nGoal achieved: ",goal_achieved)
        return goal_achieved == "Yes"
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        # Raise a descriptive error if something goes wrong
        raise ValueError(f"Failed to parse or validate response: {e}")


@app.route('/make_decision', methods=['POST'])
def make_decision():
    try:
        data = request.json
        nav_locs = data.get('viewpoint_descs', {})
        current_img_desc = data.get('current_img_desc', '')
        goal = data.get('goal','')
        history = data.get('history','')
      
        if len(nav_locs) > 1:
            v_id = select_viewpoint(current_img_desc,nav_locs,goal,history)
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
        simple = data.get('simple', False) #False is the default, when simple doesnt exist

        images_list = []
        for idx, img_str in enumerate(images_data):
            img_data = base64.b64decode(img_str)
            img_array = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            images_list.append(image)

        
        descriptions = getImageDescriptions(images_list,simple)
    

        unified_description = unify_descriptions(descriptions,simple)
       

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
            decision = get_goal_decision_single_image(Goal,image,current_history)
            if decision:
                pos_count+=1

        if(pos_count>=2):
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
        current_histroy = data.get('current_history', "")

        updated_history = update_history(current_histroy,current_img_desc)
        print("\nupdated_history: ",updated_history)

        return jsonify({"updated_history": updated_history})
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)



