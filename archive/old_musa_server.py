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


#set env varaibles


tracer = LangChainTracer()

# Create a CallbackManager and add the tracer
callback_manager = CallbackManager(handlers=[tracer])
app = Flask(__name__)

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)

llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
    callback_manager=callback_manager,
    
)

model.to("cuda:0")
def select_viewpoint(curr_img_desc,viewpoints_dict, goal):
    # Create the prompt with the dictionary of viewpoint descriptions and goal
    descriptions = "\n".join([f"Viewpoint ID: {v_id}, Description: {desc}" for v_id, desc in viewpoints_dict.items()])
    prompt = f"""
    You are given a navigation task with a goal, and a set of viewpoint descriptions. Each viewpoint description corresponds to a unique location. 
    Your task is to identify the viewpoint description that most closely aligns with the goal or subgoal for the goal.

    \nGoal: {goal}

    \nViewpoint Descriptions:
    {descriptions}

    \nHere is the description of where I am currently:
    {curr_img_desc}

    Based on the goal, select the viewpoint ID that most closely matches or aligns with the goal description. 
    Only output the exact viewpoint ID, with no additional text.
    """

    
    # Assuming llm.invoke() is the method to call the language model
    response = llm.invoke(prompt).content
    
    # Extract and return the viewpoint ID from the response
    print("Listing all viewpoints IDs and descriptions\n")
    for v_id, desc in viewpoints_dict.items():
        print(f"Viewpoint ID: {v_id} \n")
        print(f"Description: {desc} \n\n")
    print("\n\n")
    print(f"Response that should have viewpoint ID: {response} \n")
    viewpoint_id = response.strip()  # Assuming response directly returns the viewpoint ID
   
    
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
def getImageDescriptions(images_list):
    img_desc_prompt = """Describe the image with a focus on navigational elements such as hallways, doorways, stairs, and open spaces.
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

def unify_descriptions(descriptions: list):
    processed_descriptions = "\n".join(descriptions)
    prompt = f"""You are given a list of image descriptions that capture various angles of a scene in a 360-degree view around a central point. 
    Please analyze these descriptions and generate a single, cohesive summary that combines the information from all angles.
    
    Consider the following when creating the unified description:
    Highlight any consistent or notable landmarks, objects, and their relative positions.
    Describe the surroundings in a continuous flow that logically follows the 360-degree perspective, mentioning any major shifts in scenery 
    as you move around.
    Focus on spatial relationships, distances, and directional cues where possible, so that the summary represents a clear overall layout of the scene.
    Here is the list of image descriptions: {processed_descriptions}
    Please generate a single, unified scene description that conveys the main elements and spatial layout clearly.
    Make sure your responses are concise and to one brief paragraph.
    """
    response = llm.invoke(prompt).content
    return response

def reached_goal(goal: str, descriptions: list):
   
    prompt = f"Given a set of image descriptions and the goal description, determine if the goal has been reached. Answer 'yes' or 'no'.\nGoal: {goal}\n"
    for i, desc in enumerate(descriptions):
        prompt += f"Image Description-{i}: {desc}"
    prompt += "\nHas the goal been reached? (yes/no)"

    response = llm.invoke(prompt).content
   

    return response.strip().lower() == "yes"

def reached_goal_unified(goal: str, description):
    
    prompt = f"""Given the scene description below, and the goal provided, determine if the goal has been achieved based on the description. 
    Provide a clear answer with 'yes' or 'no' based on whether the goal has been reached.
    Goal: {goal}
    Scene Description: {description}
    Has the goal been reached? (yes/no)"""

    response = llm.invoke(prompt).content
   

    return response.strip().lower() == "yes"

def update_history(current_history: str, current_img_desc: str):
    
    prompt = f"""You are a summarization assistant. Your task is to create a concise, movement-based summary of a person's actions and movements based on detailed scene descriptions. The summary should focus on key movements and transitions, avoiding unnecessary detail.

If no previous summary is provided, create a new summary starting from the given scene description. If a previous summary is provided, integrate the new movement into it to update the history.
**Note:** Always Keep the final summary concise, it should not be longer than one paragaph.
**Instructions:**
- Analyze the scene description carefully.
- Identify the key movements and actions performed by the person.
- Focus on what changed or what new locations you entered, mentioning your location around furniture like a bed or a dining table for example.
- Avoid re-describing details/locations that remain unchanged from the previous summary unless essential for clarity.
- Based on this, provide the final updated summary (or start a new one if no previous summary exists)
- Always Keep the final updated summary concise, it should not be longer than one paragaph.
- Ensure the JSON is properly formatted with double quotes and no trailing commas.
- Do not include any additional information beyond the JSON object.

**Examples:**
**Previous Summary(if any):** ""
**latest scene description:** "The scene is set in a modern interior space with a narrow hallway that connects various rooms. The hallway is characterized by decorative walls, including a textured wall and a wall with a blue geometric pattern, and is bordered by a dark-colored floor. It leads to multiple doors, some of which are slightly ajar, providing access to rooms with large windows that allow natural light to enter. The rooms are well-lit, well-organized, and uncluttered, featuring furniture such as beds, nightstands, and flat-screen TVs. As you move around the space, the scenery shifts to reveal a series of connected rooms, including a bedroom and a living area, with the hallway serving as a central pathway that ties the different areas together, creating a sense of continuity and flow throughout the space."
**Response:**
```json
{{
    "updated_history": "You are currently in a bedroom with a bed in the center. The room is well-organized, featuring a bed, nightstand, and flat-screen TV, with an uncluttered and modern interior."
}}
```

---

**Next Step Example (demonstrating the previous summary is the current history of the last step):**
**Previous Summary:** "You are currently in a bedroom with large windows letting in natural light. The room is well-organized, featuring a bed, nightstand, and flat-screen TV, with an uncluttered and modern interior."
**latest scene description:** "As you leave the bedroom, you step into the same narrow hallway illuminated by recessed lighting. The walls continue their decorative patterns—one textured and one with a blue geometric design—and a dark-colored floor stretches ahead. To your left, a slightly ajar door reveals a modern bathroom featuring a sleek sink and a large mirror. Faint sounds of dripping water echo from within, and the cooler air contrasts with the warmth of the bedroom you just left."
**Response:**
```json
{{
    "updated_history": "You started in the well-lit bedroom, stepped into the hallway with decorative walls, then approached the slightly ajar bathroom door to the left."
}}
```

**Now, evaluate the current history (if any) and the latest scene description and return the JSON object with updated history:**

**Previous Summary:** {current_history}
**Latest Scene Description:** {current_img_desc}
**Response:**
"""
    response = llm.invoke(prompt).content
    
    markdown_match = re.search(r"```json\s*({.*?})\s*```", response, re.DOTALL)
    
    if markdown_match:
        response = markdown_match.group(1).strip()  # Extract the JSON content

    # Parse the JSON response
    data = json.loads(response)
    
    # Check for the "goal_achieved" key
    if "updated_history" not in data:
        raise ValueError("Response JSON does not contain 'updated_history' field.")
    
    # Extract the value of "goal_achieved"
    updated_history = data["updated_history"].strip()

    return updated_history.lower()


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
      
        if len(nav_locs) > 1:
            v_id = select_viewpoint(current_img_desc,nav_locs,goal)
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

        images_list = []
        for idx, img_str in enumerate(images_data):
            img_data = base64.b64decode(img_str)
            img_array = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            images_list.append(image)

        
        descriptions = getImageDescriptions(images_list)
    

        unified_description = unify_descriptions(descriptions)
       

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

        return jsonify({"updated_history": updated_history})
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)



