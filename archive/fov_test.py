#!/usr/bin/env python
import MatterSim
import time
import math
import cv2
import os
import numpy as np
import requests
import json
import base64
import networkx as nx
import re
from unidecode import unidecode

# ---------------------------
# Configuration and Setup
# ---------------------------
WIDTH = 800
HEIGHT = 600
VFOV = math.radians(60)
HFOV = VFOV * WIDTH / HEIGHT
TEXT_COLOR = [230, 40, 40]
EPISODE_NUMBER = '17DRP5sb8fy'
NUM_IMAGES=3

cv2.namedWindow('Python RGB')

sim = MatterSim.Simulator()
sim.setCameraResolution(WIDTH, HEIGHT)
sim.setCameraVFOV(VFOV)
sim.setDepthEnabled(False)  # Depth outputs are off
sim.initialize()
# Start an episode
# sim.newEpisode(['17DRP5sb8fy'], ['ee59d6b5e5da4def9fe85a8ba94ecf25'], [0], [0])
sim.newEpisode(['17DRP5sb8fy'],['08c774f20c984008882da2b8547850eb'],[0],[0])

# Define goal information
# goal_viewpoint = "e34dcf54d26a4a95869cc8a0c01cd2be"
goal_viewpoint="e34dcf54d26a4a95869cc8a0c01cd2be"
# goal = ("Go straight and slightly right toward the glass table and white chairs. "
#         "Pass the table and head straight, pass the couches and go into the room straight ahead. "
#         "Wait by the bed.")
goal = ("Walk forward past the living room, into the dining room. Stop in the doorway to the bedroom.")


# Some navigation parameters
heading = 0
elevation = 0
location = 0
ANGLEDELTA = 5 * math.pi / 180
SIXTYDELTA = 60 * math.pi / 180
current_history = ""
viewpoints_to_exclude = []
path_history = []

# ---------------------------
# Initialize the Navigation Graph
# ---------------------------
# The graph is empty at the start; as new viewpoints are discovered it will be updated.
nav_graph = nx.DiGraph()

print('\nPython Demo')
print('Use arrow keys to move the camera.')
print('Use number keys (not numpad) to move to nearby viewpoints indicated in the RGB view.')
print('Depth outputs are turned off by default.\n')

# ---------------------------
# Helper Functions
# ---------------------------
def get_images(sim):
    """
    Capture a 360° view by taking the current image and rotating the camera.
    """
    state = sim.getState()[0]
    locations = state.navigableLocations
    rgb = np.array(state.rgb, copy=False)

    # Capture initial image plus six rotated views
    panorama_images = [rgb]
    for i in range(6):
        sim.makeAction([location], [SIXTYDELTA], [elevation])
        state = sim.getState()[0]
        locations = state.navigableLocations
        rgb = np.array(state.rgb, copy=False)
        for idx, loc in enumerate(locations[1:]):
            fontScale = 3.0 / loc.rel_distance
            x = int(WIDTH / 2 + loc.rel_heading / HFOV * WIDTH)
            y = int(HEIGHT / 2 - loc.rel_elevation / VFOV * HEIGHT)
            cv2.putText(rgb, str(idx + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, TEXT_COLOR, thickness=3)
        if i == 5:
            continue
        panorama_images.append(np.copy(rgb))
    return panorama_images

def filter_images(images, num_images):
    """
    Takes a list of images and returns num_images from them.
    The first image is always included. Then, it alternates between the front and back 
    of the list until the required number of images is selected.
    
    :param images: List of images.
    :param num_images: Number of images to return.
    :return: List of selected images.
    """
    if not images or num_images <= 0:
        return []
    
    selected = [images[0]]  # Always include the first image
    left, right = 1, len(images) - 1  # Pointers to front and back
    
    while len(selected) < num_images and left <= right:
        if len(selected) < num_images:
            selected.append(images[left])
            left += 1
        if len(selected) < num_images and right >= left:
            selected.append(images[right])
            right -= 1
    print("Number of selected images: ",len(selected))
    return selected


def get_desc(images_list, simple=False):
    """
    Send a list of images (encoded as base64) to the server endpoint '/get_desc'
    to obtain a unified description.
    """
    url = "http://host.docker.internal:3000/get_desc"
    data_to_send = []
    for array in images_list:
        _, buffer = cv2.imencode('.jpg', array)
        img_str = base64.b64encode(buffer).decode('utf-8')
        data_to_send.append(img_str)
    response = requests.post(url, json={"arrays": data_to_send, "simple": simple})
    if response.status_code == 200:
        print("Got image description!")
    else:
        print("Failed to send data. Status code:", response.status_code)
    return response.json()

def get_history(history, current_img_desc):
    """
    Update the navigation history by sending the current image description and the existing history
    to the server endpoint '/get_history'.
    """
    url = "http://host.docker.internal:3000/get_history"
    response = requests.post(url, json={"current_img_desc": current_img_desc, "current_history": history})
    response = response.json()
    print(f"Updated history: {response['updated_history']}")
    return response['updated_history']

def nav2viewpoint(viewpointId, sim):
    """
    Navigate to a given viewpoint and display the resulting RGB image.
    """
    sim.makeAction([viewpointId], [0], [0])
    state = sim.getState()[0]
    locations = state.navigableLocations
    rgb = np.array(state.rgb, copy=False)
    for idx, loc in enumerate(locations[1:]):
        fontScale = 3.0 / loc.rel_distance
        x = int(WIDTH / 2 + loc.rel_heading / HFOV * WIDTH)
        y = int(HEIGHT / 2 - loc.rel_elevation / VFOV * HEIGHT)
        cv2.putText(rgb, str(idx + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale, TEXT_COLOR, thickness=3)
    cv2.imshow('Python RGB', rgb)
    print("This is where you are currently (Press any key to continue)")
    cv2.waitKey(0)
    return state

def update_graph(graph, current_vp, next_vp, description):
    """
    Add nodes (if not already present) and an edge from the current viewpoint to the next.
    """
    if current_vp not in graph:
        graph.add_node(current_vp, description="")
    if next_vp not in graph:
        graph.add_node(next_vp, description=description)
    graph.add_edge(current_vp, next_vp)

def update_node_description(graph, vp, description):
    """
    Update (or add) the description attribute for a given viewpoint node.
    """
    if vp in graph.nodes:
        graph.nodes[vp]['description'] = description
    else:
        graph.add_node(vp, description=description)

def graph_to_text(graph):
    """
    Convert the current navigation graph to a text summary.
    """
    text = "Navigation Graph:\nNodes:\n"
    for node, data in graph.nodes(data=True):
        text += f"  - {node}: {data.get('description', '')}\n"
    text += "Edges:\n"
    for u, v in graph.edges():
        text += f"  - {u} -> {v}\n"
    return text

def graph_to_text_shortened(graph, viewpointIds):
    """
    Convert the current navigation graph to a text summary.
    For nodes that are in the graph but NOT in viewpointIds, fetch their summarized descriptions from the LLM via API.
    The graph remains unchanged, but the text output reflects the updated descriptions.
    """

    url = "http://host.docker.internal:3000/summarize_desc"
    summarized_descriptions = {}

    # Convert viewpointIds list of objects to a set of viewpoint IDs for faster lookup
    viewpointIds_set = {loc.viewpointId for loc in viewpointIds}

    # Identify nodes that need summarization (present in the graph but NOT in viewpointIds)
    nodes_to_summarize = [node for node in graph.nodes if node not in viewpointIds_set]

    # Send each viewpoint ID and description one by one for summarization
    for node in nodes_to_summarize:
        if node in graph.nodes:
            original_description = graph.nodes[node].get('description', '')
            response = requests.post(url, json={"viewpoint_id": node, "description": original_description})
            if response.status_code == 200:
                summarized_descriptions[node] = response.json().get("summarized_description", original_description)
            else:
                print(f"Failed to summarize description for {node}. Status code:", response.status_code)
                summarized_descriptions[node] = original_description  # Fallback to original description

    # Generate text output
    text = "Navigation Graph:\nNodes:\n"
    for node, data in graph.nodes(data=True):
        # Use summarized description if available, else use the original
        description = summarized_descriptions.get(node, data.get('description', ''))
        text += f"  - {node}: {description}\n"

    text += "Edges:\n"
    for u, v in graph.edges():
        text += f"  - {u} -> {v}\n"

    return text




def get_lookahead_desc(sim, locations, viewpoint_descs, current_viewpoint, nav_graph):
    """
    Explore neighboring viewpoints (lookahead) to gather descriptions.
    Each discovered viewpoint is added to the viewpoint_descs dictionary and to the navigation graph as a node (if not already present).
    Additionally, an edge is added from the current viewpoint to each discovered neighbor.
    """
    location = 0
    elevation = 0
    current_vp_id = current_viewpoint.viewpointId  # current viewpoint's ID

    # First loop over immediate neighbors
    for loc in locations[1:]:
        print("locations")
        if loc.viewpointId in viewpoints_to_exclude:
            continue
        # Navigate to the neighbor viewpoint for lookahead
        print(loc.viewpointId)
        sim.newEpisode([EPISODE_NUMBER], [loc.viewpointId], [0], [0])
        images_list = get_images(sim)
        images_list= filter_images(images_list,NUM_IMAGES)
        print("Images list-1: ",len(images_list))
        viewpoint_id = str(loc.viewpointId)  # Convert to string to use as folder name
        url = "http://host.docker.internal:3000/save_images"
        response = requests.post(url, json={
    "images":  [img.tolist() for img in images_list],
    "viewpoint_id": viewpoint_id
})
        #desc_response = get_desc(images_list)
        viewpoint_descs[loc.viewpointId] = ""
        #desc = desc_response.get("unified_description", "")
        
        # Add the neighbor as a node if not already present
        desc=""
        if loc.viewpointId not in nav_graph:
            nav_graph.add_node(loc.viewpointId, description=desc)
        # Add an edge from the current viewpoint to this neighbor
        nav_graph.add_edge(current_vp_id, loc.viewpointId)
        
        if loc.viewpointId == goal_viewpoint:
            print("Goal Reached")
            print("This is Goal (Press any key to continue)")
            cv2.waitKey(0)
            return "Goal"

    # Second loop: rotate and explore further
    sim.newEpisode([EPISODE_NUMBER], [current_viewpoint.viewpointId], [0], [0])
    current_angle = 0
    for i in range(6):
        sim.makeAction([location], [SIXTYDELTA], [elevation])
        current_angle += SIXTYDELTA
        state = sim.getState()[0]
        locations = state.navigableLocations
        print("Number of locations: ",len(locations))

        for loc in locations[1:]:
            sim.newEpisode([EPISODE_NUMBER], [loc.viewpointId], [current_angle], [0])
            # Skip if this viewpoint was already discovered
            if loc.viewpointId in viewpoint_descs:
                continue
            if loc.viewpointId == goal_viewpoint:
                print("Goal Reached")
                print("This is Goal (Press any key to continue)")
                cv2.waitKey(0)
                return "Goal"
            images_list = get_images(sim)
            images_list=filter_images(images_list,NUM_IMAGES)
            viewpoint_id = str(loc.viewpointId)  # Convert to string to use as folder name
            url = "http://host.docker.internal:3000/save_images"
            response = requests.post(url, json={
    "images":  [img.tolist() for img in images_list],
    "viewpoint_id": viewpoint_id
})
            print("IMages list 2: ",len(images_list))
            #desc_response = get_desc(images_list)
            viewpoint_descs[loc.viewpointId] = ""
            #desc = desc_response.get("unified_description", "")
            desc=""
            if loc.viewpointId not in nav_graph:
                nav_graph.add_node(loc.viewpointId, description=desc)
            # Add an edge from the current viewpoint to this neighbor as well
            nav_graph.add_edge(current_vp_id, loc.viewpointId)
        sim.newEpisode([EPISODE_NUMBER], [current_viewpoint.viewpointId], [current_angle], [0])
    return None

# ---------------------------
# Main Simulation Loop
# ---------------------------
visited = [] #this array keeps track of the visited viewpoint ids
while True:
    viewpoint_descs = {}  # Dictionary to hold descriptions for each discovered viewpoint
    location = 0
    heading = 0
    elevation = 0
    sim.makeAction([location], [heading], [elevation])
    state = sim.getState()[0]
    locations = state.navigableLocations
    #how to access viewpoint ids:
    # for loc in locations[1:]: #of neighbours
    #     print("\nLoc: ",loc.viewpointId)
    # print("\nCurrent viewpoint: ",state.location.viewpointId)

    rgb = np.array(state.rgb, copy=False)
    cv2.imshow('Python RGB', rgb)
    cv2.waitKey(1)

    # Capture a 360° view of the current location
    panorama_images = get_images(sim)
    current_viewpoint = state.location  # Assume this object has a 'viewpointId' attribute
    current_vp_id = current_viewpoint.viewpointId

    visited.append(current_vp_id) #root of the graph added

    if current_vp_id not in nav_graph:
        nav_graph.add_node(current_vp_id, description="")

    # Get unified image descriptions (detailed and simple versions)
    # current_img_desc = get_desc(panorama_images)
    # current_img_desc_simple = get_desc(panorama_images, simple=True)
    # curr_desc = current_img_desc.get("unified_description", "")
    # update_node_description(nav_graph, current_vp_id, curr_desc)

    # Lookahead: gather descriptions for nearby viewpoints
    lookaheadgoalfound = get_lookahead_desc(sim, locations, viewpoint_descs, current_viewpoint, nav_graph)
    if lookaheadgoalfound == 'Goal':
        print("Goal found in lookahead, exiting.")
        break
    else:
        break
    