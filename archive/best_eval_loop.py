import MatterSim
import time
import math
import cv2
import os
import numpy as np
import requests
import json
import base64
import pandas as pd

def load_json(file_path):
    try:
        df = pd.read_json(file_path, encoding="utf-8")
        return df
    except ValueError as e:
        print(f"Error loading JSON: {e}")
        return None

# Correct file path
file_path = "src/driver/R2R_train_updated.json"  # Adjust based on your structure

# Load and display the JSON data
json_df = load_json(file_path)
json_df_filtered = json_df.dropna(subset=['goals'])

# Sort by the 'scan' column (ascending order)
json_df_sorted = json_df_filtered.sort_values(by='scan', ascending=True)
json_df_sorted = json_df_sorted.reset_index(drop=True)



success = 0
failure = 0
results = []


WIDTH = 800
HEIGHT = 600
VFOV = math.radians(60)
HFOV = VFOV*WIDTH/HEIGHT
TEXT_COLOR = [230, 40, 40]

cv2.namedWindow('Python RGB')

sim = MatterSim.Simulator()
sim.setCameraResolution(WIDTH, HEIGHT)
sim.setCameraVFOV(VFOV)
sim.setDepthEnabled(False) # Turn on depth only after running ./scripts/depth_to_skybox.py (see README.md)
sim.initialize()

viewpoints_to_exclude=[]

def get_images(sim):
    state = sim.getState()[0]
    locations = state.navigableLocations
    rgb = np.array(state.rgb, copy=False)

    #capturing 360 degree view
    panorama_images = [rgb]
    for i in range(6):
        # print("move")
        sim.makeAction([location], [SIXTYDELTA], [elevation])
        state = sim.getState()[0]
        # print(state.navigableLocations)
        locations = state.navigableLocations
        rgb = np.array(state.rgb, copy=False)
        for idx, loc in enumerate(locations[1:]):
            # Draw actions on the screen
            fontScale = 3.0/loc.rel_distance
            x = int(WIDTH/2 + loc.rel_heading/HFOV*WIDTH)
            y = int(HEIGHT/2 - loc.rel_elevation/VFOV*HEIGHT)
            cv2.putText(rgb, str(idx + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale, TEXT_COLOR, thickness=3)
        if(i==5):
            # print("not saved")
            continue
        heading = SIXTYDELTA
        # print(heading)
        panorama_images.append(np.copy(rgb))
    return panorama_images

def get_desc(images_list,simple=False):
    url = "http://host.docker.internal:3000/get_desc"
    data_to_send = []
    for array in images_list:
        _, buffer = cv2.imencode('.jpg', array)  # Convert to JPEG format
        img_str = base64.b64encode(buffer).decode('utf-8')  # Encode to base64
        #print data type og image str
        print(type(img_str))
        data_to_send.append(img_str)

    # Send the data via POST request
    response = requests.post(url, json={"arrays": data_to_send,"simple":simple})

    # Print the response
    if response.status_code == 200:
        print("Got image description!")
        # print("Response:", response.json())
    else:
        print("Failed to send data. Status code:", response.status_code)
    return response.json()

def get_history(history,current_img_desc):
    url = "http://host.docker.internal:3000/get_history"
    response = requests.post(url, json={"current_img_desc":current_img_desc,"current_history":history})
    response = response.json()
    print(f"Updated history: {response['updated_history']}")
    return response['updated_history']

def nav2viewpoint(viewpointId,sim):
    sim.makeAction([viewpointId], [0], [0])
    location = 0
    heading = 0
    elevation = 0
    state = sim.getState()[0]
    # print(state.navigableLocations)
    locations = state.navigableLocations
    rgb = np.array(state.rgb, copy=False)
    for idx, loc in enumerate(locations[1:]):
        # Draw actions on the screen
        fontScale = 3.0/loc.rel_distance
        x = int(WIDTH/2 + loc.rel_heading/HFOV*WIDTH)
        y = int(HEIGHT/2 - loc.rel_elevation/VFOV*HEIGHT)
        cv2.putText(rgb, str(idx + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale, TEXT_COLOR, thickness=3)
    cv2.imshow('Python RGB', rgb)
    # llavainference(rgb)
    # print output
    print("This is where you are currently (Press any key to continue)")
    # cv2.waitKey(0)
    # depth = np.array(state.depth, copy=False)
    # cv2.imshow('Python Depth', depth)

    return state


def get_lookahead_desc(sim,locations,viewpoint_descs,current_viewpoint):
    location = 0
    elevation = 0
    print("viewpoints exclude lookahaed:",viewpoints_to_exclude)
    for loc in locations[1:]:
        #naviagte to each location
        if loc.viewpointId in viewpoints_to_exclude:
            continue
        sim.newEpisode([EPISODE_NUMBER],[loc.viewpointId],[0],[0])
        # state = nav2viewpoint(loc.viewpointId,sim)
        #get images and descriptions
        images_list = get_images(sim)
        viewpoint_descs[loc.viewpointId] = get_desc(images_list)

        if(loc.viewpointId == goal_viewpoint):
            print("Goal Reached")
            print("This is Goal (Press any key to continue)")
            # cv2.waitKey(0)
            return "Goal"

        # location_infos[i] = loc.viewpointId
        # i+=1
    sim.newEpisode([EPISODE_NUMBER],[current_viewpoint.viewpointId],[0],[0])

    current_angle = 0
    for i in range(6):
        # print("move")
        sim.makeAction([location], [SIXTYDELTA], [elevation])
        current_angle+=SIXTYDELTA
        state = sim.getState()[0]
        # print(state.navigableLocations)
        locations = state.navigableLocations

        for loc in locations[1:]:
            #naviagte to each location
            if loc.viewpointId in viewpoints_to_exclude:
                continue
            sim.newEpisode([EPISODE_NUMBER],[loc.viewpointId],[0],[0])
            # state = nav2viewpoint(loc.viewpointId,sim)
            #get images and descriptions
            images_list = get_images(sim)
            if(viewpoint_descs.get(loc.viewpointId)):
                continue

            if(loc.viewpointId == goal_viewpoint):
                print("Goal Reached")
                print("This is Goal (Press any key to continue)")
                # cv2.waitKey(0)
                return "Goal"
            
            viewpoint_descs[loc.viewpointId] = get_desc(images_list)
            # location_infos[i] = loc.viewpointId
            # i+=1
        sim.newEpisode([EPISODE_NUMBER],[current_viewpoint.viewpointId],[current_angle],[0])

        if(i==5):
            # print("not saved")
            continue

for index, row in json_df_sorted.iterrows():
    print("row: ",index)


    EPISODE_NUMBER = row['scan']
    STARTING_VIEWPOINT = row['path'][0]
    goal_viewpoint = row['path'][-1]
    goal_index = 0
    goal = row['instructions'][goal_index]
    goal_reached = False
    score = len(row['path'])
    all_paths_taken = []
    path_taken = []
    llm_errors = []

    print("Episode: ",EPISODE_NUMBER)
    print("Goal: ",goal)
    print("Starting VP: ",STARTING_VIEWPOINT)

    while goal_reached == False:
        path_taken = []
        goal_reached = False
        viewpoints_to_exclude=[]

        sim.newEpisode([EPISODE_NUMBER],[STARTING_VIEWPOINT],[0],[0])
        
        for step in range(len(row['path']*2)):
            goal = row['instructions'][goal_index]

            heading = 0
            elevation = 0
            location = 0
            ANGLEDELTA = 5 * math.pi / 180
            SIXTYDELTA = 60 * math.pi / 180
            current_history = ""

            viewpoint_descs={}
            location_infos=[]
            location = 0
            heading = 0
            elevation = 0
            sim.makeAction([location], [heading], [elevation])
            state = sim.getState()[0]
            locations = state.navigableLocations
            
            rgb = np.array(state.rgb, copy=False)
            cv2.imshow('Python RGB', rgb)

            #capturing 360 degree view
            panorama_images=[]
            panorama_images.append(rgb)
            panorama_images = get_images(sim)

            current_viewpoint = state.location


            #preparing images for sending
            current_img_desc = get_desc(panorama_images)
            current_history = get_history(history=current_history,current_img_desc=current_img_desc)


            lookaheadgoalfound = get_lookahead_desc(sim,locations,viewpoint_descs,current_viewpoint)    

            if(lookaheadgoalfound == 'Goal'):
                print("Goal found in lookahead exiting")
                goal_reached = True
                break


            #navigate to current viewpoint

            #send the descriptions to the server
            url = "http://host.docker.internal:3000/make_decision"
            response = requests.post(url, json={"current_img_desc":current_img_desc,"viewpoint_descs":viewpoint_descs,"goal":goal,"history":current_history})
            best_viewpoint = response.json()
            print(f"Best viewpoint: {best_viewpoint}")
            #navigate to the best viewpoint
            try:
                if len(viewpoints_to_exclude) >= 2:
                    viewpoints_to_exclude.pop(0)
                if(best_viewpoint['v_id'] != STARTING_VIEWPOINT):
                    viewpoints_to_exclude.append(best_viewpoint['v_id'])
                print("Exlude List:",viewpoints_to_exclude)
                sim.newEpisode([EPISODE_NUMBER],[best_viewpoint['v_id']],[0],[0])
                path_taken.append(best_viewpoint['v_id'])
            except Exception as e:
                print("Cannot make new episode",e)
                llm_errors.append(f"invalid viewpoint id genereated {e}")
                goal_index += 1
                try:
                    goal = row['instructions'][goal_index]
                except: #if no more goals remain
                    score = 0
                    for path in path_taken:
                        if(path in row['path']):
                            score = max(score,row['path'].index(path))
                        pass

                    score = (score/len(row['path'])) * 10 # normalized scoring
                    break

                continue
            
            if(best_viewpoint['v_id'] == goal_viewpoint):
                print("Goal Reached")
                goal_reached = True
                break
        

        all_paths_taken.append(path_taken)

        if(goal_reached == False):
            goal_index += 1
            score = 0
            for path in path_taken:
                if(path in row['path']):
                    score = max(score,row['path'].index(path))

            score = (score/len(row['path'])) * 10 # normalized scoring
                
            try:
                goal = row['instructions'][goal_index]
            except: #if no more goals remain
                break
        else:
            success+=1
            score = 10
            break

    results.append({
    "EPISODE":EPISODE_NUMBER,
    "starting viewpoint":STARTING_VIEWPOINT,
    "goal statement":goal,
    "retries":goal_index,
    "paths_taken":all_paths_taken,
    "score":score,
    "result":"pass" if goal_reached else "fail",
    "LLM errors":llm_errors,
    })

    df = pd.DataFrame(results)

    df.to_json("eval_results.json", orient="records", indent=4)