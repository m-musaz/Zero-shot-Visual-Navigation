import MatterSim
import time
import math
import cv2
import os
import numpy as np
import requests
import json
import base64


WIDTH = 800
HEIGHT = 600
VFOV = math.radians(60)
HFOV = VFOV*WIDTH/HEIGHT
TEXT_COLOR = [230, 40, 40]
EPISODE_NUMBER = '17DRP5sb8fy'

cv2.namedWindow('Python RGB')
# cv2.namedWindow('Python Depth')

sim = MatterSim.Simulator()
sim.setCameraResolution(WIDTH, HEIGHT)
sim.setCameraVFOV(VFOV)
sim.setDepthEnabled(False) # Turn on depth only after running ./scripts/depth_to_skybox.py (see README.md)
sim.initialize()
#sim.newEpisode(['2t7WUuJeko7'], ['1e6b606b44df4a6086c0f97e826d4d15'], [0], [0])
#sim.newEpisode(['1LXtFkjw3qL'], ['0b22fa63d0f54a529c525afbf2e8bb25'], [0], [0])
# sim.newEpisode(['17DRP5sb8fy'],['db145474a5fa476d95c2cc7f09e7c83a'],[0],[0])
# sim.newEpisode(['17DRP5sb8fy'],['ee59d6b5e5da4def9fe85a8ba94ecf25'],[0],[0])
# sim.newEpisode(['17DRP5sb8fy'],['e693b5de8ad84d4cb61a79ece2e66d11'],[0],[0])
sim.newEpisode(['17DRP5sb8fy'],['08c774f20c984008882da2b8547850eb'],[0],[0])
# goal_viewpoint = "cb66d4266148455bbddf5d059a483711"
# goal_viewpoint="e34dcf54d26a4a95869cc8a0c01cd2be"
# goal_viewpoint="5e9f4f8654574e699480e90ecdd150c8"
goal_viewpoint="e34dcf54d26a4a95869cc8a0c01cd2be"
# goal="Walk to the foot of the bed and to a little sitting area.  Go through the door on the left side.  Stop once you have stepped into the bathroom."
# goal = "Go straight and slightly right toward the glass table and white chairs. Pass the table and head straight, pass the couches and go into the room straight ahead. Wait by the bed."
# goal = "Exit the bathroom, cross the room passed the black doors on the right and then go to the right, passed the room with white counters in it, then go to the spot between the glass wall and the third counter chair."
goal = "Walk past the dining room into the sittng room. Walk through the sitting room past the picture of the mermaid towards the bedroom.  Stop right after the open door of the bedroom. "
STARTING_VIEWPOINT = "08c774f20c984008882da2b8547850eb"
heading = 0
elevation = 0
location = 0
ANGLEDELTA = 5 * math.pi / 180
SIXTYDELTA = 60 * math.pi / 180
current_history = ""
viewpoints_to_exclude=[]
print('\nPython Demo')
print('Use arrow keys to move the camera.')
print('Use number keys (not numpad) to move to nearby viewpoints indicated in the RGB view.')
print('Depth outputs are turned off by default - check driver.py:L20 to enable.\n')

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
    cv2.waitKey(0)
    # depth = np.array(state.depth, copy=False)
    # cv2.imshow('Python Depth', depth)

    return state


def get_lookahead_desc(sim,locations,viewpoint_descs,current_viewpoint):
    location = 0
    elevation = 0

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
            cv2.waitKey(0)
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
                cv2.waitKey(0)
                return "Goal"
            
            viewpoint_descs[loc.viewpointId] = get_desc(images_list)
            # location_infos[i] = loc.viewpointId
            # i+=1
        sim.newEpisode([EPISODE_NUMBER],[current_viewpoint.viewpointId],[current_angle],[0])

        if(i==5):
            # print("not saved")
            continue


while True:
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
    # print("This is where you have Started from (Press any key to continue)")
    cv2.waitKey(1)

    #capturing 360 degree view
    panorama_images=[]
    panorama_images.append(rgb)
    panorama_images = get_images(sim)

    current_viewpoint = state.location


    #preparing images for sending
    current_img_desc = get_desc(panorama_images)
    current_img_desc_simple = get_desc(panorama_images,simple=True)
    current_history = get_history(history=current_history,current_img_desc=current_img_desc_simple)

    # i=0

    lookaheadgoalfound = get_lookahead_desc(sim,locations,viewpoint_descs,current_viewpoint)    

    if(lookaheadgoalfound == 'Goal'):
        print("Goal found in lookahead exiting")
        break


    #navigate to current viewpoint
    # state = nav2viewpoint(current_viewpoint,sim)
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
        sim.newEpisode([EPISODE_NUMBER],[best_viewpoint['v_id']],[0],[0])
        # state = sim.getState()[0]
        # rgb = np.array(state.rgb, copy=False)
        # cv2.imshow('Python RGB', rgb)
        # print("This is where you have navigated to (Press any key to continue)")
        # cv2.waitKey(0)
    except Exception as e:
        print("Cannot make new episode",e)
        continue
    
    if(best_viewpoint['v_id'] == goal_viewpoint):
        print("Goal Reached")
        print("This is Goal (Press any key to continue)")
        cv2.waitKey(0)
        break


    panorama_images=[]
    panorama_images.append(rgb)
    panorama_images = get_images(sim)

    data_to_send = []
    for array in panorama_images:
        _, buffer = cv2.imencode('.jpg', array)  # Convert to JPEG format
        img_str = base64.b64encode(buffer).decode('utf-8')  # Encode to base64
        #print data type og image str
        print(type(img_str))
        data_to_send.append(img_str)

    #ask server for goal condition
    url = "http://host.docker.internal:3000/is_goal"
    #pass current description and goal
    current_img_desc=viewpoint_descs[best_viewpoint['v_id']]
    # response = requests.post(url, json={"pano_images":data_to_send,"goal":goal,"current_history":current_history})
    # response = response.json()
    # if response['decision']:
    #     print("Goal Reached")
    #     print("This is Goal (Press any key to continue)")
    #     cv2.waitKey(0)
    #     break
    
    #if goal not reached, continue the loop


    # for idx, loc in enumerate(locations[1:]):
    #     # Draw actions on the screen
    #     fontScale = 3.0/loc.rel_distance
    #     x = int(WIDTH/2 + loc.rel_heading/HFOV*WIDTH)
    #     y = int(HEIGHT/2 - loc.rel_elevation/VFOV*HEIGHT)
    #     cv2.putText(rgb, str(idx + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
    #         fontScale, TEXT_COLOR, thickness=3)
    # cv2.imshow('Python RGB', rgb)
    # # llavainference(rgb)
    # # print output
    # print("This is where you are currently (Press any key to continue)")
    # cv2.waitKey(0)
    # # depth = np.array(state.depth, copy=False)
    # # cv2.imshow('Python Depth', depth)

    # next_move = response.json() #{"image_num": 1, "nav_point": 1}
    # if next_move["image_num"] == -1:
    #     print("Goal Reached")
    #     break
    # for i in range(next_move["image_num"]):
    #     sim.makeAction([location], [SIXTYDELTA], [elevation])
        
    #     location = 0
    #     heading = 0
    #     elevation = 0
    #     state = sim.getState()[0]
    #     # print(state.navigableLocations)
    #     current_location = state.location
    #     current_viewpoint = current_location.viewpointId
    #     print("Current viewpoint: ", current_viewpoint)
    #     locations = state.navigableLocations
    #     rgb = np.array(state.rgb, copy=False)
    #     for idx, loc in enumerate(locations[1:]):
    #         # Draw actions on the screen
    #         fontScale = 3.0/loc.rel_distance
    #         x = int(WIDTH/2 + loc.rel_heading/HFOV*WIDTH)
    #         y = int(HEIGHT/2 - loc.rel_elevation/VFOV*HEIGHT)
    #         cv2.putText(rgb, str(idx + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
    #             fontScale, TEXT_COLOR, thickness=3)
    #     heading = SIXTYDELTA

    # for idx, loc in enumerate(locations[1:]):
    #     # Draw actions on the screen
    #     fontScale = 3.0/loc.rel_distance
    #     x = int(WIDTH/2 + loc.rel_heading/HFOV*WIDTH)
    #     y = int(HEIGHT/2 - loc.rel_elevation/VFOV*HEIGHT)
    #     cv2.putText(rgb, str(idx + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
    #         fontScale, TEXT_COLOR, thickness=3)
    # cv2.imshow('Python RGB', rgb)

    # print("This is after turning to the predicted image (Press any key to continue)")
    # cv2.waitKey(0)

    # if 1 <= next_move['nav_point'] <= 9:
    #     location = next_move['nav_point']
    #     if location >= len(locations):
    #         print("Invalid location",next_move['nav_point'])
    #         location = 0
    
    # sim.makeAction([location], [0], [elevation])
    # location = 0
    # heading = 0
    # elevation = 0

    # state = sim.getState()[0]
    #     # print(state.navigableLocations)
    # locations = state.navigableLocations
    # rgb = np.array(state.rgb, copy=False)

    # for idx, loc in enumerate(locations[1:]):
    #     # Draw actions on the screen
    #     fontScale = 3.0/loc.rel_distance
    #     x = int(WIDTH/2 + loc.rel_heading/HFOV*WIDTH)
    #     y = int(HEIGHT/2 - loc.rel_elevation/VFOV*HEIGHT)
    #     cv2.putText(rgb, str(idx + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
    #         fontScale, TEXT_COLOR, thickness=3)
    # cv2.imshow('Python RGB', rgb)

    # print("This is after navigating to the predicted location (Press any key to continue)")
    # cv2.waitKey(0)