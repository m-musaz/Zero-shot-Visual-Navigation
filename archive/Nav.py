import MatterSim
import math
import cv2
import numpy as np
import requests
import base64
import json

class NavigationSimulator:
    WIDTH = 800
    HEIGHT = 600
    VFOV = math.radians(60)
    HFOV = VFOV * WIDTH / HEIGHT
    TEXT_COLOR = [230, 40, 40]
    ANGLE_DELTA = 5 * math.pi / 180
    SIXTY_DELTA = 60 * math.pi / 180
    URL = "http://host.docker.internal:3000/llava"

    def __init__(self):
        cv2.namedWindow('Python RGB')
        self.sim = MatterSim.Simulator()
        self.sim.setCameraResolution(self.WIDTH, self.HEIGHT)
        self.sim.setCameraVFOV(self.VFOV)
        self.sim.setDepthEnabled(False)
        self.sim.initialize()
        self.heading = 0
        self.elevation = 0
        self.location = 0

    def start_episode(self, scan, viewpoint):
        self.sim.newEpisode([scan], [viewpoint], [0], [0])
        print('\nPython Navigation Simulator')
        print('Use arrow keys to move the camera.')
        print('Use number keys (not numpad) to move to nearby viewpoints indicated in the RGB view.')

    def capture_navigation_images(self):
        images = []
        for i in range(6):
            state = self.sim.getState()[0]
            rgb = np.array(state.rgb, copy=False)
            images.append(np.copy(rgb))
            self.sim.makeAction([self.location], [self.SIXTY_DELTA], [self.elevation])
        return images

    def display_navigation_options(self, rgb, locations):
        for idx, loc in enumerate(locations[1:]):
            fontScale = 3.0 / loc.rel_distance
            x = int(self.WIDTH / 2 + loc.rel_heading / self.HFOV * self.WIDTH)
            y = int(self.HEIGHT / 2 - loc.rel_elevation / self.VFOV * self.HEIGHT)
            cv2.putText(rgb, str(idx + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, fontScale, self.TEXT_COLOR, thickness=3)

    def send_to_server(self, images,goal):
        data_to_send = []
        for img in images:
            _, buffer = cv2.imencode('.jpg', img)
            img_str = base64.b64encode(buffer).decode('utf-8')
            data_to_send.append(img_str)
        response = requests.post(self.URL, json={"arrays": data_to_send, "goal": goal})
        if response.status_code == 200:
            print("Data sent successfully!")
            return response.json()
        else:
            print("Failed to send data. Status code:", response.status_code)
            return None

    def run(self, scan, viewpoint,goal):
        self.start_episode(scan, viewpoint)
        while True:
            images = self.capture_navigation_images()
            response = self.send_to_server(images,goal)
            if response and response["image_num"] == -1:
                print("Goal Reached")
                break

            for _ in range(response["image_num"]):
                self.sim.makeAction([self.location], [self.SIXTY_DELTA], [self.elevation])
                state = self.sim.getState()[0]
                rgb = np.array(state.rgb, copy=False)
                locations = state.navigableLocations
                self.display_navigation_options(rgb, locations)
                cv2.imshow('Python RGB', rgb)
                cv2.waitKey(0)

            self.location = response['nav_point'] if 1 <= response['nav_point'] <= 9 else 0
            self.sim.makeAction([self.location], [0], [self.elevation])

            state = self.sim.getState()[0]
            rgb = np.array(state.rgb, copy=False)
            locations = state.navigableLocations
            self.display_navigation_options(rgb, locations)
            cv2.imshow('Python RGB', rgb)
            cv2.waitKey(0)

    def in_neighbourhood(self,neighbouring_viewpoints,end_viewpoint):
        for loc in neighbouring_viewpoints:
            if loc.viewpointId == end_viewpoint:
                return True
        return False

    def run_eval(self, scan, num_viewpoints):
            success = []
            
            scan_file = f"{scan}_data.json"
            scan_file_path="./src/driver/"+scan_file
            # Load navigation data from scan_data.json
            with open(scan_file_path, 'r') as f:
                scan_data = json.load(f)
            reasons = []
            

            
            scan_data = scan_data[:num_viewpoints]

            for nav_obj in scan_data:
                goal = nav_obj["goals"][0]
                viewpoint = nav_obj["path"][0]
                end_viewpoint = nav_obj["path"][-1]
                self.sim.newEpisode([scan], [viewpoint], [0], [0])
                
                step_counter = 0
                try:
                    while True:
                        if step_counter > 9:
                            print("Goal not reached")
                            reasons.append("Hallucination")
                            success.append(False)
                            break
                        
                        print("Step: ",step_counter)
                        images = self.capture_navigation_images()
                        response = self.send_to_server(images, goal)
                        
                        if response and response["image_num"] == -1:
                            state = self.sim.getState()[0]
                            current_location = state.location
                            current_viewpoint = current_location.viewpointId
                            neighbouring_viewpoints = state.navigableLocations
                            if current_viewpoint ==end_viewpoint or self.in_neighbourhood(neighbouring_viewpoints,end_viewpoint):
                                print("For goal: ",goal," Goal Reached")
                                success.append(True)
                            else:
                                print("Goal not reached")
                                reasons.append("Hallucination")
                                success.append(False)
                            break

                        for _ in range(response["image_num"]):
                            self.sim.makeAction([self.location], [self.SIXTY_DELTA], [self.elevation])
                            state = self.sim.getState()[0]
                            rgb = np.array(state.rgb, copy=False)
                            locations = state.navigableLocations
                            #self.display_navigation_options(rgb, locations)

                        self.location = response['nav_point'] if 1 <= response['nav_point'] <= 9 else 0
                        self.sim.makeAction([self.location], [0], [self.elevation])

                        state = self.sim.getState()[0]
                        rgb = np.array(state.rgb, copy=False)
                        locations = state.navigableLocations
                        #self.display_navigation_options(rgb, locations)
                        step_counter += 1
                except Exception as e:
                    print("Viewpoint: ", viewpoint, "failed due to exception: ", e)
                    reasons.append("Format")
                    success.append(False)

            # Output success or failure for each viewpoint and calculate the success rate
            print("Success rate:", sum(success) / len(success))
            print("Format failures: ", reasons.count("Format")/len(reasons))
            print("Hallucination failures: ", reasons.count("Hallucination")/len(reasons))

# Usage example:
nav_sim = NavigationSimulator()
nav_sim.run_eval("17DRP5sb8fy",2)
