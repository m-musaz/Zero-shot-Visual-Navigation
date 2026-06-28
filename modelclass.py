import math
import cv2
import numpy as np
import requests
import base64
import networkx as nx
from unidecode import unidecode
import MatterSim

class MatterSimNavigator:
    def __init__(self, episode_id, goal_viewpoint, goal_instruction, num_images=3):
        self.WIDTH = 800
        self.HEIGHT = 600
        self.VFOV = math.radians(60)
        self.HFOV = self.VFOV * self.WIDTH / self.HEIGHT
        self.TEXT_COLOR = [230, 40, 40]
        self.NUM_IMAGES = num_images
        self.EPISODE_NUMBER = episode_id
        self.goal_viewpoint = goal_viewpoint
        self.goal_instruction = goal_instruction

        self.sim = self._init_sim()
        self.nav_graph = nx.DiGraph()
        self.viewpoints_to_exclude = []
        self.path_history = []
        self.visited = []

        cv2.namedWindow('Python RGB')

    def _init_sim(self):
        sim = MatterSim.Simulator()
        sim.setCameraResolution(self.WIDTH, self.HEIGHT)
        sim.setCameraVFOV(self.VFOV)
        sim.setDepthEnabled(False)
        sim.initialize()
        return sim

    def get_images(self):
        state = self.sim.getState()[0]
        rgb = np.array(state.rgb, copy=False)
        panorama_images = [rgb]

        for _ in range(6):
            self.sim.makeAction([0], [60 * math.pi / 180], [0])
            state = self.sim.getState()[0]
            rgb = np.array(state.rgb, copy=False)
            panorama_images.append(np.copy(rgb))

        return panorama_images

    def filter_images(self, images):
        if not images or self.NUM_IMAGES <= 0:
            return []
        selected = [images[0]]
        left, right = 1, len(images) - 1
        while len(selected) < self.NUM_IMAGES and left <= right:
            if len(selected) < self.NUM_IMAGES:
                selected.append(images[left])
                left += 1
            if len(selected) < self.NUM_IMAGES:
                selected.append(images[right])
                right -= 1
        return selected

    def get_desc(self, images_list, simple=False):
        url = "http://host.docker.internal:3000/get_desc"
        encoded = [base64.b64encode(cv2.imencode('.jpg', img)[1]).decode('utf-8') for img in images_list]
        response = requests.post(url, json={"arrays": encoded, "simple": simple})
        return response.json()

    def get_lookahead_desc(self, locations, viewpoint_descs, current_vp_id):
        for loc in locations[1:]:
            if loc.viewpointId in self.viewpoints_to_exclude:
                continue
            self.sim.newEpisode([self.EPISODE_NUMBER], [loc.viewpointId], [0], [0])
            imgs = self.filter_images(self.get_images())
            desc = self.get_desc(imgs)
            unified = desc.get("unified_description", "")
            viewpoint_descs[loc.viewpointId] = desc
            self.nav_graph.add_node(loc.viewpointId, description=unified)
            self.nav_graph.add_edge(current_vp_id, loc.viewpointId)
            if loc.viewpointId == self.goal_viewpoint:
                return "Goal"
        return None

    def update_graph(self, current, next_vp, description=""):
        self.nav_graph.add_node(current, description=description)
        self.nav_graph.add_node(next_vp, description=description)
        self.nav_graph.add_edge(current, next_vp)

    def graph_to_text_shortened(self, locations):
        url = "http://host.docker.internal:3000/summarize_desc"
        vp_ids = {loc.viewpointId for loc in locations}
        summaries = {}

        to_summarize = [node for node in self.nav_graph.nodes if node not in vp_ids]
        for node in to_summarize:
            desc = self.nav_graph.nodes[node].get('description', '')
            res = requests.post(url, json={"viewpoint_id": node, "description": desc})
            if res.status_code == 200:
                summaries[node] = res.json().get("summarized_description", desc)

        text = "Navigation Graph:\nNodes:\n"
        for node, data in self.nav_graph.nodes(data=True):
            desc = summaries.get(node, data.get('description', ''))
            text += f"  - {node}: {desc}\n"
        text += "Edges:\n"
        for u, v in self.nav_graph.edges():
            text += f"  - {u} -> {v}\n"
        return unidecode(text)

    def run(self):
        self.sim.newEpisode([self.EPISODE_NUMBER], ['3577de361e1a46b1be544d37731bfde6'], [0], [0])

        while True:
            self.sim.makeAction([0], [0], [0])
            state = self.sim.getState()[0]
            locations = state.navigableLocations
            rgb = np.array(state.rgb, copy=False)
            cv2.imshow('Python RGB', rgb)
            cv2.waitKey(1)

            current_vp_id = state.location.viewpointId
            self.visited.append(current_vp_id)

            if current_vp_id not in self.nav_graph:
                self.nav_graph.add_node(current_vp_id, description="")

            panorama = self.filter_images(self.get_images())
            desc = self.get_desc(panorama)
            unified_desc = desc.get("unified_description", "")
            self.nav_graph.nodes[current_vp_id]['description'] = unified_desc

            viewpoint_descs = {}
            goal_found = self.get_lookahead_desc(locations, viewpoint_descs, current_vp_id)
            if goal_found == "Goal":
                print("Goal reached during lookahead.")
                break

            graph_text = self.graph_to_text_shortened(locations)
            decision = requests.post("http://host.docker.internal:3000/make_decision", json={
                "current_img_desc": desc,
                "viewpoint_descs": viewpoint_descs,
                "goal": self.goal_instruction,
                "history": " -> ".join(self.path_history),
                "graph_info": graph_text,
                "current_viewpoint": current_vp_id
            }).json()

            next_vp = decision['v_id']
            self.path_history.append(next_vp)

            self.update_graph(current_vp_id, next_vp,
                              viewpoint_descs.get(next_vp, {}).get("unified_description", ""))

            if next_vp == self.goal_viewpoint:
                print("Goal Reached at final decision point.")
                cv2.waitKey(0)
                break

            if len(self.viewpoints_to_exclude) >= 2:
                self.viewpoints_to_exclude.pop(0)
            self.viewpoints_to_exclude.append(next_vp)

            self.sim.newEpisode([self.EPISODE_NUMBER], [next_vp], [0], [0])
