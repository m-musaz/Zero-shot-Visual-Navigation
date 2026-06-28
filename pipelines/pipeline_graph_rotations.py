
import MatterSim
import math
import cv2
import numpy as np
import requests
import json
import base64
import networkx as nx
from unidecode import unidecode
from pipelines.pipeline import NavigationPipeline

class RotationsPipeline(NavigationPipeline):
    def __init__(self, enable_cycle_detection=False):  # Added flag
        super().__init__()
        self.WIDTH = 800
        self.HEIGHT = 600
        self.VFOV = math.radians(60)
        self.HFOV = self.VFOV * self.WIDTH / self.HEIGHT
        self.TEXT_COLOR = [230, 40, 40]
        self.ANGLEDELTA = 5 * math.pi / 180
        self.SIXTYDELTA = 60 * math.pi / 180
        self.NUM_IMAGES = 3
        self.enable_cycle_detection = enable_cycle_detection  # New flag

        self.sim = MatterSim.Simulator()
        self.sim.setCameraResolution(self.WIDTH, self.HEIGHT)
        self.sim.setCameraVFOV(self.VFOV)
        self.sim.setDepthEnabled(False)
        self.sim.initialize()

    def run(self, instruction, scan, starting_viewpoint, 
            instruction_id, goal_viewpoint,max_length):
        trajectory = []
        self.sim.newEpisode([scan], [starting_viewpoint], [0], [0])
        current_state = self.sim.getState()[0]
        trajectory.append((current_state.location.viewpointId, 
                         current_state.heading, current_state.elevation))

        nav_graph = nx.DiGraph()
        visited = [starting_viewpoint]
        path_history = [starting_viewpoint]
        viewpoints_to_exclude = []  # Original unused list remains
        viewpoint_descs = {}

        while True:
            state = self.sim.getState()[0]
            current_vp_id = state.location.viewpointId
            
            if current_vp_id == goal_viewpoint:
                break

            if len(trajectory) > max_length:
                print("Max length reached for instruction:", instruction_id)
                break

            panorama_images = self._get_images()
            if current_vp_id not in nav_graph.nodes:
                nav_graph.add_node(current_vp_id, description="")

            current_img_desc = self._get_desc(panorama_images)
            current_desc = current_img_desc.get("unified_description", "")
            self._update_node_description(nav_graph, current_vp_id, current_desc)

            lookahead_goal = self._get_lookahead_desc(state.navigableLocations, 
                                                    current_vp_id, nav_graph,
                                                    viewpoint_descs, scan, 
                                                    goal_viewpoint)
            if lookahead_goal == "Goal":
                break

            graph_text = self._graph_to_text_shortened(nav_graph, state.navigableLocations)
            response = requests.post(
                "http://host.docker.internal:3000/make_decision",
                json={
                    "current_img_desc": current_img_desc,
                    "viewpoint_descs": viewpoint_descs,
                    "goal": instruction,
                    "history": " -> ".join(path_history),
                    "graph_info": graph_text,
                    "current_viewpoint": current_vp_id
                }
            )
            best_viewpoint = response.json()
            chosen_vp = best_viewpoint.get("v_id", current_vp_id)

            # Only modified section: conditional cycle detection
            if self.enable_cycle_detection:  # New flag check
                if chosen_vp in path_history[-3:]:
                    cycle_response = requests.post(
                        "http://host.docker.internal:3000/detect_cycle",
                        json={
                            "current_viewpoint": current_vp_id,
                            "chosen_viewpoint": chosen_vp,
                            "history": " -> ".join(path_history),
                            "graph_info": graph_text,
                            "goal": instruction
                        }
                    )
                    cycle_data = cycle_response.json()
                    chosen_vp = cycle_data.get("alternative_viewpoint", chosen_vp)

            path_history.append(chosen_vp)
            self.sim.newEpisode([scan], [chosen_vp], [0], [0])
            new_state = self.sim.getState()[0]
            trajectory.append((new_state.location.viewpointId, 
                             new_state.heading, new_state.elevation))

            if new_state.location.viewpointId == goal_viewpoint:
                break

        return trajectory
    def _get_images(self):
        state = self.sim.getState()[0]
        images = [np.array(state.rgb, copy=False)]
        for _ in range(5):
            self.sim.makeAction([0], [self.SIXTYDELTA], [0])
            state = self.sim.getState()[0]
            images.append(np.array(state.rgb, copy=False))
        self.sim.makeAction([0], [self.SIXTYDELTA], [0])
        return images

    def _filter_images(self, images, num_images):
        if not images or num_images <= 0:
            return []
        selected = [images[0]]
        left, right = 1, len(images) - 1
        while len(selected) < num_images and left <= right:
            selected.append(images[left])
            left += 1
            if len(selected) < num_images and right >= left:
                selected.append(images[right])
                right -= 1
        return selected

    def _get_desc(self, images_list, simple=False):
        encoded_images = []
        for img in images_list:
            _, buffer = cv2.imencode('.jpg', img)
            encoded_images.append(base64.b64encode(buffer).decode('utf-8'))
        response = requests.post(
            "http://host.docker.internal:3000/get_desc",
            json={"arrays": encoded_images, "simple": simple}
        )
        return response.json()

    def _get_lookahead_desc(self, locations, current_vp_id, nav_graph, 
                           viewpoint_descs, scan, goal_viewpoint):
        saved_state = self.sim.getState()[0]
        current_heading = saved_state.heading
        current_elevation = saved_state.elevation

        # Check immediate neighbors
        for loc in locations[1:]:
            if loc.viewpointId == goal_viewpoint:
                self.sim.newEpisode([scan], [saved_state.location.viewpointId], 
                                   [current_heading], [current_elevation])
                return "Goal"
            if loc.viewpointId in viewpoint_descs:
                continue
            self.sim.newEpisode([scan], [loc.viewpointId], [0], [0])
            images = self._get_images()
            filtered_images = self._filter_images(images, self.NUM_IMAGES)
            desc = self._get_desc(filtered_images)
            viewpoint_descs[loc.viewpointId] = desc
            desc_text = desc.get("unified_description", "")
            if loc.viewpointId not in nav_graph.nodes:
                nav_graph.add_node(loc.viewpointId, description=desc_text)
            nav_graph.add_edge(current_vp_id, loc.viewpointId)
            self.sim.newEpisode([scan], [saved_state.location.viewpointId], 
                              [current_heading], [current_elevation])

        # Check rotated positions
        self.sim.newEpisode([scan], [current_vp_id], [0], [0])
        current_angle = 0
        for _ in range(6):
            self.sim.makeAction([0], [self.SIXTYDELTA], [0])
            current_angle += self.SIXTYDELTA
            state = self.sim.getState()[0]
            for loc in state.navigableLocations[1:]:
                if loc.viewpointId == goal_viewpoint:
                    self.sim.newEpisode([scan], [saved_state.location.viewpointId], 
                                     [current_heading], [current_elevation])
                    return "Goal"
                if loc.viewpointId in viewpoint_descs:
                    continue
                self.sim.newEpisode([scan], [loc.viewpointId], [current_angle], [0])
                images = self._get_images()
                filtered_images = self._filter_images(images, self.NUM_IMAGES)
                desc = self._get_desc(filtered_images)
                viewpoint_descs[loc.viewpointId] = desc
                desc_text = desc.get("unified_description", "")
                if loc.viewpointId not in nav_graph.nodes:
                    nav_graph.add_node(loc.viewpointId, description=desc_text)
                nav_graph.add_edge(current_vp_id, loc.viewpointId)
                self.sim.newEpisode([scan], [current_vp_id], [current_angle], [0])
            self.sim.newEpisode([scan], [current_vp_id], [current_angle], [0])

        self.sim.newEpisode([scan], [saved_state.location.viewpointId], 
                          [current_heading], [current_elevation])
        return None

    def _update_node_description(self, graph, vp, description):
        if vp in graph.nodes:
            graph.nodes[vp]['description'] = description
        else:
            graph.add_node(vp, description=description)

    def _graph_to_text_shortened(self, graph, viewpointIds):
        viewpointIds_set = {loc.viewpointId for loc in viewpointIds}
        nodes_to_summarize = [node for node in graph.nodes if node not in viewpointIds_set]
        summarized = {}
        for node in nodes_to_summarize:
            desc = graph.nodes[node].get('description', '')
            response = requests.post(
                "http://host.docker.internal:3000/summarize_desc",
                json={"viewpoint_id": node, "description": desc}
            )
            summarized[node] = response.json().get("summarized_description", desc) \
                             if response.status_code == 200 else desc

        text = "Navigation Graph:\nNodes:\n"
        for node, data in graph.nodes(data=True):
            text += f"  - {node}: {summarized.get(node, data.get('description', ''))}\n"
        text += "Edges:\n"
        for u, v in graph.edges():
            text += f"  - {u} -> {v}\n"
        return unidecode(text)