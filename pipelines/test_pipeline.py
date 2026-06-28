from pipelines.pipeline import NavigationPipeline
import json 

def extract_trajectory_viewpoints(trajectory_viewpoints):
    """Helper function to extract trajectory viewpoints."""
    trajectory = []
    for viewpoint in trajectory_viewpoints:
        viewpoint_object = (viewpoint, 0, 0)
        trajectory.append(viewpoint_object)
    return trajectory

class TestPipeline(NavigationPipeline):
    def __init__(self,data_path):
        super().__init__()
        with open(data_path, 'r', encoding='utf-8') as data:
            self.data = json.load(data)

    def run(self, instruction, scan, starting_viewpoint, 
            instruction_id, goal_viewpoint,max_length):
        """Returns trajectory [(viewpoint_id, heading_rads, elevation_rads),]"""
        trajectory=[]
        for path in self.data:
            if instruction_id == path["path_id"]:
                trajectory_viewpoints = path["path"]
                trajectory = extract_trajectory_viewpoints(trajectory_viewpoints)
                break
        
        return trajectory