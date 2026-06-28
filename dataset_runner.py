import json
import argparse
from pipelines.test_pipeline import TestPipeline
from pipelines.pipeline_graph_rotations import RotationsPipeline


class DatasetRunner:
    def __init__(self,pipeline,output_file):
        self.pipeline = pipeline
        self.output_file=output_file
    def _load_dataset(self, dataset_path: str) :
        with open(dataset_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    
    def log_entry(self, trajectory, instruction_id):
        entry = {
            "instr_id": instruction_id,
            "trajectory": trajectory  # Assuming trajectory is a list of (viewpoint_id, heading_rads, elevation_rads)
        }

        # Append to the output file
        try:
            with open(self.output_file, 'r+', encoding='utf-8') as file:
                try:
                    existing_data = json.load(file)
                    if not isinstance(existing_data, list):
                        existing_data = []  # Ensure it's a list
                except json.JSONDecodeError:
                    existing_data = []  # Handle empty or corrupted JSON file

                existing_data.append(entry)
                file.seek(0)  # Move to the beginning of the file
                json.dump(existing_data, file, indent=4)
                file.truncate()  # Remove any excess data
                print(f"Ran instruction number: {instruction_id}")
        except FileNotFoundError:
            with open(self.output_file, 'w', encoding='utf-8') as file:
                json.dump([entry], file, indent=4)
        


    def run_split(self,dataset_path):
        #runs agent on each instruction
        paths_dataset=self._load_dataset(dataset_path)
        for path in paths_dataset:
            for index, instruction in enumerate(path["instructions"]):
                max_length=2*len(path["path"])
                trajectory = self.pipeline.run(instruction, path["scan"],path["path"][0],path['path_id'],path["path"][-1],max_length)
                instruction_id = f"{path['path_id']}_{index}"
                self.log_entry(trajectory,instruction_id)

def run_dataset():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", help="The method you want to run")
    parser.add_argument("--split", help="The dataset split you want to run on")
    parser.add_argument("--output", default="test", help="Output file name (without extension)")
    args = parser.parse_args()
    
    method = args.method
    split = args.split
    
    valid_methods = {"test", "graph_rotations"}
    valid_splits = {"train", "test", "val_seen", "val_unseen"}
    
    assert method in valid_methods, f"Invalid method: {method}. Must be one of {valid_methods}"
    assert split in valid_splits, f"Invalid split: {split}. Must be one of {valid_splits}"
    assert args.output, "Output file name must be provided"

    output_path = f"src/driver/{args.output}.json"
    path="src/driver/data/task/R2R_%s.json" % split
    pipeline=None
    if method == "test":
        pipeline=TestPipeline(path)
    elif method == "graph_rotations":
        pipeline=RotationsPipeline()

    runner=DatasetRunner(pipeline, output_path)
    
    runner.run_split(path)

if __name__ == "__main__":
    run_dataset()




