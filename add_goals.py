import json



def main():
    # Load rotations_output.json (a list of objects)
    with open("rotations_output.json", "r") as f:
        rotations = json.load(f)

    # Load data/r2r_test.json (a list of objects with path_id and a path list)
    with open("data/task/R2R_test.json", "r") as f:
        test_paths = json.load(f)

    # Build a dictionary for fast lookup by path_id
    path_dict = {p["path_id"]: p for p in test_paths if "path_id" in p}

    results=[]

    for rotation in rotations:
        instruction_id = rotation.get("instr_id")
        if not instruction_id:
            continue
        key = instruction_id.split("_")[0]
        print(f"Processing instruction_id: {instruction_id}, key: {key}")
        # Get the corresponding test path object by matching path_id
        test_obj = path_dict.get(int(key))
        if not test_obj:
            print(f"Warning: No matching path found for instruction_id: {key}")
            continue

        # Get the last viewpoint from the trajectory object.
        # Each trajectory element is a tuple/list [viewpoint, heading_rad, elevation_rad]
        trajectory = rotation.get("trajectory", [])
        print(f"Trajectory length: {len(trajectory)}")

        if not trajectory:
            continue
        print(f"Trajectory: {trajectory}")
        last_traj_view = trajectory[-1][0]
        print(f"Last trajectory view: {last_traj_view}")
        # Get the last viewpoint from the path object.
        # The path list may contain just view points, or dicts containing a "viewpoint" key.
        path = test_obj.get("path", [])
        if len(trajectory) <= (2*len(path)):
            #add goal viewpoint to the end
            goal=path[-1]
            trajectory.append([goal,0.0,0.0])
        traj_obj={"instr_id":instruction_id,"trajectory":trajectory}
        results.append(traj_obj)
    
    
    with open("goals_rotations_output.json","w",encoding="utf-8") as file:
        json.dump(results,file,indent=4)

if __name__ == "__main__":
    main()





       
