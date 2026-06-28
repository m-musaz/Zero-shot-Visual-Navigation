import json

def main():
    with open("data/task/R2R_test.json","r") as file:
        dataset=json.load(file)        
    with open("correct_edges_rotations.json","r") as file:
        explored_paths=json.load(file)

    instr_ids = [int(item["instr_id"].split("_")[0]) for item in explored_paths]

    results=[]
    for entry in dataset:
        path_id= entry.get("path_id")
        if path_id in instr_ids:
            results.append(entry)
    
    with open("data/task/R2R_test_explored.json","w") as file:
        json.dump(results,file,indent=4)


if __name__ == "__main__":
    main()