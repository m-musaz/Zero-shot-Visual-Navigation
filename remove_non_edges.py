import json




def get_scan(instr_id:str,dataset):
    path_id=int(instr_id.split("_")[0])
    dataset_scan=""
    for path in dataset:
        dataset_path_id=path.get("path_id")
        if path_id == dataset_path_id:
            print("foudn path")
            dataset_scan=path.get("scan")
            break
    assert dataset_scan != "","path not found"
    return dataset_scan
    



def check_edge(previous_viewpoint,cur_viewpoint,graph):
    if previous_viewpoint == cur_viewpoint:
        return False
    prev_node = None
    cur_node_idx = None
    for idx,node in enumerate(graph):
        node_viewpoint=node.get("image_id")
        if node_viewpoint == previous_viewpoint:
            prev_node=node
        elif node_viewpoint == cur_viewpoint:
            cur_node_idx=idx
        if prev_node and cur_node_idx:
            break

    assert prev_node is not None,"Prev node not found"
    assert cur_node_idx is not None,"Cur node idx not found"
    edges=prev_node.get("unobstructed")
    edge_present=edges[cur_node_idx]
    return edge_present
            


def get_trajectory(trajectory,final_index):
    return trajectory[0:final_index]

def main():
    with open("goals_rotations_output.json","r") as file:
        rotations_output=json.load(file)

    with open("data/task/R2R_test.json","r") as file:
        dataset=json.load(file)
    results=[]
    for rotation in rotations_output:
        trajectory=rotation.get("trajectory")
        instr_id=rotation.get("instr_id")
        scan=get_scan(instr_id,dataset)
        path=f"data/connectivity/{scan}_connectivity.json"
        with open(path,"r") as file:
            graph=json.load(file)
        results_obj={"instr_id":instr_id,"trajectory":trajectory}
        for i in range(1,len(trajectory)):
            previous_viewpoint=trajectory[i-1][0]
            cur_viewpoint=trajectory[i][0]
            edge_present=check_edge(previous_viewpoint,cur_viewpoint,graph)
            if not edge_present:
                new_trajectory=get_trajectory(trajectory,i-1)
                results_obj={"instr_id":instr_id,"trajectory":new_trajectory}
                break
        results.append(results_obj)
    with open("correct_edges_rotations.json","w",encoding="utf-8") as file:
        json.dump(results,file,indent=4)

if __name__ == "__main__":
    main()