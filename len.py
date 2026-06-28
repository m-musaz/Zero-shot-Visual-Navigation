import json

# Replace 'data.json' with your file path
file_path = 'rotations_output.json'

# Read the file and parse the JSON
with open(file_path, 'r') as file:
    data = json.load(file)

# Print the length of the top-level object
print("Length of JSON object:", len(data))
