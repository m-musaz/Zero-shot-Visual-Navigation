import json

# Load JSON file
with open("R2R_test.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Check if the data is a list and return its length
if isinstance(data, list):
    print(f"Number of JSON objects: {len(data)}")
else:
    print("The JSON file does not contain a list.")
