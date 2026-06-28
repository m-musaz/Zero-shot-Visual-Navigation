import requests

# url = "http://host.docker.internal:3000/get_history"  # Change if using Docker
url = "http://host.docker.internal:3000/summarize_desc"  # Change if using Docker
data = {
    "viewpoint_id": "test_node",
    "description": "A large hallway with white marble flooring, a wooden staircase on the right, and a chandelier hanging above."
}

print(f"Sending manual request to {url}...")
response = requests.post(url, json=data)

print("Response Status:", response.status_code)
print("Response Data:", response.text)
