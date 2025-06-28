import requests

# Test script to send a POST request to the Flask API
# Make sure the Flask app is running before executing this script
url = "http://192.168.1.4:5000/predict" 
payload = {"review": "I loved this movie!"}

response = requests.post(url, json=payload)
if response.status_code == 200:
    print(response.json())
else:
    print(f"Error: {response.status_code}, {response.json()}")