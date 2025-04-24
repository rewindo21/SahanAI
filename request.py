import requests

url = "http://127.0.0.1:8000/process-text/"
data = {"text": "سلام عالی بود"} 

response = requests.post(url, data=data)
print(response.text)