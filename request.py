import requests

url = "https://sahanai.liara.run/process-text/"
data = {"text": "سلام عالی بود"} 

response = requests.post(url, data=data)
print(response.text)