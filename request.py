import requests
import time

url = "https://sahanai.liara.run/process-text/"
data = {"text": "سلام عالی بود"} 

start_time = time.time()
response = requests.post(url, data=data)
end_time = time.time()

print("Response:", response.text)
print("Time taken: {:.3f} seconds".format(end_time - start_time))