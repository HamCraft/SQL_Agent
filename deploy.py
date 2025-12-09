import requests

API_URL = "http://65.0.30.255/ask/"

payload = {
    "merchantId": 2006,
    "query": "What were my sales "
}

try:
    response = requests.post(API_URL, json=payload, timeout=20)
    print("Status Code:", response.status_code)
    print("Response:")
    print(response.json())
except Exception as e:
    print("Error:", e)
