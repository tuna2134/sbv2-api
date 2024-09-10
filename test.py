import requests


res = requests.post('http://localhost:3000/synthesize', json={
    "text": "眠たい"
})
res.raise_for_status()
with open('output.wav', 'wb') as f:
    f.write(res.content)