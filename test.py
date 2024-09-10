import requests


res = requests.post('http://localhost:3000/synthesize', json={
    "text": "初めて神戸に移り住んだ時に地元の人に教わった「阪急はオシャレして乗らなあかん。阪神はスリッパで乗っていい。JRは早い。」、好きすぎていまだに東京の人に説明するとき使ってる。"
})
with open('output.wav', 'wb') as f:
    f.write(res.content)