# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import requests

inputs = {'input': 'https://raw.githubusercontent.com/MTailorEng/mtailor_mlops_assessment/main/n01440764_tench.jpeg'}

res = requests.post('http://localhost:8000/', json = inputs)

print(res.json())