import json
from PIL import Image
import os

with open("parameter_dict.json","r") as f:
    data = json.load(f)
    

path = data["data_path"]
all_paths = []
for r,d,f in os.walk(path):
    files = [os.path.join(r,file) for file in f]
    