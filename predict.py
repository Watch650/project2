from ultralytics import YOLO

import numpy as np


model = YOLO(r'C:\Users\Asus\Desktop\scul\B2\Machine Learning and Data Mining\project2\runs\classify\train3\weights\last.pt')  # load a custom model

results = model(r'C:\Users\Asus\Desktop\scul\B2\Machine Learning and Data Mining\project2\Test Image\vegetables.png')  # predict on an image

names_dict = results[0].names

probs = results[0].probs.data.tolist()

print(names_dict)
print(probs)

print(names_dict[np.argmax(probs)])
