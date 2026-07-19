import cv2
import os
import numpy as np

dataset_path = "dataset/faces"

faces = []
labels = []
label_map = {}
current_label = 0

for person_name in os.listdir(dataset_path):

    person_folder = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_folder):
        continue

    label_map[current_label] = person_name

    for image_name in os.listdir(person_folder):

        image_path = os.path.join(person_folder, image_name)

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        faces.append(img)
        labels.append(current_label)

    current_label += 1

labels = np.array(labels)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, labels)

recognizer.save("face_trainer.yml")

with open("labels.txt", "w") as f:
    for key, value in label_map.items():
        f.write(f"{key},{value}\n")

print("Training Completed!")