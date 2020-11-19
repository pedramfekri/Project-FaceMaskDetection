import Model.ResNet as res
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as td
from PIL import Image
import numpy as np
import cv2

model_path = 'D:/OneDrive/Uni/PhD/Intro-to-AI/Project/Project-FaceMaskDetection/Train/'
model_name = 'FinalResNet.pt'

root_path = 'D:/OneDrive/Uni/PhD/Intro-to-AI/Project/Project-FaceMaskDetection/Dataset/'
dir = 'Dataset-3Class-Balanced'

data_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225]),
                                      ])
# frame_t = Image.open("/home/pedram/PycharmProjects/Project-FaceMaskDetection/Dataset/Dataset-3Class-Sample/Mask-resized/00021_Mask.jpg.jpg")
# frame_t = Image.open("/home/pedram/PycharmProjects/Project-FaceMaskDetection/Dataset/Dataset-3Class-Sample/NoMask-resized/02769.png.jpg")
# frame_t = Image.open("/home/pedram/PycharmProjects/Project-FaceMaskDetection/Dataset/Dataset-3Class-Sample/NotPerson-resized/00100.jpg.jpg")

# frame_t = Image.open("D:/OneDrive/Uni/PhD/Intro-to-AI/Project/Project-FaceMaskDetection/Dataset/Dataset-3Class-Balanced/NotPerson-resized/00010.jpg.jpg")
# frame_t = Image.open("D:/OneDrive/Uni/PhD/Intro-to-AI/Project/Project-FaceMaskDetection/Dataset/Dataset-3Class-Balanced/NoMask-resized/ffhq67955.png.jpg")
# frame_t = Image.open("D:/OneDrive/Uni/PhD/Intro-to-AI/Project/Project-FaceMaskDetection/Dataset/Dataset-3Class-Balanced/Mask-resized/00010_Mask.jpg.jpg")

cam = cv2.VideoCapture(0)
ret, frame_t = cam.read()
# frame_t = frame_t.rotate(-90)
frame_t = Image.fromarray(frame_t)

img = data_transforms(frame_t)
img = img.numpy()
img1 = np.transpose(img, [1,2,0])
plt.imshow(img1)
plt.show()

print(img.shape)
# img = np.transpose(img, (1, 0, 2))

img = img[np.newaxis, ...]
img = torch.from_numpy(img)

model = torch.load(model_path + model_name)
device = ("cpu")
model.to(device)

classes = ('mask', 'no-mask', 'not-a-person')

model.eval()
with torch.no_grad():
    outputs = model(img)
    _, predicted = torch.max(outputs.data, 1)
    print(classes[predicted])

