import cv2
import time
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt


# model_path = '/home/pedram/PycharmProjects/Project-FaceMaskDetection/Train/'
# model_name = 'entire_model.pt'

model_path = 'D:/OneDrive/Uni/PhD/Intro-to-AI/Project/Project-FaceMaskDetection/Train/'
model_name = 'FinalResNet.pt'



model = torch.load(model_path + model_name)
device = "cpu"
model.to(device)

classes = ('mask', 'no-mask', 'not-a-person')


def CaptureImage(cameraPort):

    cam = cv2.VideoCapture(cameraPort)

    cv2.namedWindow("MaskDetector")

    print("Space ==> Capture, Esc ==> Exit")
    data_transforms = transforms.Compose([transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
         ])
    while True:
        ret, frame = cam.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if not ret:
            print("Failed. Try Again!")
            break

        frame_t = Image.fromarray(frame)
        # frame_t = frame_t.rotate(-90)
        # plt.imshow(frame_t)
        # plt.show()
        img = data_transforms(frame_t)
        img = img.numpy()
        img = img[np.newaxis, ...]
        img = torch.from_numpy(img)
        model.eval()
        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs.data, 1)

        detected_class = classes[predicted]
        print(detected_class)
        # print(outputs.data)
        cv2.imshow("detected_class", frame)

        k = cv2.waitKey(1)

    cam.release()
    cv2.destroyAllWindows()


CaptureImage(0)