import Model.ResNet as res
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as td
import numpy as np

model_path = 'C:\\Users\\Pedram\\PycharmProjects\\Project-FaceMaskDetection\\Train\\'
model_name = 'entire_model.pt'

root_path = 'C:\\Users\\Pedram\\PycharmProjects\\Project-FaceMaskDetection\\Dataset\\'
dir = 'Dataset-3Class-Sample'

transform_dict = {
        'src': transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
         ]),
        'tar': transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
         ])}

data = datasets.ImageFolder(root=root_path + dir, transform=transform_dict['src'])
dataset = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True, drop_last=False, num_workers=2)


model = torch.load(model_path + model_name)
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in dataset:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        print(labels)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print('Test Accuracy of the model on the 10000 test images: {} %'
        .format((correct / total) * 100))
