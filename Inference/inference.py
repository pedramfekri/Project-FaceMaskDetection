import Model.ResNet as res
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as td
import numpy as np


model_path = 'D:/OneDrive/Uni/PhD/Intro-to-AI/Project/Project-FaceMaskDetection/Train/'
model_name = 'FinalResNet.pt'


root_path = 'D:/OneDrive/Uni/PhD/Intro-to-AI/Project/Project-FaceMaskDetection/Dataset/'
dir = 'Dataset-3Class-Balanced'

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
dataset = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True, drop_last=False, num_workers=0)


model = torch.load(model_path + model_name,map_location=torch.device('cpu'))
device="cpu"
model.to(device)

check = 1
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in dataset:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if check == 1:
            AllLabels = labels.cpu().numpy()
            AllPredictions = predicted.cpu().numpy()
            check = 0
        else:
            AllLabels = np.append(AllLabels, labels.cpu().numpy())
            AllPredictions = np.append(AllPredictions, predicted.cpu().numpy())
        print('Test Accuracy of the model on the {}/{} test images: {} %'
        .format(total, len(data), (correct / total) * 100))

    print('\n<<<<Class 0: Mask, Class 1: No Mask, Class 2: Not a Person>>>>')
    print('Classification Report:')
    from sklearn.metrics import classification_report
    print(classification_report(AllLabels, AllPredictions))
    print('Confusion Matrix:')
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(AllLabels, AllPredictions))
