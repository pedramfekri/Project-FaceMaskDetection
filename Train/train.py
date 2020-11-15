import Model.ResNet as res
import Dataset.ImageLoader as ds
import torch
import torch.nn as nn
import numpy as np

num_epochs = 4
num_classes = 3
learning_rate = 0.001

torch.cuda.set_device(0)

model = res.ResNet18()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

root_path = '/home/pedram/PycharmProjects/Project-FaceMaskDetection/Dataset/'
dir = 'Dataset-3Class-Sample'
train, test, val = ds.load_data(root_path, dir, 0.3, 0.1, 32)

total_step = len(train)
loss_list = []
acc_list = []
acc_list_val = []

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train):
        # forward
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())
        # backward & optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # train eval
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)
        if (i + 1) % 10 == 0:
            correct_v = 0
            total_v = 0
            for images_v, labels_v in val:
                outputs = model(images_v)
                _, predicted = torch.max(outputs.data, 1)
                correct_v += (predicted == labels_v).sum().item()
                total_v += labels_v.size(0)

            acc_list_val.append(correct_v / total_v)
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}% ,Validation Accuracy: {:.2f}%'.format(epoch + 1, num_epochs, i + 1, total_step,
                                                                                            loss.item(), (correct / total) * 100, (correct_v / total_v) * 100))


model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print('Test Accuracy of the model on the 10000 test images: {} %'
        .format((correct / total) * 100))
