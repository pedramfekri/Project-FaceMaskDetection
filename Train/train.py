import Model.ResNet as res
import Dataset.ImageLoader as ds
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import Model.SimpleCNN as SCNN


num_epochs = 20
num_classes = 3
learning_rate = 0.001

# torch.cuda.set_device(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())


model = res.ResNet18()
# model = SCNN.SimpleCNN()

model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

root_path = '../Dataset/'
model_path = 'FinalResNet.pt'
# model_path = 'FinalCNN.pt'

dir = 'Dataset-3Class-Balanced'

train, test, val = ds.load_data(root_path, dir, 0.3, 0.1, 32)

total_step = len(train)
loss_list = []
acc_list = []
acc_list_val = []

for epoch in range(num_epochs):
    for i, data in enumerate(train):
        # forward
        images, labels = data[0].to(device), data[1].to(device)
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

        if (i + 1) % 10 == 0:
            correct_v = 0
            total_v = 0
            for dataVal in val:
                images_v, labels_v = dataVal[0].to(device), dataVal[1].to(device)
                outputs = model(images_v)
                _, predicted = torch.max(outputs.data, 1)
                correct_v += (predicted == labels_v).sum().item()
                total_v += labels_v.size(0)

            acc_list_val.append(correct_v / total_v)
            acc_list.append(correct / total)
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}% ,Validation Accuracy: {:.2f}%'.format(epoch + 1, num_epochs, i + 1, total_step,
                                                                                            loss.item(), (correct / total) * 100, (correct_v / total_v) * 100))


torch.save(model, model_path)

plt.figure()

x = np.linspace(1, len(acc_list_val), len(acc_list_val))
plt.subplot(2, 1, 1)
plt.plot(x, acc_list_val, 'r--', x, acc_list, 'b--')
plt.legend(('validation accuracy', 'training accuracy'))
plt.title('training accuracy')
plt.grid(True)

x = np.linspace(1, len(loss_list), len(loss_list))
plt.subplot(2, 1, 2)
plt.plot(x, loss_list)
plt.legend('loss')
plt.title('loss')
plt.grid(True)

plt.show()


model.eval()
check = 1
with torch.no_grad():
    correct = 0
    total = 0
    for data in test:
        images, labels = data[0].to(device), data[1].to(device)
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

        print('Test Accuracy of the model on the {} test images: {} %'
        .format(total, (correct / total) * 100))

    print(' \n<<<<Class 0: Mask, Class 1: No Mask, Class 2: Not a Person>>>>')
    print('Classification Report:')
    from sklearn.metrics import classification_report
    print(classification_report(AllLabels, AllPredictions))
    print('Confusion Matrix:')
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(AllLabels, AllPredictions))


