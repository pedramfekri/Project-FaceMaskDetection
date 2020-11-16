import Model.ResNet as res
import Dataset.ImageLoader as ds
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import Model.SimpleCNN as SCNN
import Model.SimpleCNN2 as SCNN2

num_epochs = 1
num_classes = 3
learning_rate = 0.001

# torch.cuda.set_device(0)

# model = res.ResNet18()
model = SCNN2.SimpleCNN2()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#root_path = '/home/pedram/PycharmProjects/Project-FaceMaskDetection/Dataset/'
#model_path = 'entire_model.pt'
#dir = 'Dataset-3Class-Sample'

root_path = 'D:/OneDrive/Uni/PhD/Intro-to-AI/Project/Project-FaceMaskDetection/Dataset/'
model_path = 'SimpleCNNModel.pt'
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

        if (i + 1) % 10 == 0:
            correct_v = 0
            total_v = 0
            for images_v, labels_v in val:
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
    for images, labels in test:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if check==1:
            AllLabels = labels.numpy()
            AllPredictions = predicted.numpy()
            check=0
        else:
            AllLabels = np.append(AllLabels,labels.numpy())
            AllPredictions = np.append(AllPredictions,predicted.numpy())

        print('Test Accuracy of the model on the test images: {} %'
              .format((correct / total) * 100))

    from sklearn.metrics import classification_report
    print(classification_report(AllLabels, AllPredictions))


