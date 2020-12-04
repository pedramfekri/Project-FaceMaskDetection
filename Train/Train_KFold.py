import Model.ResNet as res
from Dataset import customDataLoader
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from Preprocessing import Df_Image
import torchvision.transforms as transforms
from sklearn.model_selection import KFold
import pandas as pd


num_epochs = 20
num_classes = 3
learning_rate = 0.001

# torch.cuda.set_device(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())


root_path = '../Dataset/'
model_path = 'FinalResNet.pt'
# model_path = 'FinalCNN.pt'

dir = 'Dataset-3Class-Balanced/'

df = Df_Image.dataFrameMaker()
x = df.iloc[:,0]
y = df.iloc[:, 1]

fold = 10

kf = KFold(n_splits=fold, shuffle=True)
kf.get_n_splits(x)

dir = 'Dataset-3Class-Balanced/'

df_loss = pd.DataFrame()
df_acc = pd.DataFrame()

composed = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                     std=[0.5, 0.5, 0.5]),
                                ])

k = 0
for train_index, test_index in kf.split(x):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    X_train = X_train.to_frame()
    X_test = X_test.to_frame()

    y_train = y_train.to_frame()
    y_test = y_test.to_frame()

    # print(type(y_train))
    test = pd.concat([X_test, y_test], axis=1, sort=False, ignore_index=True)
    data_test = customDataLoader.CustomDataLoader(root_path+dir, test, composed)
    dataLoader_test = torch.utils.data.DataLoader(data_test, batch_size=32, shuffle=True, drop_last=False, num_workers=0)

    train = pd.concat([X_train, y_train], axis=1, sort=False, ignore_index=True)
    data_train = customDataLoader.CustomDataLoader(root_path + dir, train, composed)
    dataLoader_train = torch.utils.data.DataLoader(data_train, batch_size=32, shuffle=True, drop_last=False, num_workers=0)

    model = res.ResNet18()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_step = len(train)
    loss_list = []
    acc_list = []

    k = k + 1

    for epoch in range(num_epochs):
        for i, data in enumerate(dataLoader_train):
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
            acc_list.append(correct / total)
            if (i + 1) % 10 == 0:
                print('Fold [{}/{}], Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(k, fold, epoch + 1, num_epochs, i + 1,
                                                                                                          total_step,
                                                                                                loss.item(), (correct / total) * 100))

    df_l = pd.DataFrame(loss_list)
    df_loss = pd.concat([df_loss, df_l], axis=1, sort=False, ignore_index=True)

    df_a = pd.DataFrame(acc_list)
    df_acc = pd.concat([df_acc, df_a], axis=1, sort=False, ignore_index=True)

    model.eval()
    check = 1
    with torch.no_grad():
        correct = 0
        total = 0
        for data in dataLoader_test:
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

            print('Fold [{}/{}], Test Accuracy of the model on the {} test images: {} %'
                  .format(k, fold, total, (correct / total) * 100))

        print(' \n<<<<Class 0: Mask, Class 1: No Mask, Class 2: Not a Person>>>>')
        print('Classification Report:')
        from sklearn.metrics import classification_report

        print(classification_report(AllLabels, AllPredictions))
        print('Confusion Matrix:')
        from sklearn.metrics import confusion_matrix

        print(confusion_matrix(AllLabels, AllPredictions))


# torch.save(model, model_path)

plt.figure()


plt.subplot(2, 1, 1)
df_acc = df_acc.cumsum()
df_acc.plot()
# plt.legend(('validation accuracy', 'training accuracy'))
plt.title('training accuracy - each point is related to one batch')
plt.grid(True)


plt.subplot(2, 1, 2)
df_loss = df_loss.cumsum()
df_loss.plot()
plt.legend('loss')
plt.title('loss')
plt.grid(True)

plt.show()




