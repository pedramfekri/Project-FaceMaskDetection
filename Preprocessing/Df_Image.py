import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import pathlib
from pathlib import Path
import sys
import cv2
import numpy as np
from sklearn.model_selection import KFold
from Dataset import customDataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


def dataFrameMaker():
    root_path = '../Dataset/'
    dir = 'Dataset-3Class-Balanced/'


    datasetPath =Path(root_path + dir)

    maskPath = datasetPath/'Mask-resized'
    nonMaskPath = datasetPath/'NoMask-resized'
    NotPersonPath = datasetPath/'NotPerson-resized'
    print(nonMaskPath)
    print(maskPath)
    print(NotPersonPath)


    ImageDF = pd.DataFrame()


    for imgPath in tqdm(list(maskPath.iterdir()), desc='mask'):
        parent_directory, directory_name = os.path.split(imgPath)
        parent_parent_directory, parent_directory_name = os.path.split(parent_directory)
        image=parent_directory_name+'\\'+ directory_name
        ImageDF = ImageDF.append({
                'image': image,
                'label': int(0)
        }, ignore_index=True)

    for imgPath in tqdm(list(nonMaskPath.iterdir()), desc='non mask'):
        parent_directory, directory_name = os.path.split(imgPath)
        parent_parent_directory, parent_directory_name = os.path.split(parent_directory)
        image = parent_directory_name + '\\' + directory_name
        ImageDF = ImageDF.append({
                'image': image,
                'label': int(1)
        }, ignore_index=True)

    for imgPath in tqdm(list(NotPersonPath.iterdir()), desc='notPerson'):
        parent_directory, directory_name = os.path.split(imgPath)
        parent_parent_directory, parent_directory_name = os.path.split(parent_directory)
        image = parent_directory_name + '\\' + directory_name
        ImageDF = ImageDF.append({
                'image': image,
                'label': int(2)
        }, ignore_index=True)

    # ImageDF.to_pickle('images_DF.pickle')
    return ImageDF

df = dataFrameMaker()
# print(df.shape)
# print(df.head())
# print(df.iloc[:,0])
x = df.iloc[:,0]
y = df.iloc[:, 1]

kf = KFold(n_splits=10, shuffle=True)
kf.get_n_splits(x)

# print(kf)

root_path = '../Dataset/'
dir = 'Dataset-3Class-Balanced/'


composed = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                     std=[0.5, 0.5, 0.5]),
                                ])

for train_index, test_index in kf.split(x):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    X_train = X_train.to_frame()
    X_test = X_test.to_frame()

    y_train = y_train.to_frame()
    y_test = y_test.to_frame()

    # print(type(y_train))
    Test = pd.concat([X_test, y_test], axis=1, sort=False)
    # print(y_test.head())
    # print(X_test.head())
    data = customDataLoader.CustomDataLoader(root_path+dir, Test, composed)
    dataLoader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True, drop_last=False, num_workers=0)
    # print(data)
    # print(len(dataLoader))

    train = iter(dataLoader)
    a, label = next(train)
    print(len(a))
    # im = a[1].numpy()
    # print(im.shape)
    # im = np.transpose(im, [1, 2, 0])
    # print(im.shape)
    # plt.imshow(im)
    # plt.show()
#

