import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as td
import numpy as np
import matplotlib.pyplot as plt


# get data from the folders in the root and assign labels to data of every folder separately.
def load_data(root_path, dir, test_split, val_split, batch_size, inputSizeW, inputSizeL):
    transform_dict = {
        'src': transforms.Compose(
        [transforms.Resize((inputSizeW, inputSizeL)),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5],
                              std=[0.5, 0.5, 0.5]),
         ])}

    data = datasets.ImageFolder(root=root_path + dir, transform=transform_dict['src'])

    dataset_size = len(data)
    test_size = int(test_split * dataset_size)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - (test_size + val_size)

    train_dataset, test_dataset, val_dataset = td.random_split(data,
                                               [train_size, test_size, val_size])

    data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
    data_loader_test  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
    data_loader_val   = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)

    return data_loader_train, data_loader_test, data_loader_val


# root_path = '../Dataset/'
# dir = 'Dataset-3Class-Balanced'
#
# train, test, val = load_data(root_path, dir, 0.3, 0.1, 32, 224, 224)
# print(len(train))
# print(len(test))
# print(len(val))
#
# train = iter(train)
# a, label = next(train)
# print(a[1])
# print(label[1])
# im = a[0].numpy()
# print('label')
# print(label)
# print(im.shape)
# im = np.transpose(im, [1, 2, 0])
# print(im.shape)
# plt.imshow(im)
# plt.show()
