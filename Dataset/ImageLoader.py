import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as td


# get data from the folders in the root and assign labels to data of every folder separately.
def load_data(root_path, dir, test_split, batch_size, phase):
    transform_dict = {
        'src': transforms.Compose(
        [transforms.RandomResizedCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
         ]),
        'tar': transforms.Compose(
        [transforms.Resize(224),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
         ])}

    data = datasets.ImageFolder(root=root_path + dir, transform=transform_dict[phase])

    dataset_size = len(data)
    test_size = int(test_split * dataset_size)
    train_size = dataset_size - test_size
    
    train_dataset, test_dataset = td.random_split(data,
                                               [train_size, test_size])

    data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=2)
    data_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=2)

    return data_loader_train, data_loader_test
