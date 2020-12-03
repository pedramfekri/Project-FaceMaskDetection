import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import pathlib
from pathlib import Path
import sys
import cv2

root_path = '../Dataset/'
dir = 'Dataset-3Class-Balanced'


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
            'label': 0
    }, ignore_index=True)

for imgPath in tqdm(list(nonMaskPath.iterdir()), desc='non mask'):
    parent_directory, directory_name = os.path.split(imgPath)
    parent_parent_directory, parent_directory_name = os.path.split(parent_directory)
    image = parent_directory_name + '\\' + directory_name
    ImageDF = ImageDF.append({
            'image': image,
            'label': 1
    }, ignore_index=True)

for imgPath in tqdm(list(NotPersonPath.iterdir()), desc='notPerson'):
    parent_directory, directory_name = os.path.split(imgPath)
    parent_parent_directory, parent_directory_name = os.path.split(parent_directory)
    image = parent_directory_name + '\\' + directory_name
    ImageDF = ImageDF.append({
            'image': image,
            'label': 2
    }, ignore_index=True)

ImageDF.to_pickle('images_DF.pickle')