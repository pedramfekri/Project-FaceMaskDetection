from PIL import Image
import os, sys
from tqdm import tqdm
import os
import pathlib
from pathlib import Path


datasetPath =Path('C:/Users/eaitfat/Desktop/Dataset3class')
nonMaskPath = datasetPath/'NoMask-images1024x1024'
maskPath = datasetPath/'Mask-CMFD'
notPersonPath = datasetPath/'NotPerson'
newpath='C:/Users/eaitfat/Desktop/Dataset3class/NoMask-resized/'
newpath1='C:/Users/eaitfat/Desktop/Dataset3class/Mask-resized/'
newpath2='C:/Users/eaitfat/Desktop/Dataset3class/NotPerson-resized/'


def resize():
    for subject in tqdm(list(notPersonPath.iterdir()), desc='notPerson photos'):
        for item in os.listdir( subject ):
            itm= str(item)
            sub = str(subject)
            fullpath = sub + "\\" + itm
            if os.path.isfile(fullpath):
                im = Image.open(fullpath)
                #f, e = os.path.splitext(fullpath)
                imResize = im.resize((224,224))
                imResize.save(newpath2+ itm+ '.jpg', 'JPEG', quality=90)
                print("image size: ",im.size)
                print("new size: ",imResize.size)


resize()