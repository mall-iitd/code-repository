# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 10:51:21 2019 by Attila Lengyel - attila@lengyel.nl
"""

import os
import numpy as np

from PIL import Image

from torchvision.datasets import Cityscapes

def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles
    
def getListOfFiles_mask(dirName):
    # create a list of file and sub directories 
    allFiles=[]
    for entry in range(0,len(dirName)):
        if 'labelIds' in dirName[entry] :
            #print(fullPath)
            allFiles.append(dirName[entry])
                
    return allFiles



class ACDCNightDataset(Cityscapes):
    
    voidClass = 19

    # Convert ids to train_ids
    id2trainid = np.array([label.train_id for label in Cityscapes.classes if label.train_id >= 0], dtype='uint8')
    id2trainid[np.where(id2trainid==255)] = voidClass

    # Convert train_ids to ids
    trainid2id = np.arange(len(id2trainid))[np.argsort(id2trainid)]
    
    # Convert train_ids to colors
    mask_colors = [list(label.color) for label in Cityscapes.classes if label.train_id >= 0 and label.train_id <= 19]
    mask_colors.append([0,0,0])
    mask_colors = np.array(mask_colors)
    
    # List of valid class ids
    validClasses = np.unique([label.train_id for label in Cityscapes.classes if label.id >= 0])
    validClasses[np.where(validClasses==255)] = voidClass
    validClasses = list(validClasses)
    
    # Create list of class names
    classLabels = [label.name for label in Cityscapes.classes if not (label.ignore_in_eval or label.id < 0)]
    classLabels.append('void')
    
    def __init__(self, root, split='val', transforms=None):
        self.transforms = transforms
        
        self.root = root
        self.split = split
        
        if split == 'val':
            self.imgs_root = os.path.join(root,'rgb_anon_trainvaltest/rgb_anon/night/val/')
            self.masks_root = os.path.join(root,'gt_trainval/gt/night/val/')
            #self.masks = [mask for mask in list(sorted(os.listdir(self.masks_root))) if 'labelIds' in mask]
            self.masks = getListOfFiles(self.masks_root)
            self.masks = getListOfFiles_mask(sorted(self.masks))
        else:
            self.imgs_root = os.path.join(root,'rgb_anon_trainvaltest/rgb_anon/night/test/')

        #self.imgs = list(sorted(os.listdir(self.imgs_root)))
        self.imgs = sorted(getListOfFiles(self.imgs_root))
        print(self.imgs[0])
        #print(self.masks[0])
        #exit()

        if split=='val':
            assert len(self.imgs) == len(self.masks), 'Number of images and masks must be equal'
        
        assert len(self.imgs) != 0, 'No images found'
    
    def __getitem__(self, idx):
        # Define image and mask path
        img_path = os.path.join(self.imgs_root, self.imgs[idx])
        image = Image.open(self.imgs[idx]).convert('RGB')

        if self.split == 'val':
            #mask_path = os.path.join(self.masks_root, self.masks[idx])
            target = Image.open(self.masks[idx])
        else:
            target = None
            
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        if self.split == 'val':
            target = self.id2trainid[target] # Convert class ids to train_ids and then to tensor: SLOW
            return image, target, img_path
        else:
            return image, img_path
        
    def __len__(self):
        return len(self.imgs)
