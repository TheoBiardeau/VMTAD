import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
from torchvision import transforms as T
from matplotlib import pyplot as plt 


def folder_sort(path):
    path_split = os.path.normpath(path).split(os.path.sep)[-1]
    fold_nb = int(path_split.split("_")[1])
    return(fold_nb)
    
def image_sort(path):
    path_split = os.path.normpath(path).split(os.path.sep)[-1]
    fold_nb = int(path_split.split("_")[1].split(".")[0])
    return(fold_nb)
    
class ImageSequenceDataset(Dataset):
    def __init__(self, root_dirs, sequence_length, transforms = T.ToTensor()):
        """
        root_dirs: Liste de dossiers contenant les images
        sequence_length: Longueur des séquences d'images à charger
        """

        folders_path = sorted(glob.glob(os.path.join(root_dirs, "*")), key=folder_sort)
        
        self.image_paths = np.array([])
        self.sequence_length = sequence_length
        self.transforms = transforms
        
        for folder_path in folders_path : 
            image_paths = sorted(glob.glob(os.path.join(folder_path, "*.jpg")), key=image_sort)
            nb_images = len(image_paths)
            remainder = len(image_paths) % sequence_length
            image_paths = image_paths[:(nb_images-remainder)]
            self.image_paths = np.append(self.image_paths, image_paths)
            
    def __len__(self):
        return self.image_paths.shape[0] // self.sequence_length 

    def __getitem__(self, idx):
        idx = idx * self.sequence_length 
        images = [Image.open(self.image_paths[i]).convert('RGB') for i in range (idx, idx+self.sequence_length)]
        
        state = torch.get_rng_state()
        images_transformed = []
        for img in images : 
            torch.set_rng_state(state)
            images_transformed.append(self.transforms(img))
        
        folder = os.path.split(os.path.split(self.image_paths[idx])[0])[1]
        
        return folder, torch.stack(images_transformed)

class AdImageSequenceDataset(Dataset):
    def __init__(self, root_dirs, transforms_image = T.ToTensor(), transforms_labels = T.ToTensor()):
        """
        root_dirs: Liste de dossiers contenant les images
        sequence_length: Longueur des séquences d'images à charger
        """

        folder_images_path = sorted(glob.glob(os.path.join(root_dirs, "AD", "*")), key=folder_sort)
        
        self.root_dirs = root_dirs
        self.image_paths = np.array([])
        self.labels_paths = np.array([])
        self.transforms_image = transforms_image
        self.transforms_labels = transforms_labels
        
        for folder_path in folder_images_path : 
            image_paths = sorted(glob.glob(os.path.join(folder_path, "*.jpg")), key=image_sort)
            self.image_paths = np.append(self.image_paths, image_paths)
            
    def __len__(self):
        
        return self.image_paths.shape[0]
    def __getitem__(self, idx):
        
        images = Image.open(self.image_paths[idx]).convert('RGB')
        label_folder_path = os.path.split((os.path.split(self.image_paths[idx])[0]))[1]
        label_file = os.path.splitext(os.path.basename(self.image_paths[idx]))[0]
        full_path_labels = os.path.join(self.root_dirs,'AD_labels',label_folder_path, label_file+'.png')
        labels = Image.open(full_path_labels)#.convert('RGB')
        
        state = torch.get_rng_state()
        torch.set_rng_state(state)
        
        images_transformed = []
        labels_transformed = []
        images_transformed.append(self.transforms_image(images))
        labels_transformed.append(self.transforms_labels(labels))
        
        folder = os.path.split(os.path.split(self.image_paths[idx])[0])[1]
        
        
        return label_file, folder, images_transformed[0], labels_transformed[0]  # Retourner une séquence d'images en tant que tensor