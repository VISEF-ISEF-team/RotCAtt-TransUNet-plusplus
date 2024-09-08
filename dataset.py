from torch.utils.data import Dataset
import numpy as np
import torch
from scipy.ndimage import zoom
import SimpleITK as sitk


# You can either choose between 2 CustomDataset classes for your own data structure

class CustomDataset(Dataset):
    def __init__(self, num_classes, image_paths, label_paths, img_size):
        '''
        Args:
            num_classes (int): Number of classes.
            image_paths (str): Image file paths.
            label_paths (str): Label file paths.
            img_size    (int): Training image size
        
        Note:
            Make sure to process the data into this structures
            <dataset name>
            ├── p_images
            |   ├── 0001_0001.npy
            │   ├── 0001_0002.npy
            │   ├── 0001_0003.npy
            │   ├── ...
            |
            └── p_labels
                ├── 0001_0001.npy
                ├── 0001_0002.npy
                ├── 0001_0003.npy
                ├── ...     
        '''
        self.num_classes = num_classes
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.img_size = img_size
        self.length = len(image_paths)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        image = np.load(self.image_paths[index])
        label = np.load(self.label_paths[index])
        
        x, y = image.shape
        if x != self.img_size and y != self.img_size:
            image = zoom(image, (self.img_size / x, self.img_size / y), order=0)
            label = zoom(label, (self.img_size / x, self.img_size / y), order=0)
        
        encoded_label = np.zeros( (self.num_classes, ) + label.shape)
        for i in range(self.num_classes): 
            encoded_label[i][label == i] = 1
        
        return image, encoded_label
    
    

# This CustomDataset2 is used for Synapse dataset preprocessed by TransUNet authors
class CustomDataset2(Dataset):
    def __init__(self, num_classes, case_paths, img_size):
        '''
        Args:
            num_classes (int): Number of classes.
            case_path   (str): Case file paths (including image and label).
            img_size    (int): Training image size
        
        Note:
            Make sure to process the data into this structures
            <dataset name>
            ├── 0001_0001.npz
            ├── 0001_0002.npz
            ├── 0001_0003.npz
            ├── ...   
        '''
        self.num_classes = num_classes
        self.case_paths = case_paths
        self.img_size = img_size
        self.length = len(case_paths)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        data = np.load(self.case_paths[index])
        image, label = data['image'], data['label']
        
        x, y = image.shape
        if x != self.img_size and y != self.img_size:
            image = zoom(image, (self.img_size / x, self.img_size / y), order=0)
            label = zoom(label, (self.img_size / x, self.img_size / y), order=0)
        
        encoded_label = np.zeros( (self.num_classes, ) + label.shape)
        for i in range(self.num_classes): 
            encoded_label[i][label == i] = 1
            
        return torch.tensor(image).to(torch.float32), encoded_label