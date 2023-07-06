from torch.utils.data import Dataset
import pathlib 
from typing import List, Text
from PIL import Image
import numpy as np


class ClassificationDataSet(Dataset):
    """
    expects yolo dataset format
    """
    def __init__(self, dataset_keys: List[pathlib.Path], transforms:List=None): 
        self.dataset_keys = dataset_keys
        self.transforms = transforms
        
    def __len__(self):
        return len(self.dataset_keys)

    def __getitem__(self, index: int):
        
        sample = self.dataset_keys[index] #Image absolute path
        pil_img = self.read_image(sample)
        label = sample.split(".jpg")[0][-1]
        if self.transforms:
            pil_img = self.transforms(image=np.array(pil_img))["image"]
        
        torch_img = pil_img
        return torch_img, int(label)

    @staticmethod
    def read_image(img_path:Text):
        pil_img = Image.open(img_path).convert('RGB')
        return pil_img