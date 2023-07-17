from torch.utils.data import Dataset
import pathlib 
from typing import List, Text, Dict
from PIL import Image
import numpy as np
import torch


class SegmentationDataSet(Dataset):
    """
    Segmentation Dataset Class.
    Initialized with list of images path and list of transforms.
    Expects a yolo dataset format to find masks paths (replacing "images" with "mask" in path).
    Masks should be an npy file with boolean values (0,1).
    """
    def __init__(self, dataset_keys: List[pathlib.Path], transforms:List=None): 
        self.dataset_keys = dataset_keys
        self.transforms = transforms
        
    def __len__(self):
        return len(self.dataset_keys)

    def __getitem__(self, index: int):
        
        sample = self.dataset_keys[index] #Image absolute path
        pil_img, np_mask = self.read_image_mask(sample)
        if self.transforms:
            transformed_sample = self.transforms(image=np.array(pil_img).astype(np.uint8), mask=np_mask.astype(np.uint8))
            pil_img, np_mask = transformed_sample["image"], transformed_sample["mask"]
            
        torch_img = pil_img.to(torch.float32)
        mask =  torch.unsqueeze(transformed_sample["mask"],0)
        return torch_img, mask

    @staticmethod
    def read_image_mask(img_path:Text):
        pil_img = Image.open(img_path).convert('RGB')
        mask_path = img_path.replace("jpg", "npy").replace("images","mask")
        np_mask = np.load(mask_path)
        return pil_img, np_mask.astype(np.uint8)
    