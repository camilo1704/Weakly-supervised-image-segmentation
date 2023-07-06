from typing import Text, List, Tuple
from PIL import Image
import numpy as np
from os.path import isfile, join
from os import listdir
import PIL.Image.Image as PIL_image

def generate_classification_model_input(dataset_root_path:Text, img_size:int=256):


    train_root_path = join(dataset_root_path, "train")
    val_root_path = join(dataset_root_path, "val")
    test_root_path = join(dataset_root_path, "test")

    train_images = read_files_in_folder(join(train_root_path, "images"))
    test_images = read_files_in_folder(join(test_root_path, "images"))
    val_images = read_files_in_folder(join(val_root_path, "images"))

    for image_path in train_images:
         img, label = read_image_label(image_path)


def crop_and_save_images(pil_img:PIL_image, bboxes_coords:List, save_path:Text,img_size:int):
     
    
def read_image_label(img_path:Text)->Tuple(PIL_image, List):
    pil_img = Image.open(img_path).convert('RGB')
    label_path = img_path.replace("jpg", "txt").replace("images", "labels")
    file = open(label_path, "r")
    bboxes = file.readlines()
    file.close()
    return pil_img, bboxes


def process_yolo_bboxes(bboxes_arr:List, img_size:Tuple)->np.ndarray:
    bboxes = []
    x_l, y_l = img_size
    for bbox in bboxes_arr:
        bbox = bbox.split(" ")
        x_center = float(bbox[1])*x_l
        y_center = float(bbox[2])*y_l
        r_x = x_l*float(bbox[3])/(2)
        r_y = y_l*float(bbox[4])/(2)
        x_min = int(x_center-r_x)
        y_min = int(y_center-r_y)
        x_max = int(x_center+r_x)
        y_max = int(y_center+r_y)
        bboxes.append([ max(0, x_min), max(0,y_min), min(x_max,x_l), min(y_max, y_l)])
    return np.array(bboxes)

def read_files_in_folder(folder_path:Text)->List:
        folder_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
        return folder_files