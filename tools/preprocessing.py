from typing import Text, List, Tuple
from PIL import Image
import numpy as np
from os.path import isfile, join
from os import listdir
from PIL.Image import Image as PIL_image

def generate_classification_model_input(dataset_root_path:Text, processed_dataset_path:Text, img_size:int=256):

    set_names = ["train", "val", "test"]
    for set_name in set_names:

        set_root_path = join(dataset_root_path, set_name)
        set_images = read_files_in_folder(join(set_root_path, "images"))
        for image_path in set_images:
            pil_img, label = read_image_label(join(set_root_path,"images", image_path))
            bboxes = [process_yolo_bboxes(box, pil_img.size) for box in label]
            crop_and_save_images(pil_img, bboxes, join(processed_dataset_path, set_name), image_path, img_size)

    

def crop_and_save_images(pil_img:PIL_image, bboxes_coords:List, save_path:Text, save_name:Text,  img_size:int):
     
     h,w = pil_img.size
     h_steps = h//img_size
     w_steps = w//img_size
     for h_i in range(h_steps):
        for w_i in range(w_steps):
            bbox_to_crop = (img_size*h_i, img_size*w_i, img_size*(h_i+1), img_size*(w_i+1))
            cropped_img = pil_img.crop(bbox_to_crop)
            label_present = False
            for box in bboxes_coords:
                _, overlap_perc = calculate_overlapped_area(box, bbox_to_crop)
                if overlap_perc>0.05:
                    label_present = True
                    break
            img_save_name = save_name.split(".")[0]
            img_save_name+= "_"+str(h_i)+"_"+str(w_i)
            img_save_name+="_1.jpg" if label_present else "_0.jpg"
            cropped_img.save(join(save_path, img_save_name))

def calculate_overlapped_area(bbox1, bbox2):
    """
    calculates area and % of overlapped area. 
    The bbox input format should be [x1,y1,x2,y2]
    """
    x1p = max(bbox1[0], bbox2[0])
    y1p = max(bbox1[1],bbox2[1])
    x2p = min(bbox1[2], bbox2[2])
    y2p = min(bbox1[3],bbox2[3])
    area = max(0, x2p-x1p)*max(0,y2p-y1p)
    area_bbox1 = max(0, bbox1[2]-bbox1[0])*max(0,bbox1[3]-bbox1[1])
    area_bbox2 = max(0, bbox2[2]-bbox2[0])*max(0,bbox2[3]-bbox2[1])
    mixed_area = area_bbox1+area_bbox2
    overlap_perc = 2*area/mixed_area if mixed_area>0 else 0

    return area, overlap_perc


def read_image_label(img_path:Text)->Tuple:
    pil_img = Image.open(img_path).convert('RGB')
    label_path = img_path.replace("jpg", "txt").replace("images", "labels")
    file = open(label_path, "r",encoding='utf8')
    bboxes = file.readlines()
    file.close()
    return pil_img, bboxes


def process_yolo_bboxes(bboxes_arr:List, img_size:Tuple)->np.ndarray:
    bbox = bboxes_arr.replace("\n","").split(" ")
    x_l, y_l = img_size
    
    x_center = float(bbox[1])*x_l
    y_center = float(bbox[2])*y_l
    r_x = x_l*float(bbox[3])/(2)
    r_y = y_l*float(bbox[4])/(2)
    x_min = int(x_center-r_x)
    y_min = int(y_center-r_y)
    x_max = int(x_center+r_x)
    y_max = int(y_center+r_y)
    return [ max(0, x_min), max(0,y_min), min(x_max,x_l), min(y_max, y_l)]

def read_files_in_folder(folder_path:Text)->List:
        folder_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
        return folder_files

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=True)

    parser.add_argument("--dataset_path", type=str, help="dataset path")
    parser.add_argument("--processed_dataset_path", type=str, help="processed dataset path")
    parser.add_argument("--img_size", default=256, type=int, help="cropped img size")
    args = parser.parse_args()

    generate_classification_model_input(args.dataset_path, args.processed_dataset_path)
