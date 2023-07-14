import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
from typing import Text
from PIL import Image
from PIL.Image import Image as PIL_image
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchcam.methods import SmoothGradCAMpp,CAM, GradCAM,LayerCAM
from tools import read_files_in_folder
from os.path import join

CAM_threshold= 0.15

def generate_masks(dataset_path:Text, segmentation_dataset_path:Text, model:models, layer_name:Text):
    """
    generate masks using CAM method to extract 
    """
    sets = [ "train", "val", "test"]
    for set_name in sets:
        set_images_path = join(dataset_path, set_name)
        set_files = read_files_in_folder(set_images_path)
        print(len(set_files))
        for image_path in set_files:
            mask = get_CAM_mask(join(set_images_path, image_path), model, layer_name, CAM_threshold)
            print(join(segmentation_dataset_path, set_name,"images", image_path))
            Image.open(join(set_images_path, image_path)).save(join(segmentation_dataset_path, set_name,"images", image_path))
            np.save(join(segmentation_dataset_path, set_name, "mask", image_path.split(".")[0]), mask)


def get_CAM_mask(img_path:Text, model:models, layer_name:Text, CAM_threshold:float)->np.ndarray:
    print(img_path)
    img = read_image(img_path)
    input_tensor = normalize(resize(img, (256, 256)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    cam_extractor = GradCAM(model, target_layer=layer_name)  
    out = model(input_tensor.unsqueeze(0))
    activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
    overlay = to_pil_image(activation_map[0].squeeze(0), mode='F').resize((256,256), resample=Image.BICUBIC)
    mask = np.array(overlay) < CAM_threshold
    
    return mask


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Generate CAM Masks", add_help=True)

    parser.add_argument("--dataset_path", type=str, help="dataset path")
    parser.add_argument("--segmentation_dataset_path", type=str, help="processed dataset path")
    parser.add_argument("--layer_name", type=str, default="features.denseblock4", help="output layer CAM")
    parser.add_argument("--model_name", type=str, default="densenet121", help="model used as CAM extractor")
    parser.add_argument("--model_weights", type=str, default="/home/cuchuflito/Documents/repos/Weakly-supervised-image-segmentation/runs/2/model/densenet121_24.pt",  help="path to model weights")

    
    args = parser.parse_args()
    model =  getattr(models, args.model_name)(pretrained=False, num_classes=1,)
    model.load_state_dict(torch.load(args.model_weights))
    model.eval()
    generate_masks(args.dataset_path, args.segmentation_dataset_path, model, args.layer_name)
