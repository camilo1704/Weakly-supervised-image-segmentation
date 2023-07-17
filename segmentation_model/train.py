import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from dataset import SegmentationDataSet
from tools import read_files_in_folder, append_metrics
from os.path import join
import torch
import json

import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, Text


def train_model(dataloaders:Dict, params:Dict):
    """
    Trains a unet model for binary segmentation.
    Expects a dataloader with keys ["train", "val"] and values the DataLoader for each set.
    The params dict should have as keys ["model_name", "encoder_weights", "device", "run_files"]
    The best models and metrics are saved in run_files/model and run_files/metrics paths.
    """
    model = smp.Unet(
        encoder_name=params.model_name, 
        encoder_weights=params.encoder_weights, 
        classes=1, 
        activation="sigmoid",
    )
    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
        smp.utils.metrics.Recall(threshold=0.5),
        smp.utils.metrics.Precision(threshold=0.5),
    ]

    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=0.0001),
    ])

    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=params.device,
        verbose=True,
    )
    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=params.device,
        verbose=True,
    )
    max_score = 0
    model_path = join(params.run_files, "model")
    metrics_val = {}
    metrics_train = {}
    
    for i in range(0, 15):
        
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(dataloaders["train"])
        valid_logs = valid_epoch.run(dataloaders["val"])
        append_metrics(metrics_train, train_logs)
        append_metrics(metrics_val, valid_logs)
        # do something (save model, change lr, etc.)
        print(valid_logs['iou_score'])
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, join(model_path,'best_model_'+str(i)+'.pt'))
            
        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
    metrics_path = join(params.run_files, "metrics")

    with open(join(metrics_path, "train.json"), 'w') as f:
        json.dump(metrics_train, f)
    with open(join(metrics_path, "val.json"), 'w') as f:
        json.dump(metrics_val, f)


def get_dataloaders(dataset_root_path:Text, batch_size:int=16, num_workers:int=16):
    """
    Returns dict with train and val dataloaders.
    """
    train_transform = A.Compose(
    [   A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2,p=0.5),
        A.Sharpen(alpha=(0.1, 0.2), lightness=(0.1, 1.0),p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        ToTensorV2(),
    ])
    val_transform = A.Compose(
    [  ToTensorV2(),
    ])
    train_dataset = SegmentationDataSet(read_files_in_folder(join(dataset_root_path, "train", "images")), transforms= train_transform)
    val_dataset = SegmentationDataSet(read_files_in_folder(join(dataset_root_path, "val", "images")),transforms=val_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return {
            "train": train_dataloader,
            "val": valid_dataloader
            }


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Training segmentation model", add_help=True)
    parser.add_argument("--dataset_path", type=str, help="dataset path")   
    parser.add_argument("--model_name", type=str, default="efficientnet-b0", help="model name")
    parser.add_argument("--encoder_weights", type=str, default="imagenet", help="pretrained weigths")  
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")  
    parser.add_argument("--device", type=str, default="cuda", help="device")  
    parser.add_argument("--num_workers", type=int, default=8, help="num workers")
    parser.add_argument("--save_best", type=bool, default=True, help="save best model")
    parser.add_argument("--run_files", type=str, default=True, help="path to save run files")
    args = parser.parse_args()

    dataloaders = get_dataloaders(args.dataset_path, args.batch_size, args.num_workers)
    train_model(dataloaders, args)

    