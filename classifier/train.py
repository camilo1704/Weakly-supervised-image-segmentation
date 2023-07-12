from dataset import ClassificationDataSet
from torch.utils.data import DataLoader
from os.path import isfile, join
from os import listdir
from typing import Text, Dict, List
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.models as models
from metrics import calculate_accuracy, MetricMonitor
import torch.nn as nn
import torch.optim
from tqdm import tqdm


def read_files_in_folder(folder_path:Text)->List:
        folder_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
        return folder_files

def get_dataloaders(dataset_root_path:Text, batch_size:int)->Dict:

    set_names = ["train", "val", "test"]
    dataloaders = {}
    train_transform = A.Compose(
    [   A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2,p=0.5),
        A.Sharpen(alpha=(0.1, 0.2), lightness=(0.1, 1.0),p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    val_transform = A.Compose(
    [   A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    for set_name in set_names:
        set_files_path = join(dataset_root_path, set_name)
        samples = read_files_in_folder(set_files_path)
        samples = [join(set_files_path, sample_name) for sample_name in samples]
        transforms =  train_transform if set_name=="train" else val_transform
        dataset = ClassificationDataSet(samples, transforms=transforms)
        dataloaders[set_name] = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloaders

def train_model(dataloaders:Dict, params:Dict):
    model =  getattr(models, params.model)(pretrained=False, num_classes=1,)
    model = model.to(params.device)
    criterion = nn.BCEWithLogitsLoss().to(params.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    best_acc = 0
    for epoch in range(1, params.epochs + 1):
        train(dataloaders["train"], model, criterion, optimizer, epoch, params)
        best_acc=validate(dataloaders["val"], model, criterion, epoch, params,save_best=True, best_accuracy=best_acc)


def train(train_loader, model, criterion, optimizer, epoch, params):

    metric_monitor = MetricMonitor(join(params.save_run_path, "metrics", "train"))
    model.train()
    stream = tqdm(train_loader)
    for i, (images, target) in enumerate(stream, start=1):
        images = images.to(params.device, non_blocking=True)
        target = target.to(params.device, non_blocking=True).float().view(-1, 1)
        output = model(images)
        loss = criterion(output, target)
        accuracy = calculate_accuracy(output, target)
        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("Accuracy", accuracy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        stream.set_description(
            "Epoch: {epoch}. Train.      {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
        )
    metric_monitor.save_metric(epoch)

def validate(val_loader, model, criterion, epoch, params, save_best:bool=False, best_accuracy:float=0):
    metric_monitor = MetricMonitor(join(params.save_run_path, "metrics", "val"))
    model.eval()
    stream = tqdm(val_loader)
    
    with torch.no_grad():
        for i, (images, target) in enumerate(stream, start=1):
            images = images.to(params.device, non_blocking=True)
            target = target.to(params.device, non_blocking=True).float().view(-1, 1)
            output = model(images)
            loss = criterion(output, target)
            accuracy = calculate_accuracy(output, target)
            
            metric_monitor.update("Loss", loss.item())
            metric_monitor.update("Accuracy", accuracy)
            
            stream.set_description(
                "Epoch: {epoch}. Validation. {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
            )
        print(metric_monitor.get_acc())
        if metric_monitor.get_acc()>best_accuracy:
            print(metric_monitor.get_acc(), best_accuracy)
            best_accuracy=metric_monitor.get_acc()
            if save_best:
                torch.save(model.state_dict(), join(params.save_run_path, "model", params.model+"_"+str(epoch)+".pt"))
        metric_monitor.save_metric(epoch)
        return best_accuracy

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=True)

    parser.add_argument("--dataset_root_path", type=str, help="dataset root path")
    parser.add_argument("--img_size", default=256, type=int, help="cropped img size")
    parser.add_argument("--model", default="vgg16", type=str, help="pretrained model name")
    parser.add_argument("--batch_size", default=32, type=int, help="batch size")
    parser.add_argument("--epochs", default=100, type=int, help="training epochs")
    parser.add_argument("--lr", default=0.0001, type=float, help="lr")
    parser.add_argument("--num_workers", default=0, type=int, help="num workers")
    parser.add_argument("--device", default="cuda", type=str, help="device")
    parser.add_argument("--save_run_path", default="./runs", type=str, help="save path to training run files")


    args = parser.parse_args()

    dataloaders = get_dataloaders(args.dataset_root_path, args.batch_size)
    train_model(dataloaders, args)

