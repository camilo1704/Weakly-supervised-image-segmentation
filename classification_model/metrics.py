"""
Code take and modified from https://albumentations.ai/docs/examples/pytorch_classification/
"""
import torch
from collections import defaultdict
import json
from os.path import join

def calculate_accuracy(output, target):
    output = torch.sigmoid(output) >= 0.5
    target = target == 1.0
    return torch.true_divide((target == output).sum(dim=0), output.size(0)).item()

class MetricMonitor:
    def __init__(self, save_path, float_precision=3):
        self.float_precision = float_precision
        self.save_path = save_path
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]
    def get_acc(self):
        return self.metrics["Accuracy"]["avg"]
    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"], float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )
    def save_metric(self, epoch):
        for (metric_name, metric) in self.metrics.items():
            with open(self.save_path+"_"+metric_name+".json", 'a') as fp:
                fp.write(f'{metric["avg"]}\n')


