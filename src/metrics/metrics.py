import torch
import torch.nn.functional as F
from config import cfg
from utils import recur
import numpy as np

def compute_iou(pred_mask,true_mask):
    intersection = np.logical_and(true_mask, pred_mask)
    union = np.logical_or(true_mask, pred_mask)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def compute_dice_coefficient(pred_mask,true_mask):
    intersection = np.sum(true_mask * pred_mask)
    union = np.sum(true_mask) + np.sum(pred_mask)
    dice_coefficient = (2. * intersection) / union
    return dice_coefficient

def compute_pixel_accuracy(pred_mask,true_mask):
    correct_pixels = np.sum(true_mask == pred_mask)
    total_pixels = true_mask.size
    pixel_accuracy = correct_pixels / total_pixels
    return pixel_accuracy


def Accuracy(output, target, topk=1):
    with torch.no_grad():
        if target.dtype != torch.int64:
            target = (target.topk(1, 1, True, True)[1]).view(-1)
        batch_size = target.size(0)
        pred_k = output.topk(topk, 1, True, True)[1]
        correct_k = pred_k.eq(target.view(-1, 1).expand_as(pred_k)).float().sum()
        acc = (correct_k * (100.0 / batch_size)).item()
    return acc


def MAccuracy(output, target, mask, topk=1):
    if torch.any(mask):
        output = output[mask]
        target = target[mask]
        acc = Accuracy(output, target, topk)
    else:
        acc = 0
    return acc


def LabelRatio(mask):
    with torch.no_grad():
        lr = mask.float().mean().item()
    return lr


class Metric(object):
    def __init__(self, metric_name):
        self.metric_name = self.make_metric_name(metric_name)
        self.pivot, self.pivot_name, self.pivot_direction = self.make_pivot()
        self.metric = {'Loss': (lambda input, output: output['loss'].item()),
                       'Accuracy': (lambda input, output: recur(Accuracy, output['target'], input['target'])),
                       'PAccuracy': (lambda input, output: recur(Accuracy, output['target'], input['target'])),
                       'MAccuracy': (lambda input, output: recur(MAccuracy, output['target'], input['target'],
                                                                 output['mask'])),
                       'LabelRatio': (lambda input, output: recur(LabelRatio, output['mask'])),
                       'iou' : (lambda input,output :recur(compute_iou,output['target'],input['target'])),
                        'dice' : (lambda input,output :recur(compute_dice_coefficient,output['target'],input['target'])),
                        'pixel_accuracy' : (lambda input,output :recur(compute_pixel_accuracy,output['target'],input['target']))
                       
                       }

    def make_metric_name(self, metric_name):
        return metric_name

    def make_pivot(self):
        if cfg['data_name'] in ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100', 'SVHN', 'STL10', 'voc']:
            pivot = -float('inf')
            pivot_direction = 'up'
            pivot_name = 'Accuracy'
        else:
            raise ValueError('Not valid data name')
        return pivot, pivot_name, pivot_direction

    def evaluate(self, metric_names, input, output):
        print(output.keys())
        evaluation = {}
        for metric_name in metric_names:
            evaluation[metric_name] = self.metric[metric_name](input, output)
        return evaluation

    def compare(self, val):
        if self.pivot_direction == 'down':
            compared = self.pivot > val
        elif self.pivot_direction == 'up':
            compared = self.pivot < val
        else:
            raise ValueError('Not valid pivot direction')
        return compared

    def update(self, val):
        self.pivot = val
        return
