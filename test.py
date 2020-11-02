import time
from datetime import datetime
import socket
import os
import glob
from tqdm import tqdm
import importlib
import argparse

import numpy as np
import random

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from datasets import UCF101Dataset
from models import S3DG

from utils.process import load_checkpoint_model
from utils.utils import AverageMeter, accuracy
from utils.settings import from_file


# CUDA_DEVICES = [0, 1, 2, 3, 4, 5, 6, 7]
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"
# CUDA_DEVICES = [0]
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.benchmark = True

# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("\nGPU being used:", torch.cuda.is_available())

def parse_args():
    parser = argparse.ArgumentParser(description='Test model on action recognition')
    parser.add_argument('config', help='config file path, and its format is config_file.config')
    parser.add_argument('--resume_from', type=str, help='the checkpoint file to init model')
    parser.add_argument('--gpus', type=int, default=8, help='gpu number')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    cfg = from_file(args.config)
    if args.gpus is not None:
        cfg.gpus = args.gpus
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    else:
        raise ValueError('Please specify the path to load checkpoint')


    ################ 1 DATA ###################
    print('Testing model on {} dataset...'.format(cfg.data['dataset']))
    batch_size = 1 * cfg.gpus # since a video contain 10 clips
    test_dataset = UCF101Dataset(data_file=cfg.data['test_file'], img_tmpl=cfg.data['test_img_tmp'],
    							clip_len=cfg.data['test_clip_len'], size=cfg.data['size'], mode='test', shuffle=False)
    test_dataloader= DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    ################ 2 MODEL ##################
    # init model from checkpoint
    model = S3DG(num_class=cfg.model['num_class'])
    load_checkpoint_model(model, checkpoint_path=cfg.resume_from)

    if torch.cuda.device_count() > 1:  
        print('use %d gpus' % (torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=range(cfg.gpus))
    else:
        print('use 1 gpu')

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)

    # ################### 3 CRITERION  #########################
    criterion = nn.CrossEntropyLoss().to(device)  # standard crossentropy loss for classification
    # criterion = nn.BCEWithLogitsLoss().to(device)
    
    ################## 4 BEGIN TEST #########################
    avg_loss, avg_acc = test(test_dataloader, model, criterion)

def test(test_loader, model, criterion):
    torch.set_grad_enabled(False)
    model.eval()

    total_loss = 0.0
    correct = 0
    total_num = 0

    pbar = tqdm(test_loader)
    for step, (inputs,labels) in enumerate(pbar):
        inputs = inputs.cuda()  # (bs, 10, C, T, H,W)
        labels = labels.cuda()  # (bs)
        outputs = []
        for clip in inputs:
            clip = clip.cuda()  # (10, C, T, H, W)
            out = model(clip)   # (10, 101)
            out = torch.mean(out, dim=0)  # (101,)
            outputs.append(out)
        outputs = torch.stack(outputs) # (bs, 101)

        loss = criterion(outputs, labels)
        # compute loss and acc
        total_loss += loss.item()

        pts = torch.argmax(outputs, dim=1)
        correct += torch.sum(labels == pts).item()
        total_num += inputs.size(0)
        # print('correct: {}, {}, {}'.format(correct, targets, pts))
        pbar.set_description('{}/{}: correct {}/{}'.format(step, len(test_loader,
                                correct, total_num)))
        # print(str(step), len(test_loader))
        # print(correct)

    avg_loss = total_loss / len(test_loader)
    # avg_loss = total_loss / (len(val_loader)+len(train_loader))
    avg_acc = correct / len(test_loader.dataset)
    # avg_acc = correct / (len(val_loader.dataset)+len(train_loader.dataset))
    print('[TEST] loss: {:.3f}, acc: {:.3f}'.format(avg_loss, avg_acc))
    return avg_loss, avg_acc



if __name__ == '__main__':
	main()

