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

from utils.process import load_pretrained_model, load_checkpoint_model
from utils.utils import AverageMeter, accuracy
from utils.utils import Logger, savefig
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
    parser = argparse.ArgumentParser(description='Train an SpeedNet')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--load_from', help='the pretrained weight to init model')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--validate', '-v', action='store_true', default=False,
    					help='whether do the validation during training')
    parser.add_argument('--gpus', type=int, default=8, help='gpu number')
    args = parser.parse_args()
    return args

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    args = parse_args()

    cfg = from_file(args.config)
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.seed is not None:
        cfg.seed = args.seed
    if args.gpus is not None:
        cfg.gpus = args.gpus
    # set random seeds
    if cfg.seed is not None:
        print('Set random seed to {}'.format(cfg.seed))
        set_random_seed(cfg.seed)

    if not os.path.exists(cfg.work_dir):
        os.makedirs(cfg.work_dir)

    ################ 1 DATA ###################
    print('Training model on {} dataset...'.format(cfg.data['dataset']))
    batch_size = cfg.data['batch_size'] * cfg.gpus
    train_dataset = UCF101Dataset(data_file=cfg.data['train_file'], img_tmpl=cfg.data['train_img_tmp'],
    							clip_len=cfg.data['train_clip_len'], size=cfg.data['size'], mode='train', shuffle=True)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_dataset = UCF101Dataset(data_file=cfg.data['val_file'], img_tmpl=cfg.data['val_img_tmp'],
    							clip_len=cfg.data['val_clip_len'], size=cfg.data['size'], mode='val', shuffle=False)
    val_dataloader= DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    ################ 2 MODEL ##################
    if cfg.load_from is not None:
        print('Init the model from pretrained weight {}.'.format(cfg.load_from))
        model = S3DG(num_class=cfg.model['num_class'])
        load_pretrained_model(model, pretrained_path=cfg.load_from)

    else:
        print('Init the model from scratch.')
        model = S3DG(num_class=cfg.model['num_class'])

    # MODEL
    # NOTE: train and resume train must have same number of GPU, since the name 'module'
    # nn.parallel
    if cfg.resume_from is not None:
        load_checkpoint_model(model, checkpoint_path=cfg.resume_from)

    if torch.cuda.device_count() > 1:  
        print('use %d gpus' % (torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=range(cfg.gpus))
    else:
        print('use 1 gpu')

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)

    # ################### 3 CRITERION and OPTIMIZER #########################
    criterion = nn.CrossEntropyLoss().to(device)  # standard crossentropy loss for classification
    # criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=5e-4)
    # set lr scheduler
    if cfg.lr_scheduler is not None:
        if cfg.lr_scheduler['type'] == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_scheduler['step'], gamma=cfg.lr_scheduler['gamma'])
        elif cfg.lr_scheduler['type'] == 'multistep':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.lr_scheduler['step'], gamma=cfg.lr_scheduler['gamma'])
        elif cfg.lr_scheduler['type'] == 'exponent':
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.lr_scheduler['gamma'])
    
    log_path = cfg.work_dir
    # IF RESUME
    if cfg.resume_from is not None:
        checkpoint = torch.load(cfg.resume_from)
        print("Resume training from checkpoint: {}...".format(cfg.resume_from))
        optimizer.load_state_dict(checkpoint['opt_dict'])
        scheduler.load_state_dict(checkpoint['lr_dict'])
        resume_epoch = checkpoint['epoch'] + 1
        logger = Logger(os.path.join(log_path, 'log.txt'), resume=True)
    else:
        print("Training model from start...")
        resume_epoch = 0
        logger = Logger(os.path.join(log_path, 'log.txt'))
        logger.set_names(['Learning Rate', 'Train Loss', 'Val Loss', 'Train Acc.', 'Val Acc.'])

    # tensorboard 
    log_dir = os.path.join(cfg.work_dir, datetime.now().strftime('%b%d_%H-%M-%S'))
    writer = SummaryWriter(log_dir=log_dir)

    ################## 4 BEGIN TRAINING #########################
    num_epochs = cfg.num_epochs
    save_epoch = cfg.interval
    save_dir = cfg.work_dir
    display = cfg.display

    best_acc = 0.0
    best_epoch = 0

    for epoch in tqdm(range(resume_epoch, num_epochs)):
        print('\n----------------- Training -------------------')
        print('Epoch: {}/{}'.format(epoch, num_epochs-1))
        train_loss, train_acc = train(train_dataloader, model, criterion, optimizer, epoch, writer, display)
        if args.validate:
            print('\n----------------- Validation -------------------')
            print('Epoch: {}/{}'.format(epoch, num_epochs-1))
            val_loss, val_acc = validation(val_dataloader, model, criterion, optimizer, epoch, writer, display)
            if val_acc >= best_acc:
                best_acc = val_acc
                best_epoch = epoch
            print("\nThe best validation top1-accuracy: {:.3f}%, the best epoch: {}".format(best_acc,best_epoch))

        # EPOCH
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        if args.validate:
            logger.append([lr, train_loss, val_loss, train_acc, val_acc])
        else:
            logger.append([lr, train_loss, 0.0, train_acc, 0.0]) # no valid
        writer.add_scalar('train/learning_rate', optimizer.state_dict()['param_groups'][0]['lr'], epoch)

        if cfg.lr_scheduler is not None:
            scheduler.step()

        if epoch % save_epoch == 0:
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(),
                'lr_dict': scheduler.state_dict()
            }, os.path.join(save_dir, 'epoch-' + str(epoch) + '.pth'))

    writer.close()
    logger.close()
    logger.plot()
    savefig(os.path.join(log_path, 'log.eps'))

def train(train_loader, model, criterion, optimizer, epoch, writer, display):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    top1 = AverageMeter()
    top5 = AverageMeter()

    pbar = tqdm(train_loader)
    for step, (inputs, labels) in enumerate(pbar):
        data_time.update(time.time() - end)

        inputs = inputs.cuda() # (bs, C, T, H, W)
        labels = labels.cuda() # (bs)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))

        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (step + 1) % display == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        epoch, step, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr']))
            print(output)

            total_step = epoch * len(train_loader) + step + 1
            writer.add_scalar('train/loss', losses.avg, total_step)
            writer.add_scalar('train/top1-acc', top1.avg, total_step)
            writer.add_scalar('train/top5-acc', top5.avg, total_step)
            for name, layer in model.named_parameters():
                writer.add_histogram('train/' + name + '._grad', layer.grad.cpu().data.numpy(), total_step)
                writer.add_histogram('train/' + name + '._data', layer.cpu().data.numpy(), total_step)
    return losses.avg, top1.avg

def validation(val_loader,model,criterion,optimizer,epoch, writer, display):
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    total_loss = 0.0

    with torch.no_grad():
        pbar = tqdm(val_loader)
        for step, (inputs,labels) in enumerate(pbar):
            data_time.update(time.time() - end)

            inputs = inputs.cuda()
            labels = labels.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.update(loss.item(),inputs.size(0))

            prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))

            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            total_loss += loss.item()

			if (step +1) % display == 0:
				output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        epoch, step, len(val_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr']))
                print(output)

                total_step = epoch * len(train_loader) + step + 1
                writer.add_scalar('val/loss', losses.avg, total_step)
                writer.add_scalar('val/top1-acc', top1.avg, total_step)
                writer.add_scalar('val/top5-acc', top5.avg, total_step)

    avg_loss = total_loss / len(val_loader)
    return avg_loss,top1.avg

if __name__ == '__main__':
    main()

