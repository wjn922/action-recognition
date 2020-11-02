import os
import argparse
import importlib

import torch
from torch.autograd import Variable
from torchvision import transforms

import numpy as np
import random
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from datasets import UCF101Dataset
from models import S3DG

from utils.process import read_label_data, load_pretrained_model, load_checkpoint_model
from utils.utils import AverageMeter, accuracy
from utils.settings import from_file
from utils.visualization import calculate_cam

import pdb

# change the number of gpu here!
CUDA_DEVICES = [0]
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.benchmark = True

# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("\nGPU being used:", torch.cuda.is_available())


def parse_args():
    parser = argparse.ArgumentParser(description='Test model on a movie')
    parser.add_argument('config', help='config file path, and its format is config_file.config')
    parser.add_argument('--load_from', type=str, help='the pretrained weight to init model')
    parser.add_argument('--resume_from', type=str, help='the checkpoint file to init model')
    parser.add_argument('--cam', '-c', action='store_true', default=False)
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    cfg = from_file(args.config)
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if cfg.load_from is None and cfg.resume_from is None:
        raise ValueError('Please specify the path to load checkpoint')

    ######################### init model
    # inference only use 1 GPU, because the input is a single video(clip), size (1, C, T, H, W)
    if cfg.load_from is not None:
        model = S3DG(num_class=cfg.model['num_class'])
        load_pretrained_model(model, cfg.load_from)
    if cfg.resume_from is not None:
        model = S3DG(num_class=cfg.model['num_class'])
        load_checkpoint_model(model, cfg.resume_from)
    
    model.to(device)
    model.eval()

    # hook the 3D feature
    finalconv_name = 'base'
    feature_blobs = []
    def hook_feature(module, input, output):
        feature_blobs.append(output.data.cpu().numpy())
    model._modules.get(finalconv_name).register_forward_hook(hook_feature)
    weight_fc = model.fc[0].weight.data.detach().cpu().numpy()    # size(num_class, 1024, 1, 1, 1)
    # since model.fc is a sequential
   
    ########################### read video
    if cfg.data['dataset'] == 'ucf101':
        label_path = 'data/ucf101/ucf101_label.txt'
    elif cfg.data['dataset'] == 'kinetics400':
        label_path = 'data/kinetics400/kinetics400_label.txt'
    else:
        raise IOError("The dataset should be 'ucf101' or 'kinetics400'.")
    id2label, label2id = read_label_data(label_path)

    video = './data/videos/0a8T8M0gt20.avi'
    if not os.path.exists(video):
        raise OSError('The video does not exist!!!')
    
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)                 # fps
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))    # size

    # output_path
    output_path = "./data/videos/0a8T8M0gt20_action.avi" 
    output_viedo = cv2.VideoWriter()
    #These are the FOURCCs to compressed formats
    fourcc = cv2.VideoWriter_fourcc('P', 'I', 'M', '1')  
    output_viedo.open(output_path , fourcc, fps, size, isColor=True)


    ###################  Inference setting
    clip_len = 16
    resize = cfg.data['size'] if isinstance(cfg.data['size'], tuple) \
                                else (cfg.data['size'], cfg.data['size'])
    mean = [0.43216, 0.394666, 0.37645]
    std = [0.22803, 0.22145, 0.216989]
    transform_val = transforms.Compose([
        transforms.Resize(int(1.2 * min(resize))),
        transforms.CenterCrop(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        ])

    ############## Beginning Inference
    print("Begin infering the action of video: %s" %(os.path.basename(video)))

    clip = []   # store the PIL2Tensor frame of length clip_len
    
    frame_count = 0
    success = True
    while success:
        frame_count += 1
        success, frame = cap.read()
        if not success and frame is None:
            continue

        # frame is a cv2 format (h, w, c) with BGR
        # the input frame size can be variable     
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, (resize[0], resize[1]), interpolation=cv2.INTER_LINEAR)
        img = Image.fromarray(img)
        img = transform_val(img)   # torch tensor (C, H ,W)
        clip.append(img)
        if len(clip) == clip_len:
            inputs = torch.stack(clip, dim=0)                # (T, C, H, W)
            inputs = inputs.transpose(0,1).unsqueeze(0)      # (1, C, T, H, W)
            inputs = Variable(inputs, requires_grad=False).to(device)

            with torch.no_grad():
                outputs = model.forward(inputs)

            probs = torch.nn.Softmax(dim=1)(outputs)  # (1, num_class)
            label_id = torch.max(probs, 1)[1].detach().cpu().numpy()[0]

            ##### Use CAM visualization
            if args.cam:  
                cam = calculate_cam(feature_blobs[-1], weight_fc[label_id])  # (T, H, W)
                # cam = feature_blobs[-1].squeeze(axis=0).mean(axis=0) # T,H,W
                cam -= np.min(cam)
                cam /= np.max(cam) 
                heatmap = cv2.applyColorMap(cv2.resize(np.uint8(255*cam[-1]), size), cv2.COLORMAP_JET)
                frame = cv2.addWeighted(src1=frame, alpha=0.5, src2=heatmap, beta=0.8, gamma=0)
            

            cv2.putText(frame, id2label[label_id].split(' ')[-1].strip(), (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 1)
            cv2.putText(frame, "prob: %.4f" % probs[0][label_id], (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 1)

            cv2.imwrite('./heatmap/frame{:05d}.jpg'.format(frame_count), frame)
            output_viedo.write(frame)

            clip.pop(0)
            feature_blobs.pop(0)  # in case it needs large memory 


    output_viedo.release()
    cap.release()

    print("Completed video: %s" %(os.path.basename(video)))




if __name__ == '__main__':
    main()
