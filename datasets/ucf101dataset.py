import os
from sklearn.model_selection import train_test_split
import math
import re

import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

import random
from PIL import Image
import pdb

class UCF101Dataset(Dataset):
    """
    The dataset of UCF101 for training action recognition. class index start from 0.
    """
    def __init__(self, data_file, img_tmpl, clip_len=16, size=(224, 224), mode='train', shuffle=False):
        self.data_file = data_file
        self.img_tmpl = img_tmpl
        # the data ann file should in the same directory as frames
        #self.root_dir = os.path.join(os.path.dirname(data_file), 'frames')

        self.clip_len = clip_len
        self.size = size if isinstance(size, tuple) else (size, size)
        assert mode in ['train', 'val', 'test']
        self.mode = mode   # train, val, test

        # use the videos that satisfies the min duration time
        self.fnames = self.load_file_list(shuffle)
        self.video_num = len(self.fnames)
        print('Number of {} videos: {:d}'.format(mode, self.video_num))


        # for training we doing the RandomCrop, for vaildation, we would not implement that.
        mean = [0.43216, 0.394666, 0.37645]   # for kinetics
        std = [0.22803, 0.22145, 0.216989]
        self.transform_train = transforms.Compose([
        transforms.Resize(int(1.2 * min(self.size))),
        transforms.RandomCrop(self.size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        ])   
        self.transform_val = transforms.Compose([
        transforms.Resize(int(1.2 * min(self.size))),
        transforms.CenterCrop(self.size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        ])
        self.transform_test = self.transform_val
   

    def __len__(self):
        return self.video_num

    def __getitem__(self, index):
    	frame_path, frame_num, label = self.fnames[index].split(' ')
    	frame_num = int(frame_num)
    	label = int(label)
    	clip = self.load_clip(frame_path, frame_num, label)
    	return clip, torch.tensor(label)

    def load_clip(self, frame_path, frame_num, label):
        """
        frame_names = []
        for frame_name in os.listdir(frame_path):
            frame_names.append(os.path.join(frame_path, frame_name))
        # A huge PROBLEM: since the sorted order in UBUNTU system is not 1,2,3,...,10,11,12
        # it is 10,11,12,1,2,...
        # So needs to sort the frames
        frame_names = sorted(frame_names, key=lambda i:int(re.findall('\d+',i)[-1]))
        frame_num = len(frame_names)
        """

        # random select a clip from the video for training and validation
        if self.mode == 'train' or self.mode == 'val':
            start_id = random.randint(1, frame_num - self.clip_len)
            # a hidden problem: the img_name may not start from 00000.jpg

            frames = []
            for i in range(start_id, start_id + self.clip_len):
                frame_name = os.path.join(frame_path, self.img_tmpl.format(i))
                # print(frame_name)
                frame = cv2.imread(frame_name)   # (h, w, c)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # (h, w, c)
                if self.mode == 'train':
                    frame = Image.fromarray(frame)
                    frame = self.transform_train(frame)
                else:
                    frame = Image.fromarray(frame)
                    frame = self.transform_val(frame)
                frames.append(frame)
            clip = torch.stack(frames).permute([1, 0, 2, 3])  # (T,C,H,W) => (C,T,H,W)
            return clip # (C,T,H,W)
        elif self.mode == 'test':
            all_clips = []
            for samle_num in np.linspace(1, frame_num - self.clip_len, 10):
                start_id = int(samle_num)
                frames = []
                for i in range(start_id, start_id + self.clip_len):
                    frame_name = os.path.join(frame_path, self.img_tmpl.format(i))
                    frame = cv2.imread(frame_name)   # (h, w, c)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # (h, w, c)
                    frame = Image.fromarray(frame)
                    frame = self.transform_test(frame)
                    frames.append(frame)
                clip = torch.stack(frames).permute([1, 0, 2, 3])   # (T,C,H,W) => (C,T,H,W)
                all_clips.append(clip)  
            return torch.stack(all_clips) # (10, C, T, H, W)


    # Use the video whose duration is larger than self.clip_len, return the path to frame folder
    def load_file_list(self, shuffle=False):
        file_list = []
        temp_list = []
        with open(self.data_file, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                temp_list.append(line)

        for i in range(len(temp_list)):
        	frame_number = int(temp_list[i].split(' ')[1])
        	if frame_number > self.clip_len:
        		file_list.append(temp_list[i])

        if shuffle:
        	random.shuffle(file_list)

        return file_list

if __name__ == "__main__":
    
    # train_data = VideoDataset(dataset='ucf101', split='test', clip_len=8, preprocess=False)
    # train_loader = DataLoader(train_data, batch_size=100, shuffle=True, num_workers=4)

    train_data = UCF101Dataset('./data/train_folder.txt', img_tmpl='{:05d}.jpg',
    							clip_len=16, mode='train')
    train_loader = DataLoader(train_data, batch_size=2, shuffle=True, num_workers=0)
    pdb.set_trace()
    for i, (inputs, labels) in enumerate(train_loader):
    	pdb.set_trace()
    	print(inputs.size())
    	print(labels.size())
