# Refer Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import os
import cv2
from multiprocessing import Pool

dataset = 'ucf101'

if dataset == 'hmdb51':
    VIDEO_ROOT = '/mnt/lustre/share/wujiannan/hmdb51/videos'
    FRAME_ROOT = '/mnt/lustre/share/wujiannan/hmdb51/frames'
    NUM_WORKER = 8
    LEVEL = 2
elif dataset == 'ucf101':
    VIDEO_ROOT = '/mnt/lustre/share/wujiannan/ucf101/videos'
    FRAME_ROOT = '/mnt/lustre/share/wujiannan/ucf101/frames'
    NUM_WORKER = 8
    LEVEL = 2
elif dataset == 'sthv2':
    VIDEO_ROOT = '/mnt/lustre/share/wujiannan/something-something-v2/videos'
    FRAME_ROOT = '/mnt/lustre/share/wujiannan/something-something-v2/frames'
    NUM_WORKER = 8
    LEVEL = 1
else:  # specify your dataset setting here
    VIDEO_ROOT = './data/videos'
    FRAME_ROOT = './data/frames'
    NUM_WORKER = 8
    LEVEL = 1    


def extract_frame(video_item):
    video_base_path, video_name, video_id, video_num = video_item
    video_path = os.path.join(video_base_path, video_name)

    if LEVEL == 1:
        output_dir = os.path.join(FRAME_ROOT, video_name.split('.')[0])
    elif LEVEL == 2:
        output_path = os.path.join(FRAME_ROOT, video_base_path.split('/')[-1])
        output_dir = os.path.join(output_path, video_name.split('.')[0])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path) 
    # Ensure the fps is around 25
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps > 25:
        frame_gap = fps // 25
    else: 
        frame_gap = 1

    frame_count = 1
    success = True
    while(success): 
        success, frame = cap.read()
        if not success:
            break

        # For python3
        # print('Frame  {}'.format(str(frame_count)))
        if frame_count % frame_gap == 0:
            params = [] 
            params.append(90)   # represent the quality of the image, for jpg, the higher, the better
            try:
                cv2.imwrite(output_dir + "/%05d.jpg" % frame_count, frame, params) 
            except:
                success = False

        frame_count = frame_count + 1

    cap.release() 
    print(f' {video_id}/{video_num} done')
    return True


if __name__ == '__main__':

    if not os.path.exists(VIDEO_ROOT):
        raise ValueError('Please download videos and set VIDEO_ROOT variable.')
    if not os.path.exists(FRAME_ROOT):
        os.makedirs(FRAME_ROOT)

    if LEVEL == 1:
        video_list = os.listdir(VIDEO_ROOT)
        video_num = len(video_list)
        video_path_list = video_num * [VIDEO_ROOT]
    elif LEVEL == 2:
        video_list = []
        video_path_list = []
        for sub_dir in os.listdir(VIDEO_ROOT):
            video_path = os.path.join(VIDEO_ROOT, sub_dir)
            video = os.listdir(video_path)
            cur_video_num = len(video)
            video_list = video_list + video
            video_path_list = video_path_list + cur_video_num * [video_path]
        video_num = len(video_list)
    else:
        raise ValueError('Please check your directory structure, the level must be 1 or 2')


    pool = Pool(NUM_WORKER)
    pool.map(
        extract_frame,
        zip(video_path_list, video_list, range(1, video_num + 1), video_num * [video_num]))