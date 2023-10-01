from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
sys.path.append('../')
import sys
sys.path.append('/home/lijiarui/TCTrack_v1/TCTrack-main') 

import argparse
import cv2
import torch
import numpy as np
import math
import torchvision
from PIL import Image
from basic_ops import *
from glob import glob

from pysot.core.config import cfg
from pysot.models.utile_tctrack.model_builder import ModelBuilder_tctrack
from pysot.tracker.tctrack_tracker import TCTrackTracker
from pysot.utils.model_load import load_pretrain

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='TCTrack demo')
parser.add_argument('--config', type=str, default='../experiments/TCTrack/config.yaml', help='config file')
parser.add_argument('--snapshot', type=str, default='./snapshot/tctrack.pth', help='model name')
parser.add_argument('--video_name', default='../test_dataset/sequence_name', type=str, help='videos or image files')
parser.add_argument('--init_rect', nargs=4, type=int, default=[100, 100, 50, 80], help='initial bounding box')
args = parser.parse_args()

def draw_horizontal_line(y, x_start, x_end, image, color=(0, 0, 255)):
    cv2.line(image, (x_start, y), (x_end, y), color, 2)
    return image

def get_frames(video_name):
    print('Getting video %s' % video_name)
    if not video_name:
        cap = cv2.VideoCapture(0)

        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
        video_name.endswith('mp4'):
        print('video is %s' % video_name)
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = sorted(glob(os.path.join(video_name, 'img', '*.jpg')))
        print('video name is %s' % video_name)
        print('images are %s' % images)
        for img in images:
            print('I am in for!')
            print('yes!')
            frame = cv2.imread(img)
            yield frame

def main():
    parser = argparse.ArgumentParser(description='TCTrack demo')
    parser.add_argument('--config', type=str, default='../experiments/TCTrack/config.yaml', help='config file')
    parser.add_argument('--snapshot', type=str, default='./snapshot/tctrack.pth', help='model name')
    parser.add_argument('--video_name', default='../test_dataset/sequence_name', type=str, help='videos or image files')
    parser.add_argument('--init_rect', nargs=4, type=int, default=[100, 100, 50, 80], help='initial bounding box')
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if cfg.CUDA else 'cpu')
    
    print('Loading model...')

    model = ModelBuilder_tctrack('test')

    model = load_pretrain(model, args.snapshot).eval().to(device)

    tracker = TCTrackTracker(model)
    hp = [cfg.TRACK.PENALTY_K, cfg.TRACK.WINDOW_INFLUENCE, cfg.TRACK.LR]  # Fixed indentation

    first_frame = True
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'

    frame_counter = 0
    line_coords = [(150, 200, 180, 300)]

    save_dir = '/home/lijiarui/TCTrack_v1/TCTrack-main/tools/output_frames'
    os.makedirs(save_dir, exist_ok=True)
    
    print('Start to get frames...')

    for frame in get_frames(args.video_name):
        frame_counter += 1 
        print('Gettinng frame %d'%(frame_counter))

        if first_frame:
            print('This is first frame')
            init_rect = (250, 120, 50, 80) 
            tracker.init(frame, init_rect)
            first_frame = False
        else:
            print('This is not the first frame')
            outputs = tracker.track(frame, hp)
            bbox = list(map(int, outputs['bbox']))

            # 绘制标注的线段并进行跟踪
            for line in line_coords:
                y1, x1, y2, x2 = line
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 3)
            
            # 保存帧图像
            frame_filename = os.path.join(save_dir, f'frame_{frame_counter:04d}.jpg')
            res = cv2.imwrite(frame_filename, frame)
            print(res)

if __name__ == '__main__':
    main()
