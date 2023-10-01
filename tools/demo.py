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

# 将直线进行霍夫变换
def convert_line_to_hough(line, size=(32, 32)):
    H, W = size
    theta = line.angle()
    alpha = theta + np.pi / 2 
    if theta == -np.pi / 2:
        r = line.coord[1] - W/2  
    else:
        k = np.tan(theta)
        y1 = line.coord[0] - H/2
        x1 = line.coord[1] - W/2 
        r = (y1 - k*x1) / np.sqrt(1 + k**2)
    return alpha, r

# 将直线映射到霍夫变换空间
def line2hough(line, numAngle, numRho, size=(32, 32)):
    H, W = size
    alpha, r = convert_line_to_hough(line, size)

    irho = int(np.sqrt(H*H + W*W) + 1) / ((numRho - 1)) # 距离映射到离散空间的步长
    itheta = np.pi / numAngle

    r = int(np.round(r / irho)) + int((numRho) / 2)
    alpha = int(np.round(alpha / itheta))
    if alpha >= numAngle:
        alpha = numAngle - 1
    return alpha, r

def line2hough_float(line, numAngle, numRho, size=(32, 32)):
    H, W = size
    alpha, r = convert_line_to_hough(line, size)

    irho = int(np.sqrt(H*H + W*W) + 1) / ((numRho - 1))
    itheta = np.pi / numAngle

    r = r / irho + numRho / 2
    alpha = alpha / itheta
    if alpha >= numAngle:
        alpha = numAngle - 1
    return alpha, r

def reverse_mapping(point_list, numAngle, numRho, size=(32, 32)):
    #return type: [(y1, x1, y2, x2)]
    H, W = size
    irho = int(np.sqrt(H*H + W*W) + 1) / ((numRho - 1))
    itheta = np.pi / numAngle
    b_points = []

    for (thetai, ri) in point_list:
        theta = thetai * itheta
        r = ri - numRho // 2
        cosi = np.cos(theta) / irho
        sini = np.sin(theta) / irho
        if sini == 0:
            x = np.round(r / cosi + W / 2)
            b_points.append((0, int(x), H-1, int(x)))
        else:
            # print('k = %.4f', - cosi / sini)
            # print('b = %.2f', np.round(r / sini + W * cosi / sini / 2 + H / 2))
            angle = np.arctan(- cosi / sini)
            y = np.round(r / sini + W * cosi / sini / 2 + H / 2)
            p1, p2 = get_boundary_point(int(y), 0, angle, H, W) # 获取一条直线与图像相交的2个点
            if p1 is not None and p2 is not None:
                b_points.append((p1[1], p1[0], p2[1], p2[0]))
    return b_points

def visulize_mapping(b_points, size, filename):
    img = cv2.imread(os.path.join('/home/lijiarui/DHT_new/', filename)) #change the path when using other dataset.
    img = cv2.resize(img, size)
    for (y1, x1, y2, x2) in b_points:
        img = cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), thickness=int(0.01*max(size[0], size[1])))
    return img

def get_frames(video_name):
    print('Getting video %s' % video_name)
    if not video_name:
        cap = cv2.VideoCapture(0)

        # warmup
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
    image_path = '/home/lijiarui/DHT_new/test.jpg'

    image = cv2.imread(image_path)

    line_coords = [(150, 200, 180, 300)] 

    rect_coords = [0, 0, 0, 0]

    if not os.path.exists('output_frames'):
        os.makedirs('output_frames')
        
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if cfg.CUDA else 'cpu')
    
    print('Loading model...')

    model = ModelBuilder_tctrack('test')

    model = load_pretrain(model, args.snapshot).eval().to(device)

    tracker = TCTrackTracker(model)
    hp = [cfg.TRACK.PENALTY_K, cfg.TRACK.WINDOW_INFLUENCE, cfg.TRACK.LR]

    first_frame = True
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'

    frame_counter = 0  

    print('Start to get frames...')

    for frame in get_frames(args.video_name):
        frame_counter += 1  

        if first_frame:
            init_rect = (250, 120, 50, 80) 
            tracker.init(frame, init_rect)
            rect_coords = [init_rect[0], init_rect[1], init_rect[0] + init_rect[2], init_rect[1] + init_rect[3]]
            first_frame = False
        else:
            outputs = tracker.track(frame, hp)
            bbox = list(map(int, outputs['bbox']))
            cv2.rectangle(frame, (bbox[0], bbox[1]),
                           (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                           (0, 255, 0), 3)

            rect_coords = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]

        for coords in line_coords:
            y1, x1, y2, x2 = coords
            frame_with_line = draw_horizontal_line(y1, x1, x2, frame.copy(), color=(0, 0, 255))

        cv2.rectangle(frame_with_line, (rect_coords[0], rect_coords[1]),
                      (rect_coords[2], rect_coords[3]), (0, 255, 0), 3)

        frame_filename = f'output_frames/frame_{frame_counter:04d}.jpg'
        cv2.imwrite(frame_filename, frame_with_line)

if __name__ == '__main__':
    main()
