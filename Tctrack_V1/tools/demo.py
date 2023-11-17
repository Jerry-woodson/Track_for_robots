import os
import sys
sys.path.append('../')

import argparse
import torch
from glob import glob
import cv2
import numpy as np
from PIL import Image

from pysot.core.config import cfg
from pysot.models.utile_tctrack.model_builder import ModelBuilder_tctrack
from pysot.tracker.tctrack_tracker import TCTrackTracker
from pysot.utils.model_load import load_pretrain
from natsort import natsorted

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='TCTrack demo')
parser.add_argument('--config', type=str, default='../experiments/TCTrack/config.yaml', help='config file')
parser.add_argument('--snapshot', type=str, default='./snapshot/tctrack.pth', help='model name')
parser.add_argument('--video_name', default='/home/xiangfuli96/tctrack/test_dataset/dataset/', type=str, help='videos or image files')
args = parser.parse_args()

def get_frames(video_name):
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

    elif video_name.endswith('avi') or video_name.endswith('mp4'):
        cap = cv2.VideoCapture(video_name) 
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break

    else:
        images = natsorted(glob(os.path.join(video_name, 'img*.jpg')))
        for img in images:
            frame = np.array(Image.open(img))
            yield frame

def main():
    # 加载配置
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # 创建模型
    model = ModelBuilder_tctrack('test')

    # 加载模型
    model = load_pretrain(model, args.snapshot).eval().to(device)

    # 构建追踪器
    tracker = TCTrackTracker(model)
    hp = [cfg.TRACK.PENALTY_K, cfg.TRACK.WINDOW_INFLUENCE, cfg.TRACK.LR]

    first_frame = True
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'

    # 创建目录保存图像文件
    output_dir = 'tracked_images'
    os.makedirs(output_dir, exist_ok=True)
    print("after mkdirs")

    # 创建 VideoWriter 对象
    video_writer = cv2.VideoWriter('tracked_video.avi', cv2.VideoWriter_fourcc(*'XVID'), 20, (640, 480))

    for frame_number, frame in enumerate(get_frames(args.video_name), start=1):
        print("first frame")
        if first_frame:
            print("in first frame")
            init_rect = (250, 180, 30, 30)  # x y width height 
            print("first frame init")
            tracker.init(frame, init_rect)
            first_frame = False

        print("outputs")
        outputs = tracker.track(frame, hp)
        bbox = list(map(int, outputs['bbox']))
        cv2.rectangle(frame, (bbox[0], bbox[1]),
                      (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                      (0, 255, 0), 3)

        # 保存帧为图像文件
        output_path = os.path.join(output_dir, f'frame_{frame_number:04d}.jpg')
        cv2.imwrite(output_path, frame)

        # 写入视频
        video_writer.write(frame)

    # 释放 VideoWriter 对象
    video_writer.release()

if __name__ == '__main__':
    print("I am before main!")
    main()
    print("I am after main!")
