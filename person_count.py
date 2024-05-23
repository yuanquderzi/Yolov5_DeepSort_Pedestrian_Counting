#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import cv2
import torch
import warnings
import argparse
import numpy as np
import onnxruntime as ort
from utils.datasets import LoadStreams, LoadImages
from utils.draw import draw_boxes
from utils.general import check_img_size
from utils.torch_utils import time_synchronized
from person_detect_yolov5 import Person_detect
from deep_sort import build_tracker
from utils.parser import get_config
from utils.log import get_logger
from utils.torch_utils import select_device, load_classifier, time_synchronized
# count
from collections import Counter
from collections import deque
import math
from PIL import Image, ImageDraw, ImageFont
import time
from joblib import Parallel, delayed
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def tlbr_midpoint(box):
    minX, minY, maxX, maxY = box
    midpoint = (int((minX + maxX) / 2), int((minY + maxY) / 2))  # minus y coordinates to get proper xy format
    return midpoint

def xiamian_midpoint(box):
    minX, minY, maxX, maxY = box
    midpoint = (int((maxX + minX) / 2), int(maxY))
    return midpoint


def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def vector_angle(midpoint, previous_midpoint):
    x = midpoint[0] - previous_midpoint[0]
    y = midpoint[1] - previous_midpoint[1]
    return math.degrees(math.atan2(y, x))


def get_size_with_pil(label,size=25):
    font = ImageFont.truetype("./configs/simkai.ttf", size, encoding="utf-8")  # simhei.ttf
    return font.getsize(label)


#为了支持中文，用pil
def put_text_to_cv2_img_with_pil(cv2_img,label,pt,color):
    pil_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同，需转RGB模式
    pilimg = Image.fromarray(pil_img)  # Image.fromarray()将数组类型转成图片格式，与np.array()相反
    draw = ImageDraw.Draw(pilimg)  # PIL图片上打印汉字
    font = ImageFont.truetype("./configs/simkai.ttf", 25, encoding="utf-8") #simhei.ttf
    draw.text(pt, label, color,font=font)
    return cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)  # 将图片转成cv2.imshow()可以显示的数组格式


colors = np.array([
    [1,0,1],
    [0,0,1],
    [0,1,1],
    [0,1,0],
    [1,1,0],
    [1,0,0]
    ]);

def get_color(c, x, max):
    ratio = (x / max) * 5;
    i = math.floor(ratio);
    j = math.ceil(ratio);
    ratio -= i;
    r = (1 - ratio) * colors[i][c] + ratio * colors[j][c];
    return r;

def compute_color_for_labels(class_id,class_total=80):
    offset = (class_id + 0) * 123457 % class_total;
    red = get_color(2, offset, class_total);
    green = get_color(1, offset, class_total);
    blue = get_color(0, offset, class_total);
    return (int(red*256),int(green*256),int(blue*256))

def seconds_to_hms(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return int(hours), int(minutes), int(seconds)


class yolo_reid():
    def __init__(self, cfg, args):
        self.logger = get_logger("root")
        self.args = args
        #self.video_path = path
        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        #self.person_detect = Person_detect(self.args, self.video_path)
        self.imgsz = check_img_size(args.img_size, s=32)  # self.model.stride.max())  # check img_size
        #self.dataset = LoadImages(self.video_path, img_size=self.imgsz)
        self.deepsort = build_tracker(cfg, args.sort, use_cuda=use_cuda)

    def deep_sort(self):
        idx_frame = 0
        results = []
        paths = {}
        track_cls = 0
        last_track_id = -1.
        total_track = 0
        angle = -1
        total_counter = 0
        up_count = 0
        down_count = 0
        class_counter = Counter()   # store counts of each detected class
        already_counted = deque(maxlen=50)   # temporary memory for storing counted IDs

        # 保存处理后的视频
        cap = cv2.VideoCapture(self.video_path)
        frame_num = int(cap.get(7))  # 总帧数
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取原视频的宽
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取原视频的搞
        fps = int(cap.get(cv2.CAP_PROP_FPS))  # 帧率
        # fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))  # 视频的编码
        # fourcc = cv2.VideoWriter_fourcc('M','P','4','V') # mp4
        # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        size = (width, height)  # (width, height) 数值可根据需要进行调整
        #video_all = cv2.VideoWriter("NOR_8888888_000000_20230417103635_0020_all.MP4", fourcc, fps, size, isColor=True)
        
        video_dst_folder = args.output_path + '/' + os.path.dirname(self.video_path)
        os.makedirs(video_dst_folder, exist_ok=True)

        #video_all = cv2.VideoWriter(args.output_path + '/' + self.video_path.split('.')[0] + "_all.mp4", fourcc, fps, size, isColor=True)
        
        dst_folder = video_dst_folder + '/extract_folder_' + os.path.basename(self.video_path).split('.')[0]  # 存放帧图片的位置
        os.makedirs(dst_folder, exist_ok=True)

        index = 1

        for video_path, img, ori_img, vid_cap in self.dataset:
            idx_frame += 1
            # print('aaaaaaaa', video_path, img.shape, im0s.shape, vid_cap)
            t1 = time_synchronized()

            # yolo detection
            bbox_xywh, cls_conf, cls_ids, xy = self.person_detect.detect(video_path, img, ori_img, vid_cap)

            # do tracking
            outputs = self.deepsort.update(bbox_xywh, cls_conf, ori_img)

            # 1.视频中间画行黄线
            # line = [(int(ori_img.shape[1]/2), int(ori_img.shape[0] *3/5)), (int(ori_img.shape[1]), int(ori_img.shape[0]*3/5))]
            #line1 = [(1071, 757), (1670, 984)] 2024_1_12
            #line1 = [(1215, 873), (1525, 947)] 2024_1_18
            line1 = [(586, 676), (929, 584)]

            cv2.line(ori_img, line1[0], line1[1], (0, 255, 255), 4)

            # 2. 统计人数
            for track in outputs:
                bbox = track[:4]
                track_id = track[-1]
                midpoint = xiamian_midpoint(bbox)  #下边框中心
                #midpoint = tlbr_midpoint(bbox)  #框中心
                origin_midpoint = (midpoint[0], ori_img.shape[0] - midpoint[1])  # get midpoint respective to botton-left

                if track_id not in paths:
                    paths[track_id] = deque(maxlen=2)
                    total_track = track_id
                paths[track_id].append(midpoint)
                previous_midpoint = paths[track_id][0]
                origin_previous_midpoint = (previous_midpoint[0], ori_img.shape[0] - previous_midpoint[1])

                if intersect(midpoint, previous_midpoint, line1[0], line1[1]) and track_id not in already_counted:
                    class_counter[track_cls] += 1
                    total_counter += 1
                    last_track_id = track_id;
                    # draw red line
                    cv2.line(ori_img, line1[0], line1[1], (0, 0, 255), 10)

                    already_counted.append(track_id)  # Set already counted for ID to true.

                    
                    angle = vector_angle(origin_midpoint, origin_previous_midpoint)

                    if angle > 0:
                        up_count += 1
                    if angle < 0:
                        down_count += 1
                

                if len(paths) > 50:
                    del paths[list(paths)[0]]

            # 3. 绘制人员
            if len(outputs) > 0:
                bbox_tlwh = []
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                ori_img = draw_boxes(ori_img, bbox_xyxy, identities)

                for bb_xyxy in bbox_xyxy:
                    bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))

                # results.append((idx_frame - 1, bbox_tlwh, identities))
            print("yolo+deepsort:", time_synchronized() - t1)

            # 4. 绘制统计信息
            '''
            label = "客流总数: {}".format(str(total_track))
            t_size = get_size_with_pil(label, 25)
            x1 = 20
            y1 = 50
            color = compute_color_for_labels(2)
            cv2.rectangle(ori_img, (x1 - 1, y1), (x1 + t_size[0] + 10, y1 - t_size[1]), color, 2)
            ori_img = put_text_to_cv2_img_with_pil(ori_img, label, (x1 + 5, y1 - t_size[1] - 2), (0, 0, 0))
            '''

            label = "穿过黄线人数: {} ({} 向上, {} 向下)".format(str(total_counter), str(up_count), str(down_count))
            #label = "穿过黄线人数: {}".format(str(total_counter))
            t_size = get_size_with_pil(label, 25)
            x1 = 20
            y1 = 100
            color = compute_color_for_labels(2)
            cv2.rectangle(ori_img, (x1 - 1, y1), (x1 + t_size[0] + 10, y1 - t_size[1]), color, 2)
            ori_img = put_text_to_cv2_img_with_pil(ori_img, label, (x1 + 5, y1 - t_size[1] - 2), (0, 0, 0))

            '''
            if last_track_id >= 0:
                label = "最新: 行人{}号{}穿过黄线".format(str(last_track_id), str("向上") if angle >= 0 else str('向下'))
                t_size = get_size_with_pil(label, 25)
                x1 = 20
                y1 = 150
                color = compute_color_for_labels(2)
                cv2.rectangle(ori_img, (x1 - 1, y1), (x1 + t_size[0] + 10, y1 - t_size[1]), color, 2)
                ori_img = put_text_to_cv2_img_with_pil(ori_img, label, (x1 + 5, y1 - t_size[1] - 2), (255, 0, 0))
            '''

            end = time_synchronized()

            #video_all.write(ori_img)  # 将图像写入视频

            save_path = "{}/{}_{:>03d}.jpg".format(dst_folder, os.path.basename(self.video_path).split('.')[0], index)
            # 保存最后1张图片
            if index > frame_num-1:
                cv2.imwrite(save_path, ori_img)
                with open(video_dst_folder + '/output.txt', 'a') as file:
                    file.write(f'{os.path.basename(self.video_path)}: {label}\n')
            index += 1

            if self.args.display:
                cv2.imshow("test", ori_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            self.logger.info("{}/time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
                             .format(idx_frame, end - t1, 1 / (end - t1),
                                     bbox_xywh.shape[0], len(outputs)))
            
        #video_all.release()
    
    def __call__(self, path):
        self.video_path = path
        self.person_detect = Person_detect(self.args, self.video_path)
        self.dataset = LoadImages(self.video_path, img_size=self.imgsz)
        with torch.no_grad():
            self.deep_sort()


def process_video(each_video_path):
    each_time_start = time.time()
    yolo_reid(each_video_path)
    hours, minutes, seconds = seconds_to_hms(time.time() - each_time_start)
    print(f"{each_video_path} cost: {hours}小时 {minutes}分钟 {seconds}秒")

            
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default='linyi_20240124', type=str)
    parser.add_argument("--output_path", default='output', type=str)
    parser.add_argument('--num_workers', type=int, default=10, help='Number of worker processes')
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # yolov5
    parser.add_argument('--weights', nargs='+', type=str, default='./weights/yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=960, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--classes', default=[0], type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')

    # deep_sort
    parser.add_argument("--sort", default=True, help='True: sort model, False: reid model')
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--display", default=False, help='show resule')
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    cfg = get_config()
    cfg.merge_from_file(args.config_deepsort)
    yolo_reid = yolo_reid(cfg, args)

    all_time_start = time.time()
    
    video_paths = [os.path.join(args.video_path, each_video) for each_video in os.listdir(args.video_path)]

    # 使用 Parallel 和 delayed 来并行处理循环任务
    Parallel(n_jobs=args.num_workers)(delayed(process_video)(video_path) for video_path in video_paths)

    hours, minutes, seconds = seconds_to_hms(time.time() - all_time_start)
    print(f"{args.video_path} all_cost: {hours}小时 {minutes}分钟 {seconds}秒")
