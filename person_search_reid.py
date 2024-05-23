#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import torch
import warnings
import argparse
import onnxruntime as ort
from utils.datasets import LoadStreams, LoadImages
from utils.draw import draw_boxes, draw_person
from utils.general import check_img_size
from utils.torch_utils import time_synchronized
from person_detect_yolov5 import Person_detect
from deep_sort import build_tracker
from utils.parser import get_config
from utils.log import get_logger
from utils.torch_utils import select_device, load_classifier, time_synchronized
from sklearn.metrics.pairwise import cosine_similarity
import time
from joblib import Parallel, delayed


def seconds_to_hms(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return int(hours), int(minutes), int(seconds)


class yolo_reid():
    def __init__(self, cfg, args):
        self.args = args
        #self.video_path = path
        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)
        # Person_detect行人检测类
        #self.person_detect = Person_detect(self.args, self.video_path)
        # deepsort 类
        self.deepsort = build_tracker(cfg, args.sort, use_cuda=use_cuda)
        self.imgsz = check_img_size(args.img_size, s=32)  # self.model.stride.max())  # check img_size
        #self.dataset = LoadImages(self.video_path, img_size=self.imgsz)
        self.query_feat = np.load(args.query)
        self.names = np.load(args.names)

    def deep_sort(self):
        idx_frame = 0
        results = []
        time_point = []

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
        video_dst_folder = args.output_path + '/' + os.path.dirname(self.video_path)
        os.makedirs(video_dst_folder, exist_ok=True)
        #video_all = cv2.VideoWriter(args.output_path + '/' + self.video_path.split('.')[0] + "_all.mp4", fourcc, fps, size, isColor=True)
        #video_600 = cv2.VideoWriter("test_600.mp4", fourcc, fps, size, isColor=True)

        #dst_folder = './extract_folder'  # 存放帧图片的位置
        #os.mkdir(dst_folder)
        #index = 1

        for video_path, img, ori_img, vid_cap in self.dataset:
            idx_frame += 1
            # print('aaaaaaaa', video_path, img.shape, im0s.shape, vid_cap)
            t1 = time_synchronized()

            # yolo detection
            bbox_xywh, cls_conf, cls_ids, xy = self.person_detect.detect(video_path, img, ori_img, vid_cap)
            try:
                # do tracking  # features:reid模型输出512dim特征
                outputs, features = self.deepsort.update(bbox_xywh, cls_conf, ori_img)
                print(len(outputs), len(bbox_xywh), features.shape)

                person_cossim = cosine_similarity(features, self.query_feat)
                max_idx = np.argmax(person_cossim, axis=1)
                maximum = np.max(person_cossim, axis=1)
                max_idx[maximum < 0.6] = -1
                score = maximum
                reid_results = max_idx
                draw_person(ori_img, xy, reid_results, self.names)  # draw_person name
                
                # 若查询到待检测的人，则输出时间点
                if not np.all(reid_results == -1):
                    total_seconds = idx_frame / fps
                    #hours, minutes, seconds = seconds_to_hms(total_seconds)
                    time_point.append(total_seconds)

                # print(features.shape, self.query_feat.shape, person_cossim.shape, features[1].shape)

                if len(outputs) > 0:
                    bbox_tlwh = []
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    ori_im = draw_boxes(ori_img, bbox_xyxy, identities)

                    for bb_xyxy in bbox_xyxy:
                        bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))

                    # results.append((idx_frame - 1, bbox_tlwh, identities))
                # print("yolo+deepsort:", time_synchronized() - t1)

                #video_all.write(ori_img)  # 将图像写入视频

                if self.args.display:
                    cv2.imshow("test", ori_img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            except:
                continue




        #video_all.release()
        #video_600.release()
        # 打开文件，如果不存在则创建，以写入模式打开
        #video_dst_folder = args.output_path + '/' + os.path.dirname(self.video_path)
        #os.makedirs(video_dst_folder, exist_ok=True)

        with open('{}.txt'.format(video_dst_folder + '/' + os.path.basename(self.video_path).split('.')[0]), 'a') as file:
            for i in sorted(list(set([f"{int(sec // 3600):02d}:{int((sec % 3600) // 60):02d}:{int(sec % 60):02d}" for sec in time_point]))):
                file.write(i+'\n')

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
    parser.add_argument('--num_workers', type=int, default=6, help='Number of worker processes')
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # yolov5
    parser.add_argument('--weights', nargs='+', type=str, default='./weights/yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=1080, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--classes', default=[0], type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')

    # deep_sort
    parser.add_argument("--sort", default=False, help='True: sort model or False: reid model')
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--display", default=False, help='show resule')
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)

    # reid
    parser.add_argument("--query", type=str, default="./fast_reid/query/query_features.npy")
    parser.add_argument("--names", type=str, default="./fast_reid/query/names.npy")

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
