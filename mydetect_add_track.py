import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from tqdm import tqdm
from scipy import misc
# % matplotlib inline
from matplotlib import pyplot as plt
from objecttracker.KalmanFilterTrackerCell import Tracker  # 加载卡尔曼滤波函数
import copy
import colorsys
import os, sys, argparse, random
from timeit import default_timer as timer
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from tqdm import tqdm
from scipy import misc

def calc_center(out_boxes, out_classes, out_scores, score_limit=0.7):
    outboxes_filter = []
    for x, y, z in zip(out_boxes, out_classes, out_scores):
        if z > score_limit:
            if y == 0:  # cell标签为0
                outboxes_filter.append(x)
    centers = []
    number = len(outboxes_filter)
    # need to resume the x_center y_center[0,1] to original size[1,imgsize]
    for box in outboxes_filter:
        # x_center, y_center, width, height = box
        x_center = int((int(box[0]) + int(box[2])) / 2)  # x center
        y_center = int((int(box[1]) + int(box[3])) / 2 ) # y center
        center = np.array([x_center, y_center])
        centers.append(center)
    return centers, number


def get_colors_for_classes(num_classes):
    """Return list of random colors for number of classes given."""
    # Use previously generated colors if num_classes is the same.
    if (hasattr(get_colors_for_classes, "colors") and
            len(get_colors_for_classes.colors) == num_classes):
        return get_colors_for_classes.colors

    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    # colors = [(255,99,71) if c==(255,0,0) else c for c in colors ]  # 单独修正颜色，可去除
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    get_colors_for_classes.colors = colors  # Save colors for future calls.
    return colors


def tracker_detection(tracker, image, centers, xyxy_oneimage,number, max_point_distance=50, max_colors=20, track_id_size=0.8):
    '''
        - max_point_distance为两个点之间的欧式距离不能超过30
            - 有多条轨迹,tracker.tracks;
            - 每条轨迹有多个点,tracker.tracks[i].trace
        - max_colors,最大颜色数量
        - track_id_size,每个中心点，显示的标记字体大小
    '''
    track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
               (0, 255, 255), (255, 0, 255), (255, 127, 255),
               (127, 0, 255), (127, 0, 127), (127, 255, 255),
               (255, 255, 127), (127, 255, 0), (127, 127, 0),
               (0, 127, 127), (255, 127, 0), (255, 0, 127)]
    # track_colors = get_colors_for_classes(max_colors)

    result = np.asarray(image)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.putText(result, str(number), (20, 40), font, 1, (0, 0, 255), 5)  # 左上角，人数计数

    if (len(centers) > 0):
        # Track object using Kalman Filter
        tracker.Update(centers,xyxy_oneimage)
        # For identified object tracks draw tracking line
        # Use various colors to indicate different track_id
        for i in range(len(tracker.tracks)):
            # 多个轨迹
            if (len(tracker.tracks[i].trace) >= 1):
                # x0, y0 = tracker.tracks[i].trace[-1][0][0], tracker.tracks[i].trace[-1][1][0]
                # cv2.putText(result, str(tracker.tracks[i].track_id), (int(x0), int(y0)), font, track_id_size,
                #             (255, 255, 255), 4)
                # (image,text,(x,y),font,size,color,粗细)

                #draw the last boundingbox of each trace instead of all
                clr = tracker.tracks[i].track_id % 10
                names = 'cell'
                colors = track_colors[clr]
                idx = len(tracker.tracks[i].trace) - 1
                xyxy = tracker.tracks[i].trace_boundingbox[idx]
                label = f'{names}'
                plot_one_box(xyxy, result, label=label, color=colors, line_thickness=1)
                #
                # for j in range(len(tracker.tracks[i].trace) - 1):
                #     # 每条轨迹的每个点
                #     # Draw trace line
                #     x1 = tracker.tracks[i].trace[j][0][0]
                #     y1 = tracker.tracks[i].trace[j][0][1]
                #     x2 = tracker.tracks[i].trace[j + 1][0][0]
                #     y2 = tracker.tracks[i].trace[j + 1][0][1]
                #     # clr = tracker.tracks[i].track_id % 9
                #     distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                #     if distance < max_point_distance:
                #         cv2.line(result, (int(x1), int(y1)), (int(x2), int(y2)),
                #                  track_colors[clr], 4)
                #     # draw boundingboxes
                #     names = 'cell'
                #     colors =track_colors[clr]
                #     xyxy=tracker.tracks[i].trace_boundingbox[j]
                #     label = f'{names}'
                #
                #     plot_one_box(xyxy, result, label=label, color=colors, line_thickness=1)

    return tracker, result

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    # tracker info
    tracker = Tracker(50, 4, 500, 100)

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)


        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = Path(path[i]), '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = Path(path), '', im0s, getattr(dataset, 'frame', 0)

            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f'{n} {names[int(c)]}s, '  # add to string

                # Write detection results
                for *xyxy, conf, cls in reversed(det):  # traverse each boundingbox in the det
                    if save_txt:  # Write to file
                        if  float(conf) > 0.7 :
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    # if save_img or view_img:  # Add bbox to image
                    #     label = f'{names[int(cls)]}'
                    #     plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                # process tracking
                xyxy_oneimage=[]
                cls_oneimage=[]
                conf_oneimage=[]
                for *xyxy, conf, cls in reversed(det):
                    if float(conf) > 0.7:
                        # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        xyxy_oneimage.append(xyxy)
                        cls_oneimage.append(int(cls))
                        conf_oneimage.append(float(conf))

                # imgoriginalsize = im0.shape[1]
                centers, number = calc_center(xyxy_oneimage, cls_oneimage, conf_oneimage, score_limit=0.7)

                for center in centers:
                    cv2.circle(im0, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)

                tracker, result = tracker_detection(tracker, im0, centers, xyxy_oneimage,number, max_point_distance=50)
                cv2.imwrite(save_path, result, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # # Save results (image with detections)
            # if save_img:
            #     if dataset.mode == 'image':
            #         cv2.imwrite(save_path, im0)
            #     else:  # 'video'
            #         if vid_path != save_path:  # new video
            #             vid_path = save_path
            #             if isinstance(vid_writer, cv2.VideoWriter):
            #                 vid_writer.release()  # release previous video writer
            #
            #             fourcc = 'mp4v'  # output video codec
            #             fps = vid_cap.get(cv2.CAP_PROP_FPS)
            #             w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            #             h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #             vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            #         vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
