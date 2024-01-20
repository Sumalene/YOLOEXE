from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks, ops
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from ultralytics.yolo.utils.torch_utils import smart_inference_mode
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils.checks import check_imshow
from ultralytics.yolo.cfg import get_cfg
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu
from PySide6.QtGui import QImage, QPixmap, QColor
from PySide6.QtCore import QTimer, QThread, Signal, QObject, QPoint, Qt
from ui.CustomMessageBox import MessageBox
from ui.home import Ui_MainWindow
from UIFunctions import *
from collections import defaultdict
from pathlib import Path
from utils.capnums import Camera
from utils.rtsp_win import Window
from PIL import Image

import numpy as np
import time
import json
import torch
import sys
import cv2
import os

"""
yoloPre:该文件是 yolo 的预测文件，用于实现 yolo 的预测功能。
主要方法:
    1. run: 用于实现 yolo 的预测功能。
    2. camera_run: 用于实现 yolo 的预测功能，但是该方法是用于摄像头的预测。
    3. get_annotator: 用于获取 Annotator 对象，该对象用于绘制检测结果。
    4. preprocess: 用于对图像进行预处理。
    5. postprocess: 用于对模型的预测结果进行后处理。
    6. write_results: 用于将模型的预测结果写入到文件中。
"""

class YoloPredictor(BasePredictor, QObject):
    yolo2main_pre_img = Signal(np.ndarray)  # raw image signal
    yolo2main_res_img = Signal(np.ndarray)  # test result signal
    yolo2main_status_msg = Signal(str)  # Detecting/pausing/stopping/testing complete/error reporting signal
    yolo2main_fps = Signal(str)  # fps
    yolo2main_labels = Signal(dict)  # Detected target results (number of each category)
    yolo2main_progress = Signal(int)  # Completeness
    yolo2main_class_num = Signal(int)  # Number of categories detected
    yolo2main_target_num = Signal(int)  # Targets detected

    def __init__(self, cfg=DEFAULT_CFG, overrides=None):
        super(YoloPredictor, self).__init__()
        QObject.__init__(self)

        self.args = get_cfg(cfg, overrides)
        project = self.args.project or Path(SETTINGS['runs_dir']) / self.args.task
        name = f'{self.args.mode}'
        self.save_dir = increment_path(Path(project) / name, exist_ok=self.args.exist_ok)
        self.done_warmup = False
        if self.args.show:
            self.args.show = check_imshow(warn=True)

        self.iscamera = 0
        self.capture_frame = None

        self.used_model_name = None  # The detection model name to use
        self.new_model_name = None  # Models that change in real time
        self.source = ''  # input source
        self.stop_dtc = False  # Termination detection
        self.continue_dtc = True  # pause
        self.save_res = False  # Save test results
        self.save_txt = False  # save label(txt) file
        self.iou_thres = 0.45  # iou
        self.conf_thres = 0.25  # conf
        self.speed_thres = 10  # delay, ms
        self.labels_dict = {}  # return a dictionary of results
        self.progress_value = 0  # progress bar

        # Usable if setup is done
        self.model = None
        self.data = self.args.data  # data_dict
        self.imgsz = None
        self.device = None
        self.dataset = None
        self.vid_path, self.vid_writer = None, None
        self.annotator = None
        self.data_path = None
        self.source_type = None
        self.batch = None
        self.callbacks = defaultdict(list, callbacks.default_callbacks)  # add callbacks
        callbacks.add_integration_callbacks(self)

    @smart_inference_mode()
    def camera_run(self):

        global all_count
        try:
            if self.args.verbose:
                LOGGER.info('')

            # set model
            self.yolo2main_status_msg.emit('Loding Model...')
            print('加载模型')

            if not self.model:
                self.setup_model(self.new_model_name)
                self.used_model_name = self.new_model_name

            # Check save path/label
            print('检查保存路径/标签')
            if self.save_res or self.save_txt:
                (self.save_dir / 'labels' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

            self.seen, self.windows, self.dt, self.batch = 0, [], (ops.Profile(), ops.Profile(), ops.Profile()), None
            print('start detection')
            # start detection

            count = 0  # run location frame
            start_time = time.time()  # used to calculate the frame rate

            # ------------------
            self.capture = cv2.VideoCapture(0)

            while True:
                time.sleep(0.03)
                ret, frame = self.capture.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.capture_frame = Image.fromarray(frame_rgb)

                # print('设置资源模式')
                print(self.capture_frame)
                self.setup_source(self.capture_frame)

                if not self.done_warmup:
                    self.model.warmup(
                        imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))

                    self.done_warmup = True

                batch = iter(self.dataset)
                if self.continue_dtc:
                    self.yolo2main_status_msg.emit('Detecting...')
                    batch = next(self.dataset)  # next data

                    self.batch = batch
                    path, im, im0s, vid_cap, s = batch
                    visualize = increment_path(self.save_dir / Path(path).stem,
                                               mkdir=True) if self.args.visualize else False

                    # Calculation completion and frame rate (to be optimized)
                    count += 1  # frame count +1
                    all_count = 87
                    self.progress_value = int(count / all_count * 1000)
                    if count % 5 == 0 and count >= 5:
                        self.yolo2main_fps.emit(str(int(5 / (time.time() - start_time))))
                        start_time = time.time()

                    # print('preprocess...')
                    with self.dt[0]:
                        im = self.preprocess(im)
                        if len(im.shape) == 3:
                            im = im[None]
                    # inference
                    with self.dt[1]:
                        preds = self.model(im, augment=self.args.augment, visualize=visualize)
                    # postprocess
                    with self.dt[2]:

                        self.results = self.postprocess(preds, im, im0s)

                    # print('visualize, save, write results...')
                    n = len(im)  # To be improved: support multiple img

                    for i in range(n):
                        self.results[i].speed = {
                            'preprocess': self.dt[0].dt * 1E3 / n,
                            'inference': self.dt[1].dt * 1E3 / n,
                            'postprocess': self.dt[2].dt * 1E3 / n
                        }
                        p, im0 = (path[i], im0s[i].copy()) if self.source_type.webcam or self.source_type.from_img \
                            else (path, im0s.copy())

                        p = Path(p)  # the source dir

                        # s:::   video 1/1 (6/6557) 'path':
                        # must, to get boxs\labels
                        label_str = self.write_results(i, self.results, (p, im, im0))  # labels   /// original :s +=

                        # labels and nums dict
                        class_nums = 0
                        target_nums = 0
                        self.labels_dict = {}
                        if 'no detections' in label_str:
                            pass
                        else:
                            for ii in label_str.split(',')[:-1]:
                                nums, label_name = ii.split('~')

                                li = nums.split(':')[-1]

                                print(li)

                                self.labels_dict[label_name] = int(li)
                                target_nums += int(li)
                                class_nums += 1

                        # save img or video result
                        if self.save_res:
                            self.save_preds(vid_cap, i, str(self.save_dir / p.name))
                        self.yolo2main_res_img.emit(im0)  # after detection
                        self.yolo2main_pre_img.emit(im0s if isinstance(im0s, np.ndarray) else im0s[0])  # Before testing
                        self.yolo2main_class_num.emit(class_nums)
                        self.yolo2main_target_num.emit(target_nums)

                        if self.speed_thres != 0:
                            time.sleep(self.speed_thres / 1000)  # delay , ms

                    self.yolo2main_progress.emit(self.progress_value)  # progress bar

                # Detection completed
                if count + 1 >= all_count:
                    if isinstance(self.vid_writer[-1], cv2.VideoWriter):
                        self.vid_writer[-1].release()  # release final video writer
                    self.yolo2main_status_msg.emit('Detection completed')
                    break

        except Exception as e:
            self.capture.release()
            print('error:', e)
            self.yolo2main_status_msg.emit('%s' % e)

    # main for detect
    @smart_inference_mode()
    def run(self):
        try:

            if self.args.verbose:
                LOGGER.info('')

            # set model
            self.yolo2main_status_msg.emit('Loding Model...')

            if not self.model:
                self.setup_model(self.new_model_name)
                self.used_model_name = self.new_model_name

            print('设置资源模式')
            self.setup_source(self.source if self.source is not None else self.args.source)

            # Check save path/label
            print('检查保存路径/标签')
            if self.save_res or self.save_txt:
                (self.save_dir / 'labels' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

            # warmup model

            if not self.done_warmup:
                self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
                self.done_warmup = True

            self.seen, self.windows, self.dt, self.batch = 0, [], (ops.Profile(), ops.Profile(), ops.Profile()), None

            print('start')

            count = 0  # run location frame
            start_time = time.time()  # used to calculate the frame rate
            batch = iter(self.dataset)
            while True:
                if self.stop_dtc:
                    if isinstance(self.vid_writer[-1], cv2.VideoWriter):
                        self.vid_writer[-1].release()  # release final video writer
                    self.yolo2main_status_msg.emit('Detection terminated!')
                    break
                if self.used_model_name != self.new_model_name:
                    self.setup_model(self.new_model_name)
                    self.used_model_name = self.new_model_name

                if self.continue_dtc:
                    self.yolo2main_status_msg.emit('Detecting...')
                    batch = next(self.dataset)

                    self.batch = batch
                    path, im, im0s, vid_cap, s = batch
                    visualize = increment_path(self.save_dir / Path(path).stem,
                                               mkdir=True) if self.args.visualize else False
                    count += 1
                    if vid_cap:
                        all_count = vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    else:
                        all_count = 1
                    self.progress_value = int(count / all_count * 1000)  # progress bar(0~1000)
                    if count % 5 == 0 and count >= 5:  # Calculate the frame rate every 5 frames
                        self.yolo2main_fps.emit(str(int(5 / (time.time() - start_time))))
                        start_time = time.time()

                    with self.dt[0]:
                        im = self.preprocess(im)
                        if len(im.shape) == 3:
                            im = im[None]  # 扩大批量调暗
                    # inference
                    with self.dt[1]:
                        preds = self.model(im, augment=self.args.augment, visualize=visualize)

                    with self.dt[2]:
                        self.results = self.postprocess(preds, im, im0s)

                    n = len(im)
                    for i in range(n):
                        self.results[i].speed = {
                            'preprocess': self.dt[0].dt * 1E3 / n,
                            'inference': self.dt[1].dt * 1E3 / n,
                            'postprocess': self.dt[2].dt * 1E3 / n}
                        p, im0 = (path[i], im0s[i].copy()) if self.source_type.webcam or self.source_type.from_img \
                            else (path, im0s.copy())

                        p = Path(p)  # the source dir

                        label_str = self.write_results(i, self.results, (p, im, im0))

                        # labels and nums dict
                        class_nums = 0
                        target_nums = 0
                        self.labels_dict = {}
                        if 'no detections' in label_str:
                            pass
                        else:
                            for ii in label_str.split(',')[:-1]:
                                nums, label_name = ii.split('~')
                                self.labels_dict[label_name] = int(nums)
                                target_nums += int(nums)
                                class_nums += 1

                        # save img or video result
                        if self.save_res:
                            self.save_preds(vid_cap, i, str(self.save_dir / p.name))

                        self.yolo2main_res_img.emit(im0)
                        self.yolo2main_pre_img.emit(im0s if isinstance(im0s, np.ndarray) else im0s[0])
                        # self.yolo2main_labels.emit(self.labels_dict)
                        self.yolo2main_class_num.emit(class_nums)
                        self.yolo2main_target_num.emit(target_nums)

                        print('send success!')

                        if self.speed_thres != 0:
                            time.sleep(self.speed_thres / 1000)

                    self.yolo2main_progress.emit(self.progress_value)

                # Detection completed
                if count + 1 >= all_count:
                    if isinstance(self.vid_writer[-1], cv2.VideoWriter):
                        self.vid_writer[-1].release()
                    self.yolo2main_status_msg.emit('Detection completed')
                    break

        except Exception as e:
            print(e)
            self.yolo2main_status_msg.emit('%s' % e)

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        """
        :param preds:  模型的预测结果
        :param img:   模型的输入图像
        :param orig_img:  原始图像
        :return: 返回模型的预测结果
        """
        preds = ops.non_max_suppression(preds,
                                        self.conf_thres,
                                        self.iou_thres,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_img[i] if isinstance(orig_img, list) else orig_img
            shape = orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
            path, _, _, _, _ = self.batch
            img_path = path[i] if isinstance(path, list) else path
            results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred))
        # print(results)

        return results

    def write_results(self, idx, results, batch):
        """
        :param idx:  用于标记当前图像的索引
        :param results:  模型的预测结果
        :param batch:
        :return:
        """
        p, im, im0 = batch
        log_string = ''
        if len(im.shape) == 3:
            im = im[None]
        self.seen += 1
        imc = im0.copy() if self.args.save_crop else im0
        if self.source_type.webcam or self.source_type.from_img:
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)
        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        self.annotator = self.get_annotator(im0)

        det = results[idx].boxes

        if len(det) == 0:
            return f'{log_string}(no detections), '

        for c in det.cls.unique():
            n = (det.cls == c).sum()
            log_string += f"{n}~{self.model.names[int(c)]},"

        # write
        for d in reversed(det):
            cls, conf = d.cls.squeeze(), d.conf.squeeze()
            if self.save_txt:
                line = (cls, *(d.xywhn.view(-1).tolist()), conf) \
                    if self.args.save_conf else (cls, *(d.xywhn.view(-1).tolist()))
                with open(f'{self.txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')
            if self.save_res or self.args.save_crop or self.args.show or True:
                c = int(cls)  # integer class
                name = f'id:{int(d.id.item())} {self.model.names[c]}' if d.id is not None else self.model.names[c]
                label = None if self.args.hide_labels else (name if self.args.hide_conf else f'{name} {conf:.2f}')
                self.annotator.box_label(d.xyxy.squeeze(), label, color=colors(c, True))
            if self.args.save_crop:
                save_one_box(d.xyxy,
                             imc,
                             file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
                             BGR=True)

        return log_string
