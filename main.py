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

import numpy as np
import time
import json
import torch
import sys
import cv2
import os

from CameraViewer import CameraViewer
from yoloPre import YoloPredictor

"""
主函数:
    1. MainWindow(QMainWindow, Ui_MainWindow)：主窗口
    2. show_image(img_src, label)：主窗口显示原始图像和检测结果
    3. run_or_continue(self)：控制开始/暂停
    4. show_status(self, msg)：底部状态栏信息
    5. open_src_file(self)：选择本地文件
    6. chose_cam(self)：选择摄像头
    7. chose_rtsp(self)：选择网络源
    8. load_rtsp(self, ip)：加载网络源
    9. is_save_res(self)：保存测试结果按钮--图片/视频
    10. is_save_txt(self)：保存测试结果按钮--标签(txt)
    11. load_config(self)：配置初始化
    12. stop(self)：终止按钮及相关状态
    13. change_val(self, x, flag)：更改检测参数
    14. change_model(self,x)：更换模型
    15. show_labels(self, labels_dic)：显示检测结果
    16. ModelBoxRefre(self)：循环监控模型文件变化
    17. mousePressEvent(self, event)：获取鼠标位置（用于按住标题栏拖动窗口）
    18. resizeEvent(self, event)：拖动窗口大小时优化调整
    19. closeEvent(self, event)：退出线程，保存设置
"""


class MainWindow(QMainWindow, Ui_MainWindow):
    main2yolo_begin_sgl = Signal()  # The main window sends an execution signal to the yolo instance

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        # basic interface
        self.setupUi(self)

        self.setAttribute(Qt.WA_TranslucentBackground)  # rounded transparent
        self.setWindowFlags(Qt.FramelessWindowHint)  # Set window flag: hide window borders
        UIFuncitons.uiDefinitions(self)

        # read model folder
        self.pt_list = os.listdir('./models')
        self.pt_list = [file for file in self.pt_list if file.endswith('.pt')]
        self.pt_list.sort(key=lambda x: os.path.getsize('./models/' + x))  # sort by file size
        self.model_box.clear()
        self.model_box.addItems(self.pt_list)
        self.Qtimer_ModelBox = QTimer(self)  # Timer: Monitor model file changes every 2 seconds
        self.Qtimer_ModelBox.timeout.connect(self.ModelBoxRefre)
        self.Qtimer_ModelBox.start(2000)

        # Yolo-v8 thread
        self.yolo_predict = YoloPredictor()  # Create a Yolo instance
        self.select_model = self.model_box.currentText()  # default model
        self.yolo_predict.new_model_name = "./models/%s" % self.select_model
        self.yolo_thread = QThread()  # Create yolo thread

        # self.yolo_predict.yolo2main_pre_img.connect(lambda x: self.show_image(x, self.pre_video))
        self.yolo_predict.yolo2main_res_img.connect(lambda x: self.show_image(x, self.res_video))
        self.yolo_predict.yolo2main_status_msg.connect(lambda x: self.show_status(x))
        self.yolo_predict.yolo2main_fps.connect(lambda x: self.fps_label.setText(x))
        # self.yolo_predict.yolo2main_labels.connect(self.show_labels)
        self.yolo_predict.yolo2main_class_num.connect(lambda x: self.Class_num.setText(str(x)))
        self.yolo_predict.yolo2main_target_num.connect(lambda x: self.Target_num.setText(str(x)))
        self.yolo_predict.yolo2main_progress.connect(lambda x: self.progress_bar.setValue(x))

        # 将主线程的信号绑定到 Yolo 类的槽函数上
        self.main2yolo_begin_sgl.connect(self.yolo_predict.run)
        self.yolo_predict.moveToThread(self.yolo_thread)

        # Model parameters
        self.model_box.currentTextChanged.connect(self.change_model)
        self.iou_spinbox.valueChanged.connect(lambda x: self.change_val(x, 'iou_spinbox'))  # iou box
        self.iou_slider.valueChanged.connect(lambda x: self.change_val(x, 'iou_slider'))  # iou scroll bar
        self.conf_spinbox.valueChanged.connect(lambda x: self.change_val(x, 'conf_spinbox'))  # conf box
        self.conf_slider.valueChanged.connect(lambda x: self.change_val(x, 'conf_slider'))  # conf scroll bar
        self.speed_spinbox.valueChanged.connect(lambda x: self.change_val(x, 'speed_spinbox'))  # speed box
        self.speed_slider.valueChanged.connect(lambda x: self.change_val(x, 'speed_slider'))  # speed scroll bar

        # Prompt window initialization
        self.Class_num.setText('--')
        self.Target_num.setText('--')
        self.fps_label.setText('--')
        self.Model_name.setText(self.select_model)

        # Select detection source
        self.src_file_button.clicked.connect(self.open_src_file)  # select local file
        self.src_cam_button.clicked.connect(self.chose_cam)  # chose_camera

        # start testing button
        self.run_button.clicked.connect(self.run_or_continue)  # pause/start
        self.stop_button.clicked.connect(self.stop)  # termination

        # Other function buttons
        self.save_res_button.toggled.connect(self.is_save_res)  # save image option
        self.save_txt_button.toggled.connect(self.is_save_txt)  # Save label option

        self.ToggleBotton.clicked.connect(lambda: UIFuncitons.toggleMenu(self, True))  # left navigation button
        self.settings_button.clicked.connect(lambda: UIFuncitons.settingBox(self, True))  # top right settings button

        # initialization
        self.load_config()

    @staticmethod
    def show_image(img_src, label):
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()

            if iw / w > ih / h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))
            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))


        except Exception as e:
            print(repr(e))

    # Control start/pause
    def run_or_continue(self):

        if self.yolo_predict.source == '':
            # print('Please select a video source')
            print('请选择视频源')
            # self.show_status('Please select a video source before starting detection...')
            self.show_status('请在开始检测前选择视频源...')
            self.run_button.setChecked(False)
        else:
            self.yolo_predict.stop_dtc = False
            print('start')
            if self.run_button.isChecked():
                self.run_button.setChecked(True)  # start button
                self.save_txt_button.setEnabled(False)  # It is forbidden to check and save after starting the detection
                self.save_res_button.setEnabled(False)
                self.show_status('Detecting...')
                self.yolo_predict.continue_dtc = True  # Control whether Yolo is paused

                if not self.yolo_thread.isRunning():
                    self.yolo_thread.start()
                    self.main2yolo_begin_sgl.emit()

            else:
                self.yolo_predict.continue_dtc = False
                self.show_status("Pause...")
                self.run_button.setChecked(False)  # start button

    # bottom status bar information
    def show_status(self, msg):
        self.status_bar.setText(msg)
        print(msg)
        if msg == 'Detection completed' or msg == '检测完成':
            self.save_res_button.setEnabled(True)
            self.save_txt_button.setEnabled(True)
            self.run_button.setChecked(False)
            self.progress_bar.setValue(0)
            if self.yolo_thread.isRunning():
                self.yolo_thread.quit()
        elif msg == 'Detection terminated!' or msg == '检测终止':
            self.save_res_button.setEnabled(True)
            self.save_txt_button.setEnabled(True)
            self.run_button.setChecked(False)
            self.progress_bar.setValue(0)
            if self.yolo_thread.isRunning():
                self.yolo_thread.quit()
                # self.pre_video.clear()
            self.res_video.clear()
            self.Class_num.setText('--')
            self.Target_num.setText('--')
            self.fps_label.setText('--')

    # select local file
    def open_src_file(self):
        print('local file')
        config_file = 'config/fold.json'
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        open_fold = config['open_fold']
        if not os.path.exists(open_fold):
            open_fold = os.getcwd()

        name, _ = QFileDialog.getOpenFileName(self, 'Video/image', open_fold,
                                              "Pic File(*.mp4 *.mkv *.avi *.flv *.jpg *.png)")
        if name:
            self.yolo_predict.source = name
            self.show_status('Load File：{}'.format(os.path.basename(name)))
            config['open_fold'] = os.path.dirname(name)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
            self.stop()

    def chose_cam(self):
        try:
            print('open camera')
            self.yolo_predict.camera_run()

        except Exception as e:
            print(e)
            self.show_status('%s' % e)

    # select network source
    def chose_rtsp(self):
        self.rtsp_window = Window()
        config_file = 'config/ip.json'
        if not os.path.exists(config_file):
            ip = "rtsp://admin:admin888@192.168.1.2:555"
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            ip = config['ip']
        self.rtsp_window.rtspEdit.setText(ip)
        self.rtsp_window.show()
        self.rtsp_window.rtspButton.clicked.connect(lambda: self.load_rtsp(self.rtsp_window.rtspEdit.text()))

    # load network sources
    def load_rtsp(self, ip):
        try:
            self.stop()
            MessageBox(
                self.close_button, title='提示', text='加载 rtsp...', time=1000, auto=True).exec()
            self.yolo_predict.source = ip
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open('config/ip.json', 'w', encoding='utf-8') as f:
                f.write(new_json)
            self.show_status('Loading rtsp：{}'.format(ip))
            self.rtsp_window.close()
        except Exception as e:
            self.show_status('%s' % e)

    # Save test result button--picture/video
    def is_save_res(self):
        if self.save_res_button.checkState() == Qt.CheckState.Unchecked:
            self.show_status('NOTE: Run image results are not saved.')
            self.yolo_predict.save_res = False
        elif self.save_res_button.checkState() == Qt.CheckState.Checked:
            self.show_status('NOTE: Run image results will be saved.')
            self.yolo_predict.save_res = True

    def is_save_txt(self):
        if self.save_txt_button.checkState() == Qt.CheckState.Unchecked:
            self.show_status('NOTE: Labels results are not saved.')
            self.yolo_predict.save_txt = False
        elif self.save_txt_button.checkState() == Qt.CheckState.Checked:
            self.show_status('NOTE: Labels results will be saved.')
            self.yolo_predict.save_txt = True

    def load_config(self):
        config_file = 'config/setting.json'
        if not os.path.exists(config_file):
            iou = 0.26
            conf = 0.33
            rate = 10
            save_res = 0
            save_txt = 0
            new_config = {"iou": iou,
                          "conf": conf,
                          "rate": rate,
                          "save_res": save_res,
                          "save_txt": save_txt
                          }
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            if len(config) != 5:
                iou = 0.26
                conf = 0.33
                rate = 10
                save_res = 0
                save_txt = 0
            else:
                iou = config['iou']
                conf = config['conf']
                rate = config['rate']
                save_res = config['save_res']
                save_txt = config['save_txt']
        self.save_res_button.setCheckState(Qt.CheckState(save_res))
        self.yolo_predict.save_res = (False if save_res == 0 else True)
        self.save_txt_button.setCheckState(Qt.CheckState(save_txt))
        self.yolo_predict.save_txt = (False if save_txt == 0 else True)
        self.run_button.setChecked(False)
        self.show_status("s1--")

    # Terminate button and associated state
    def stop(self):
        print('stop')
        if self.yolo_thread.isRunning():
            self.yolo_thread.quit()  # end thread
        self.yolo_predict.stop_dtc = True
        self.run_button.setChecked(False)  # start key recovery
        self.save_res_button.setEnabled(True)  # Ability to use the save button
        self.save_txt_button.setEnabled(True)  # Ability to use the save button
        # self.pre_video.clear()           # clear image display
        self.res_video.clear()  # clear image display
        self.progress_bar.setValue(0)
        self.Class_num.setText('--')
        self.Target_num.setText('--')
        self.fps_label.setText('--')

    # Change detection parameters
    def change_val(self, x, flag):
        if flag == 'iou_spinbox':
            self.iou_slider.setValue(int(x * 100))  # The box value changes, changing the slider
        elif flag == 'iou_slider':
            self.iou_spinbox.setValue(x / 100)  # The slider value changes, changing the box
            self.show_status('IOU Threshold: %s' % str(x / 100))
            self.yolo_predict.iou_thres = x / 100
        elif flag == 'conf_spinbox':
            self.conf_slider.setValue(int(x * 100))
        elif flag == 'conf_slider':
            self.conf_spinbox.setValue(x / 100)
            self.show_status('Conf Threshold: %s' % str(x / 100))
            self.yolo_predict.conf_thres = x / 100
        elif flag == 'speed_spinbox':
            self.speed_slider.setValue(x)
        elif flag == 'speed_slider':
            self.speed_spinbox.setValue(x)
            self.show_status('Delay: %s ms' % str(x))
            self.yolo_predict.speed_thres = x  # ms

    # change model 【更换模型】
    def change_model(self, x):
        self.select_model = self.model_box.currentText()
        self.yolo_predict.new_model_name = "./models/%s" % self.select_model
        self.show_status('Change Model：%s' % self.select_model)
        self.Model_name.setText(self.select_model)

    def show_labels(self, labels_dic):
        try:
            self.result_label.clear()
            labels_dic = sorted(labels_dic.items(), key=lambda x: x[1], reverse=True)
            labels_dic = [i for i in labels_dic if i[1] > 0]
            result = [' ' + str(i[0]) + '：' + str(i[1]) for i in labels_dic]
            self.result_label.addItems(result)
        except Exception as e:
            self.show_status(e)

    def ModelBoxRefre(self):
        pt_list = os.listdir('./models')
        pt_list = [file for file in pt_list if file.endswith('.pt')]
        pt_list.sort(key=lambda x: os.path.getsize('./models/' + x))
        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.model_box.clear()
            self.model_box.addItems(self.pt_list)

    def mousePressEvent(self, event):
        p = event.globalPosition()
        globalPos = p.toPoint()
        self.dragPos = globalPos

    # Optimize the adjustment when dragging the bottom and right edges of the window size
    def resizeEvent(self, event):
        # Update Size Grips
        UIFuncitons.resize_grips(self)

    # Exit thread, save settings
    def closeEvent(self, event):
        config_file = 'config/setting.json'
        config = dict()
        config['iou'] = self.iou_spinbox.value()
        config['conf'] = self.conf_spinbox.value()
        config['rate'] = self.speed_spinbox.value()
        config['save_res'] = (0 if self.save_res_button.checkState() == Qt.Unchecked else 2)
        config['save_txt'] = (0 if self.save_txt_button.checkState() == Qt.Unchecked else 2)
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_json)
        # Exit the process before closing
        if self.yolo_thread.isRunning():
            self.yolo_predict.stop_dtc = True
            self.yolo_thread.quit()
            MessageBox(
                self.close_button, title='提示', text='等待进程分离', time=3000, auto=True).exec()
            sys.exit(0)
        else:
            sys.exit(0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    Home = MainWindow()
    Home.show()
    sys.exit(app.exec())
