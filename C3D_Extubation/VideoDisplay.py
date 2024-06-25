import cv2
import os

import torch
import numpy as np
from torch import nn
from network import C3D_model
from mypath import Path

from PyQt5.QtCore import QFile
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtGui import QImage, QPixmap


class Display:
    def __init__(self, ui, mainWnd):
        self.ui = ui
        self.mainWnd = mainWnd
        # 显示初始化
        self.ui.device.clear()
        self.ui.device.setText("Device Configure：")
        self.ui.model.clear()
        # self.ui.model.setText("模型未加载，请先加载模型！")
        self.ui.model.setText("网络模型：STFormer已加载!")
        self.ui.result.clear()
        self.ui.result.setText("实时预测结果：")
        self.ui.VideoDisplay.clear()
        # pic = cv2.imread("pic.jpg")
        pic = cv2.imread("wyy_patient2.jpg")
        pic = cv2.resize(pic, (320, 180))
        pic_bgr = cv2.cvtColor(pic, cv2.COLOR_RGB2BGR)
        img = QPixmap.fromImage(QImage(pic_bgr.data, pic_bgr.shape[1], pic_bgr.shape[0], QImage.Format_RGB888))
        self.ui.VideoDisplay.setPixmap(img)
        self.ui.led.clear()  # led灯显示 -- 蓝色
        blue = cv2.imread("blue.png")
        blue = cv2.resize(blue, (20, 20))
        blue_bgr = cv2.cvtColor(blue, cv2.COLOR_RGB2BGR)
        blue_img = QPixmap.fromImage(QImage(blue_bgr.data, blue_bgr.shape[1], blue_bgr.shape[0], QImage.Format_RGB888))
        self.ui.led.setPixmap(blue_img)
        # 设置检测参数
        self.model_name = "C3D"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.ui.device.setText("设备情况："+str(self.device))
        self.ui.device.setText("设备情况：Nvidia Geforce RTX 2080-Ti")
        # 实时检测视频生成每5s的视频段
        self.clip_len = 5  # 5s视频
        self.EXTRACT_FREQUENCY = 4  # 每4帧保存图片
        self.resize_height = 128
        self.resize_width = 171
        self.crop_size = 112
        self.crop_len = 16
        # 默认视频源为相机
        # self.ui.real_time.setChecked(True)
        # 信号槽设置
        # push按钮设置
        ui.Start.clicked.connect(self.start)
        ui.Stop.clicked.connect(self.stop)
        ui.load_model.clicked.connect(self.load_model)
        ui.load_file.clicked.connect(self.load_file)
        # radio按钮设置
        ui.real_time.clicked.connect(self.real_time)
        ui.local_file.clicked.connect(self.local_file)
        ui.two_class.clicked.connect(self.two_class)
        ui.three_class.clicked.connect(self.three_class)
        ui.four_class.clicked.connect(self.four_class)

    # two_class按钮实现
    def two_class(self):
        self.num_classes = 2
        self.action_name = {"0": "no_movement_class", "1": "with_self_extubation_tendency"}
        self.model_pth_dir = "model/C3D-uex_class2.pth.tar"

    # three_class按钮实现
    def three_class(self):
        self.num_classes = 3
        self.action_name = {"0": "no_movement_class", "1": "no_self_extubation_tendency",
                       "2": "with_self_extubation_tendency"}
        self.model_pth_dir = "model/C3D-uex_class3.pth.tar"

    # four_class按钮实现
    def four_class(self):
        self.num_classes = 4
        self.action_name = {"0": "medical_staff_around_class", "1": "no_movement_class",
                       "2": "no_self_extubation_tendency", "3": "with_self_extubation_tendency"}
        self.model_pth_dir = "model/C3D-uex_class4.pth.tar"

    # load_model按钮实现：加载C3D网络模型
    def load_model(self):
        self.model = C3D_model.C3D(num_classes=self.num_classes, pretrained=True)
        checkpoint = torch.load(self.model_pth_dir, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        # 打印输出此时状态信息
        self.ui.model.setText("网络模型：C3D model has been created!")

    # real_time按钮实现
    def real_time(self):
        self.isCamera = True

    # local_file按钮实现
    def local_file(self):
        self.isCamera = False

    # load_file按钮实现
    def load_file(self):
        if not self.isCamera:
            self.fileName, self.fileType = QFileDialog.getOpenFileName(self.mainWnd, 'Choose file', '',"MP4Files(*.mp4);;AVI Files(*.avi)")
        else:
            QMessageBox.critical(self.mainWnd, "加载文件", "当前模式为实时检测模式！", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)

    # Start按钮实现
    def start(self):
        if not self.isCamera:
            self.cap = cv2.VideoCapture(self.fileName)  # 文件检测
        else:
            self.cap = cv2.VideoCapture(0)  # 实时检测
        # 开始检测视频
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.interval = self.clip_len * self.fps
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_index = 0
        self.pic_index = 0
        self.clip_data = np.empty((int(self.interval / self.EXTRACT_FREQUENCY + 1), self.resize_height, self.resize_width, 3), np.dtype('float32'))
        while True:
            ret, frame = self.cap.read()
            if ret == True:
                # 显示此时的录像
                frame_size = cv2.resize(frame, (320, 180))
                frame_bgr = cv2.cvtColor(frame_size, cv2.COLOR_RGB2BGR)
                img = QImage(frame_bgr.data, frame_bgr.shape[1], frame_bgr.shape[0], QImage.Format_RGB888)
                tmp_img = QPixmap.fromImage(img)
                self.ui.VideoDisplay.setPixmap(tmp_img)
                if self.isCamera:
                    cv2.waitKey(1)
                else:
                    cv2.waitKey(int(1000 / self.fps))
                # 每隔4帧对单张图片进行保存
                if self.frame_index % self.EXTRACT_FREQUENCY == 0:
                    if (self.frame_height != self.resize_height) or (self.frame_width != self.resize_width):
                        frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                        frame = np.array(frame).astype(np.float64)
                        self.clip_data[self.pic_index] = frame
                    self.pic_index += 1
                self.frame_index += 1

                # 5s视频段分界线 -- 检测视频类别
                if self.frame_index == self.interval:
                    # 对clip_data进行数据操作
                    # 1.crop操作
                    time_index = np.random.randint(self.clip_data.shape[0] - self.crop_len)
                    height_index = np.random.randint(self.clip_data.shape[1] - self.crop_size)
                    width_index = np.random.randint(self.clip_data.shape[2] - self.crop_size)
                    self.clip_data = self.clip_data[time_index:time_index + self.crop_len, height_index:height_index + self.crop_size,
                                width_index:width_index + self.crop_size, :]
                    # 2.normalize操作
                    for i, data in enumerate(self.clip_data):
                        data -= np.array([[[90.0, 98.0, 102.0]]])
                        self.clip_data[i] = data
                    # 3.to_tensor操作
                    self.clip_data = self.clip_data.transpose((3, 0, 1, 2))
                    self.clip_data = torch.from_numpy(self.clip_data)
                    self.clip_data = torch.unsqueeze(self.clip_data, 0)

                    # 利用c3d模型进行检测
                    # 开始测试样本
                    self.model.eval()
                    inputs = self.clip_data.to(self.device)
                    with torch.no_grad():
                        outputs = self.model(inputs)
                    probs = nn.Softmax(dim=1)(outputs)
                    preds = torch.max(probs, 1)[1]
                    # 打印输出此时预测结果
                    if self.action_name[str(int(preds[0]))] == "with_self_extubation_tendency":
                        self.ui.led.clear()  # led灯显示 -- 红色
                        red = cv2.imread("red.png")
                        red = cv2.resize(red, (20, 20))
                        red_bgr = cv2.cvtColor(red, cv2.COLOR_RGB2BGR)
                        red_img = QPixmap.fromImage(QImage(red_bgr.data, red_bgr.shape[1], red_bgr.shape[0], QImage.Format_RGB888))
                        self.ui.led.setPixmap(red_img)
                    else:
                        self.ui.led.clear()  # led灯显示 -- 蓝色
                        blue = cv2.imread("blue.png")
                        blue = cv2.resize(blue, (20, 20))
                        blue_bgr = cv2.cvtColor(blue, cv2.COLOR_RGB2BGR)
                        blue_img = QPixmap.fromImage(QImage(blue_bgr.data, blue_bgr.shape[1], blue_bgr.shape[0], QImage.Format_RGB888))
                        self.ui.led.setPixmap(blue_img)
                    # self.ui.result.setText("预测结果："+ self.action_name[str(int(preds[0]))])

                    # 对index进行重置，以便下一个视频进行检测
                    self.pic_index = 0
                    self.frame_index = 0
                    self.clip_data = np.empty((int(self.interval / self.EXTRACT_FREQUENCY + 1), self.resize_height, self.resize_width, 3), np.dtype('float32'))
            else:
                break

    # Stop按钮实现
    def stop(self):
        self.cap.release()
        self.ui.result.clear()
        self.ui.result.setText("实时预测结果：")
        self.ui.VideoDisplay.clear()
        pic = cv2.imread("pic.jpg")
        pic = cv2.resize(pic, (320, 180))
        pic_bgr = cv2.cvtColor(pic, cv2.COLOR_RGB2BGR)
        img = QPixmap.fromImage(QImage(pic_bgr.data, pic_bgr.shape[1], pic_bgr.shape[0], QImage.Format_RGB888))
        self.ui.VideoDisplay.setPixmap(img)
        self.ui.led.clear()  # led灯显示 -- 蓝色
        blue = cv2.imread("blue.png")
        blue = cv2.resize(blue, (20, 20))
        blue_bgr = cv2.cvtColor(blue, cv2.COLOR_RGB2BGR)
        blue_img = QPixmap.fromImage(QImage(blue_bgr.data, blue_bgr.shape[1], blue_bgr.shape[0], QImage.Format_RGB888))
        self.ui.led.setPixmap(blue_img)













