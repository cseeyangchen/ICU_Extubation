import cv2
import time
import threading
import multiprocessing as mp
import torch
import numpy as np
from network import C3D_model

from PyQt5.QtCore import QFile
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtGui import QImage, QPixmap

from actionRecognition import Recognition



class Display():
    def __init__(self, ui, mainWnd):
        self.ui = ui
        self.mainWnd = mainWnd
        # 显示初始化
        self.ui.device.clear()
        self.ui.device.setText("设备情况：")
        self.ui.model.clear()
        self.ui.model.setText("模型未加载，请先加载模型！")

        # camera初始化显示电子科技大学图标
        self.camera_list = [self.ui.camera1,self.ui.camera2]
        for camera in self.camera_list:
            camera.clear()
        pic = cv2.imread("pic.jpg")
        pic = cv2.resize(pic, (120, 90))
        pic_bgr = cv2.cvtColor(pic, cv2.COLOR_RGB2BGR)
        img = QPixmap.fromImage(QImage(pic_bgr.data, pic_bgr.shape[1], pic_bgr.shape[0], QImage.Format_RGB888))
        for camera in self.camera_list:
            camera.setPixmap(img)

        # led灯显示 -- 蓝色
        self.led_list = [self.ui.led1,self.ui.led2]
        for led in self.led_list:
            led.clear()
        blue = cv2.imread("blue.png")
        blue = cv2.resize(blue, (20, 20))
        blue_bgr = cv2.cvtColor(blue, cv2.COLOR_RGB2BGR)
        blue_img = QPixmap.fromImage(QImage(blue_bgr.data, blue_bgr.shape[1], blue_bgr.shape[0], QImage.Format_RGB888))
        for led in self.led_list:
            led.setPixmap(blue_img)

        # 设置检测参数
        self.model_name = "C3D"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ui.device.setText("设备情况："+str(self.device))

        # 默认视频源为相机
        self.ui.real_time.setChecked(True)
        self.isCamera = True

        # 摄像头ip设置
        self.user_name = "admin"
        self.user_pwd = "uex_2022"
        self.channel = 2
        self.camera_ips = [
            "192.168.254.3",  # 摄像头1
            "192.168.254.11"  # 摄像头2
        ]

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
        print("load")
        self.model = C3D_model.C3D(num_classes=self.num_classes, pretrained=False)
        print("1111")
        checkpoint = torch.load(self.model_pth_dir)
        # checkpoint = torch.load(self.model_pth_dir, map_location=lambda storage, loc: storage)
        print("model done")
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
        if self.isCamera:
            # 实时监测模式
            # 多进程
            self.thread = []
            self.camera_objects = []
            for index, camera_ip in enumerate(self.camera_ips):
                url = "rtsp://{}:{}@{}/Streaming/Channels/{}".format(self.user_name, self.user_pwd, camera_ip, self.channel)
                camera_object = Recognition(url, self.camera_list[index], self.led_list[index], self.device, self.model, self.action_name, True)
                self.camera_objects.append(camera_object)
                self.thread.append(threading.Thread(target=camera_object.queue_img_put, args=(camera_object.frame_queue,),daemon=True))
                self.thread.append(threading.Thread(target=camera_object.c3d_detect, args=(camera_object.frame_queue,),daemon=True))
                self.thread.append(threading.Thread(target=camera_object.queue_img_get, args=(camera_object.frame_queue,),daemon=True))
            for thread in self.thread:
                thread.start()
        else:
            # 文件检测模式
            print("文件检测模式")

    # Stop按钮实现
    def stop(self):
        # 摄像头爆内存
        for camera_object in self.camera_objects:
            camera_object.flag = False
            time.sleep(0.1)
        # camera初始化显示电子科技大学图标
        self.camera_list = [self.ui.camera1, self.ui.camera2]
        for camera in self.camera_list:
            camera.clear()
        pic = cv2.imread("pic.jpg")
        pic = cv2.resize(pic, (120, 90))
        pic_bgr = cv2.cvtColor(pic, cv2.COLOR_RGB2BGR)
        img = QPixmap.fromImage(QImage(pic_bgr.data, pic_bgr.shape[1], pic_bgr.shape[0], QImage.Format_RGB888))
        for camera in self.camera_list:
            camera.setPixmap(img)
        # led灯显示 -- 蓝色
        self.led_list = [self.ui.led1, self.ui.led2]
        for led in self.led_list:
            led.clear()
        blue = cv2.imread("blue.png")
        blue = cv2.resize(blue, (20, 20))
        blue_bgr = cv2.cvtColor(blue, cv2.COLOR_RGB2BGR)
        blue_img = QPixmap.fromImage(QImage(blue_bgr.data, blue_bgr.shape[1], blue_bgr.shape[0], QImage.Format_RGB888))
        for led in self.led_list:
            led.setPixmap(blue_img)
        print("done")



