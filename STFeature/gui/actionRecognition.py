import cv2
import torch
import queue
import threading
import numpy as np
from network import C3D_model
from PyQt5.QtGui import QImage, QPixmap


class Recognition:
    """利用c3d模型进行实时检测：单摄像头"""
    def __init__(self, url, camera_id, led_id, device, model, action_name, flag):
        self.url = url
        self.camera_id = camera_id
        self.led_id = led_id
        self.device = device
        self.model = model
        self.action_name = action_name
        self.frame_queue = queue.Queue(maxsize=4)
        self.clip = []
        self.flag = flag

    # 将逐帧采取到的图片放入队列中
    def queue_img_put(self, q_put):
        self.cap = cv2.VideoCapture(self.url)
        while self.flag:
            ret, frame = self.cap.read()
            q_put.put(frame) if ret else None
            q_put.get() if q_put.qsize() > 2 else None

    # 将逐帧采取到放入队列的图片读取出来
    def queue_img_get(self, q_get):
        while self.flag:
            frame = q_get.get()
            # 显示此时的录像
            frame_size = cv2.resize(frame, (120, 90))
            frame_bgr = cv2.cvtColor(frame_size, cv2.COLOR_RGB2BGR)
            img = QImage(frame_bgr.data, frame_bgr.shape[1], frame_bgr.shape[0], QImage.Format_RGB888)
            tmp_img = QPixmap.fromImage(img)
            self.camera_id.setPixmap(tmp_img)
            cv2.waitKey(1)

    def c3d_detect(self, q_put):
        while self.flag:
            # 此时q_put里面没有图片，防止多进程出错
            while q_put.qsize() == 0:
                continue
            # 此时q_put里面有图片
            origin_frame = q_put.get()
            # 1.crop操作
            frame = np.copy(origin_frame)
            frame = cv2.resize(frame, (171, 128))
            frame = frame[8:120, 30:142, :]
            frame = np.array(frame).astype(np.uint8)
            # 2.normalize操作
            frame = frame - np.array([[[90.0, 98.0, 102.0]]])
            self.clip.append(frame)
            if len(self.clip) == 16:  # 每16帧进行一次检测
                # 3.to_tensor操作
                inputs = np.array(self.clip).astype(np.float32)
                inputs = np.expand_dims(inputs, axis=0)
                inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
                inputs = torch.from_numpy(inputs)
                inputs = torch.autograd.Variable(inputs, requires_grad=False).to(self.device)
                # 开始测试样本
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model.forward(inputs)
                probs = torch.nn.Softmax(dim=1)(outputs)
                label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]
                print("probs: {}, label: {}".format(probs, label))
                # 打印输出此时预测结果
                if self.action_name[str(int(label))] == "with_self_extubation_tendency":
                    self.led_id.clear()  # led灯显示 -- 红色
                    red = cv2.imread("red.png")
                    red = cv2.resize(red, (20, 20))
                    red_bgr = cv2.cvtColor(red, cv2.COLOR_RGB2BGR)
                    red_img = QPixmap.fromImage(
                        QImage(red_bgr.data, red_bgr.shape[1], red_bgr.shape[0], QImage.Format_RGB888))
                    self.led_id.setPixmap(red_img)
                else:
                    self.led_id.clear()  # led灯显示 -- 蓝色
                    blue = cv2.imread("blue.png")
                    blue = cv2.resize(blue, (20, 20))
                    blue_bgr = cv2.cvtColor(blue, cv2.COLOR_RGB2BGR)
                    blue_img = QPixmap.fromImage(
                        QImage(blue_bgr.data, blue_bgr.shape[1], blue_bgr.shape[0], QImage.Format_RGB888))
                    self.led_id.setPixmap(blue_img)
                self.clip.pop(0)

    # def single_camera(self):
    #     threads = []
    #     threads.append(threading.Thread(target=self.queue_img_put, args=(self.frame_queue,),daemon=True))
    #     threads.append(threading.Thread(target=self.c3d_detect, args=(self.frame_queue,),daemon=True))
    #     threads.append(threading.Thread(target=self.queue_img_get, args=(self.frame_queue,),daemon=True))
    #     for thread in threads:
    #         thread.start()
    #     for thread in threads:
    #         thread.join()







