import cv2
import time
import multiprocessing as mp

import torch
import numpy as np
from torch import nn
from network import C3D_model


# 将逐帧采取到的图片放入队列中
def queue_img_put(q_put, name, pwd, ip, channel=2):
    # 读取摄像头
    cap = cv2.VideoCapture("rtsp://{}:{}@{}/Streaming/Channels/{}".format(name, pwd, ip, channel))
    while True:
        ret, frame = cap.read()
        q_put.put(frame) if ret else None
        q_put.get() if q_put.qsize()>2 else None


# 将逐帧采取到放入队列的图片读取出来
def queue_img_get(q_get, window_name):
    cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
    while True:
        frame = q_get.get()
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)


# 加载c3d模型
def load_model(model_path, num_classes, device):
    c3d_model = C3D_model.C3D(num_classes=num_classes, pretrained=False)   # 暂时默认分为4类
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    c3d_model.load_state_dict(checkpoint['state_dict'])
    c3d_model.to(device)
    return c3d_model

# pytorch c3d model进行检测
def c3d_detect(q_put, device, model):
    clip = []
    while True:
        # 此时q_put里面没有图片，防止多进程出错
        while q_put.qsize()==0:
            continue
        # 此时q_put里面有图片
        origin_frame = q_put.get()
        # 开始处理
        # 1.crop操作
        frame = np.copy(origin_frame)
        frame = cv2.resize(frame, (171,128))
        frame = frame[8:120, 30:142, :]
        frame = np.array(frame).astype(np.uint8)
        # 2.normalize操作
        frame = frame-np.array([[[90.0, 98.0, 102.0]]])
        clip.append(frame)
        if len(clip) == 16:   # 每16帧进行一次检测
            # 3.to_tensor操作
            inputs = np.array(clip).astype(np.float32)
            inputs = np.expand_dims(inputs, axis=0)
            inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
            inputs = torch.from_numpy(inputs)
            inputs = torch.autograd.Variable(inputs, requires_grad=False).to(device)
            # 开始测试样本
            model.eval()
            with torch.no_grad():
                outputs = model.forward(inputs)
            probs = torch.nn.Softmax(dim=1)(outputs)
            label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]
            print("probs: {}, label: {}".format(probs, label))
            # 清空clip
            clip.pop(0)


# 多摄像头调用
def run_multi_camera():
    user_name, user_pwd = "admin", "uex_2022"
    camera_ips = [
        "192.168.254.2",  # 摄像头1
        "192.168.254.3"  # 摄像头1
    ]

    mp.set_start_method(method='spawn', force=True)
    queue_camera = [mp.Queue(maxsize=4) for _ in camera_ips]

    # 1.先加载c3d模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = "model/C3D-uex_class4.pth.tar"
    model = load_model(model_path, num_classes=4, device=device)
    print("Done!")

    # 2.多进程
    processes = []
    for index, camera_ip in enumerate(camera_ips):
        processes.append(mp.Process(target=queue_img_put, args=(queue_camera[index], user_name, user_pwd, camera_ip)))
        processes.append(mp.Process(target=c3d_detect, args=(queue_camera[index],  device, model)))
        processes.append(mp.Process(target=queue_img_get, args=(queue_camera[index], camera_ip)))

    for process in processes:
        process.daemon = True
        process.start()
    for process in processes:
        process.join()


if __name__ == "__main__":
    run_multi_camera()