import os
import cv2
import torch
import numpy as np
from torch import nn

from network import C3D_model
from mypath import Path


# 参数设置
dataset = "uex_class4"
model_name = "C3D"
test_method = "test-sample"  # 或者现存文件test-sample
# GPU设置：有GPU使用GPU,反之使用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 多分类方式
if dataset == "uex_class2":
    num_classes = 2
    action_name = {"0":"no_movement_class","1":"with_self_extubation_tendency"}
    model_pth_dir = "model/C3D-uex_class2.pth.tar"
elif dataset == "uex_class3":
    num_classes = 3
    action_name = {"0":"no_movement_class","1":"no_self_extubation_tendency",
                   "2":"with_self_extubation_tendency"}
    model_pth_dir = "model/C3D-uex_class3.pth.tar"
elif dataset == "uex_class4":
    num_classes = 4
    action_name = {"0":"medical_staff_around_class","1":"no_movement_calss",
                   "2":"no_self_extubation_tendency","3":"with_self_extubation_tendency"}
    model_pth_dir = "model/C3D-uex_class4.pth.tar"
else:
    print("We only implemented uex_class2, uex_class3, uex_class4")
    raise NotImplementedError


# 实时测试样本
model = C3D_model.C3D(num_classes=num_classes, pretrained=True)
print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
checkpoint = torch.load(model_pth_dir, map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint['state_dict'])
model.to(device)


# 实时检测视频生成每5s的视频段
clip_len = 5   # 5s视频
EXTRACT_FREQUENCY = 4     # 每4帧保存图片
resize_height = 128
resize_width = 171
crop_size = 112
crop_len = 16
# 实时读取视频
# 或通过测试用例验证
if test_method == "real-time":
    cap = cv2.VideoCapture(0)
elif test_method == "test-sample":
    file_dir = Path.db_dir(dataset)
    for filename in sorted(os.listdir(file_dir)):
        sample_path = os.path.join(file_dir, filename)
        break
    cap = cv2.VideoCapture(sample_path)
else:
    print("We only implemented real-time and test-sample")
    raise NotImplementedError
fps = cap.get(cv2.CAP_PROP_FPS)
interval = clip_len*fps
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_index = 0
pic_index = 0
clip_data = np.empty((int(interval/EXTRACT_FREQUENCY+1), resize_height, resize_width, 3), np.dtype('float32'))
while True:
    ret, frame = cap.read()
    if ret == True:
        # 每隔4帧对单张图片进行保存
        if frame_index % EXTRACT_FREQUENCY == 0:
            if (frame_height != resize_height) or (frame_width != resize_width):
                frame = cv2.resize(frame, (resize_width, resize_height))
                frame = np.array(frame).astype(np.float64)
                clip_data[pic_index] = frame
            pic_index += 1
        frame_index += 1

        # 5s视频段分界线 -- 检测视频类别
        if frame_index == interval:
            # 对clip_data进行数据操作
            # 1.crop操作
            time_index = np.random.randint(clip_data.shape[0] - crop_len)
            height_index = np.random.randint(clip_data.shape[1] - crop_size)
            width_index = np.random.randint(clip_data.shape[2] - crop_size)
            clip_data = clip_data[time_index:time_index + crop_len, height_index:height_index + crop_size,width_index:width_index + crop_size, :]
            # 2.normalize操作
            for i, data in enumerate(clip_data):
                data -= np.array([[[90.0, 98.0, 102.0]]])
                clip_data[i] = data
            # 3.to_tensor操作
            clip_data = clip_data.transpose((3, 0, 1, 2))
            clip_data = torch.from_numpy(clip_data)
            clip_data = torch.unsqueeze(clip_data, 0)

            # 利用c3d模型进行检测
            # 开始测试样本
            model.eval()
            inputs = clip_data.to(device)
            with torch.no_grad():
                outputs = model(inputs)
            probs = nn.Softmax(dim=1)(outputs)
            preds = torch.max(probs, 1)[1]
            print(probs)
            print("preds action:", action_name[str(int(preds[0]))])

            # 对index进行重置，以便下一个视频进行检测
            pic_index = 0
            frame_index = 0
            clip_data = np.empty((int(interval/EXTRACT_FREQUENCY+1), resize_height, resize_width, 3), np.dtype('float32'))
    else:
        break
cap.release()





