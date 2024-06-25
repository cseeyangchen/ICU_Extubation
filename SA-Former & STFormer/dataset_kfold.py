import os
from sklearn.model_selection import train_test_split
import shutil
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    def __init__(self, data_name, label_array):
        self.fnames = data_name
        self.label_array = label_array
        self.resize_height = 112
        self.resize_width = 112

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        # 加载feature文件
        path1 = self.fnames[index]
        path2 = path1.split('video_data')[0] + 'video_data/uex_class4_mask_feature' + path1.split('uex_class4_slowfast_feature')[-1]
        buffer1 = np.load(self.fnames[index])
        buffer2 = np.load(path2)
        # 处理c3d视频
        # print(self.fnames[index])
        # buffer = self.load_frames(self.fnames[index])
        # buffer = buffer.transpose((3, 0, 1, 2))
        labels = np.array(self.label_array[index])
        # return torch.from_numpy(buffer1), torch.from_numpy(labels)
        return torch.from_numpy(buffer1), torch.from_numpy(buffer2), torch.from_numpy(labels)

    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, str(img, encoding="utf-8")) for img in os.listdir(file_dir)])
        # print(frames)
        frame_count = len(frames)
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            buffer[i] = frame
        return buffer










