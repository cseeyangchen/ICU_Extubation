import os
from sklearn.model_selection import train_test_split
import shutil
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset


class Path(object):
    @staticmethod
    def dataset_dir(dataset):
        if dataset == 'uex_class2':
            root_dir = 'video_data/uex_class2_slowfast_feature'
            out_dir = 'video_data/uex_class2_slowfast_split'
            return root_dir, out_dir
        elif dataset == 'uex_class3':
            root_dir = 'video_data/uex_class3_slowfast_feature'
            out_dir = 'video_data/uex_class3_slowfast_split'
            return root_dir, out_dir
        elif dataset == 'uex_class4':
            root_dir = 'video_data/uex_class4_slowfast_feature'
            out_dir = 'video_data/uex_class4_slowfast_split'
            return root_dir, out_dir
        else:
            print('Dataset {} not available.'.format(dataset))


class VideoDataset(Dataset):
    def __init__(self, dataset='uex_class4', split='train'):
        self.root_dir, self.output_dir = Path.dataset_dir(dataset)
        self.split = split
        folder = os.path.join(self.output_dir, self.split)

        # 检查dataset路径是否正确
        if not os.path.exists(self.root_dir):
            raise RuntimeError('Dataset not found. You need to check your dataset.')

        # 检查dataset数据集是否进行过划分，没有划分的话对其进行划分
        if not os.path.exists(self.output_dir) or not os.path.exists(os.path.join(self.output_dir, 'train')):
            print('Preprocessing of {} dataset, this will take long, but it will be done only once.'.format(dataset))
            self.split_dataset()

        # 获取标签label信息
        self.fnames = []
        self.labels = []
        for label in sorted(os.listdir(folder)):
            for fname in os.listdir(os.path.join(folder, label)):
                self.fnames.append(os.path.join(folder, label, fname))
                self.labels.append(label)
        assert len(self.labels) == len(self.fnames)
        print('Number of {} videos: {:d}'.format(split, len(self.fnames)))
        # 将label名称转化为数字
        self.label2index = {label: index for index, label in enumerate(sorted(set(self.labels)))}
        self.label_array = np.array([self.label2index[label] for label in self.labels], dtype=int)

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        # 加载feature文件
        buffer = np.load(self.fnames[index])
        labels = np.array(self.label_array[index])
        return torch.from_numpy(buffer), torch.from_numpy(labels)

    def split_dataset(self):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
            os.mkdir(os.path.join(self.output_dir, 'train'))
            os.mkdir(os.path.join(self.output_dir, 'test'))

        # 划分数据集
        for file in os.listdir(self.root_dir):
            file_path = os.path.join(self.root_dir, file)
            feature_files = [name for name in os.listdir(file_path)]

            train, test = train_test_split(feature_files, test_size=0.2, random_state=42)

            train_dir = os.path.join(self.output_dir,'train',file)
            test_dir = os.path.join(self.output_dir,'test',file)

            if not os.path.exists(train_dir):
                os.mkdir(train_dir)
            if not os.path.exists(test_dir):
                os.mkdir(test_dir)

            for video_feature in train:
                shutil.copyfile(os.path.join(file_path,video_feature), os.path.join(train_dir,video_feature))
            for video_feature in test:
                shutil.copyfile(os.path.join(file_path,video_feature), os.path.join(test_dir,video_feature))
        print('Spliting dataset operation has finished.')






