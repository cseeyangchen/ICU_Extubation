import sys
sys.path.append("..")
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from net.UEX_Transformer import Model
from dataset_kfold import VideoDataset
from collections import OrderedDict
from torch.autograd import Variable



# 1.读取数据集
action_names_dict = {"medical_staff_around_class":0, "no_movement_class":1,
                "no_self_extubation_tendency":2, "with_self_extubation_tendency":3}
action_names_list = ["Medical staff around", "No movement state",
                "No UEX tendencies", "UEX tendencies"]
filedata_names = []
filelabel_names = []
root_dir = '/home/uestc/cy/UEX_Transformer/video_data/uex_class4_slowfast_feature/with_self_extubation_tendency/video1_clip33_32.npy'
filedata_names.append(root_dir)
filelabel_names.append(3)
label_array = np.array(filelabel_names)

state_dict_path_list = ["/home/uestc/cy/UEX_Transformer/model/uex_class4/token_size/type4-16×8×8×8-fold1.pt",
                        "/home/uestc/cy/UEX_Transformer/model/uex_class4/token_size/type4-16×8×8×8-fold2.pt",
                        "/home/uestc/cy/UEX_Transformer/model/uex_class4/token_size/type4-16×8×8×8-fold3.pt",
                        "/home/uestc/cy/UEX_Transformer/model/uex_class4/token_size/type4-16×8×8×8-fold4.pt"]


k_index = 0
for i in range(4):
    test_name = np.array(filedata_names)
    test_label = np.array(label_array)
    test_dataloader = DataLoader(
                VideoDataset(data_name=test_name, label_array=test_label),
                batch_size=1,
                num_workers=0
            )

    # 2.加载模型
    model = Model(time=32, height=8, width=8, channel=256,
                              out_time=16, out_channel=8, kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0),
                              head_spatial_dim=256, head_spatial_cnt=4, scale_spatial_dim=2,
                              head_temporal_dim=512, head_temporal_cnt=4, scale_temporal_dim=2,
                              st_dim=512, head_st_dim=128, head_st_cnt=4, scale_st_dim=2,
                              num_class=4, dropout=0.1)
    weights = torch.load(state_dict_path_list[k_index])
    weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in weights.items()])
    model.load_state_dict(weights)

    # 3.进行推理，取得预测结果
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    feature_frag = []
    for inputs, labels in tqdm(test_dataloader):
        inputs = Variable(inputs).to(device)
        labels = Variable(labels).to(device).long()
        with torch.no_grad():
            feature = model.extract(inputs)
        # 记录test过程中的结果
        feature_frag.append(feature.data.cpu().numpy())
    result = np.concatenate(feature_frag)


    # 4.绘制混淆矩阵
    print(result.shape)
    k_index +=1


