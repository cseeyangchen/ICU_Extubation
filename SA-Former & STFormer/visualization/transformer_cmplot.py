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
# root_dir = '/home/uestc/cy/UEX_Transformer/video_data/uex_class4_slowfast_feature'
root_dir = '/home/uestc/cy/UEX_Transformer/video_data/uex_class4_slowfast_feature'
for file in os.listdir(root_dir):
    file_path = os.path.join(root_dir, file)
    for name in os.listdir(file_path):
        filedata_names.append(os.path.join(file_path, name))
        filelabel_names.append(action_names_dict[file])
label_array = np.array(filelabel_names)

state_dict_path_list = ["/home/uestc/cy/UEX_Transformer/model/uex_class4/token_size/type4-16×8×8×8-fold1.pt",
                        "/home/uestc/cy/UEX_Transformer/model/uex_class4/token_size/type4-16×8×8×8-fold2.pt",
                        "/home/uestc/cy/UEX_Transformer/model/uex_class4/token_size/type4-16×8×8×8-fold3.pt",
                        "/home/uestc/cy/UEX_Transformer/model/uex_class4/token_size/type4-16×8×8×8-fold4.pt"]

k_index = 0
skf = StratifiedKFold(n_splits=4)
for train, test in skf.split(filedata_names, label_array):
    train_name = np.array(filedata_names)[train]
    train_label = np.array(label_array)[train]
    test_name = np.array(filedata_names)[test]
    test_label = np.array(label_array)[test]
    test_dataloader = DataLoader(
                VideoDataset(data_name=test_name, label_array=test_label),
                batch_size=64,
                num_workers=4
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
    result_frag = []
    label_frag = []
    for inputs, labels in tqdm(test_dataloader):
        inputs = Variable(inputs).to(device)
        labels = Variable(labels).to(device).long()
        with torch.no_grad():
            output = model(inputs)
        # 记录test过程中的结果
        result_frag.append(output.data.cpu().numpy())
        label_frag.append(labels.data.cpu().numpy())
    result = np.concatenate(result_frag)
    true_label = np.concatenate(label_frag)
    rank = result.argsort()
    predict_label = np.array([rank[i, -1] for i, l in enumerate(true_label)])
    print("Fold: {}, 准确率Top1：{}".format(k_index, sum(predict_label == true_label) / predict_label.shape[0]))

    # 4.绘制混淆矩阵
    cm = confusion_matrix(true_label, predict_label)
    # np.savetxt("cm_ntu120.csv", cm, delimiter=',')
    print(cm.shape)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig = plt.figure(figsize=(14, 12))
    plt.imshow(cm, interpolation='nearest')  # 更换配色：cmap=plt.cm.Set3
    # plt.title("MGL Confusion Matrix")
    plt.colorbar()  # 热力图渐变色条
    num_action = np.array(range(len(action_names_list)))
    plt.xticks(num_action, action_names_list, rotation=90, fontproperties='Times New Roman', fontsize=10)  # 将标签印在x轴上
    plt.yticks(num_action, action_names_list, fontproperties='Times New Roman', fontsize=10)  # 将标签印在y轴上
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')
    path = 'fold-'+str(k_index)+'-cm.png'
    plt.savefig(path, dpi=600)

    k_index +=1


