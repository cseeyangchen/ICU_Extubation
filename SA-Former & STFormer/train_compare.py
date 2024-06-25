import torch
from torch import nn, optim
import timeit
import numpy as np
import os
from tensorboardX import SummaryWriter
from collections import OrderedDict
from dataset_kfold import VideoDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from net.UEX_Transformer import Model
from torch.autograd import Variable

from net.compare.slowfast import Model as slowfast_model
from net.compare.c3d import Model as c3d_model
from net.compare.i3d import Model as i3d_model
from net.compare.r21d import Model as r21d_model
from net.compare.s3d import Model as s3d_model
from net.compare.c3d_simple import Model as c3d_simple_model
from net.compare import c3d_simple

from sklearn.model_selection import StratifiedKFold

def processor(dataset, network, optimizer_name, lr, weight_decay, device_list,
              train_batch_size, test_batch_size, num_epoch,
              save_dir_root, save_epoch):
    if network == "i3d":
        # 读取所有样本
        filedata_names = []
        filelabel_names = []
        root_dir = '/home/uestc/cy/features_extract/output/i3d'
        for file in sorted(os.listdir(root_dir)):
            file_path = os.path.join(root_dir, file)
            for name in sorted(os.listdir(file_path)):
                if name.split('.')[0].split('_')[2] == 'rgb':
                    filedata_names.append(os.path.join(file_path, name))
                    filelabel_names.append(file)
        # 将label名称转化为数字
        label2index = {label: index for index, label in enumerate(sorted(set(filelabel_names)))}
        label_array = np.array([label2index[label] for label in filelabel_names], dtype=int)
    elif network == 'c3d_simple':
        # 读取所有样本
        filedata_names = []
        filelabel_names = []
        root_dir = '/home/mii2/project/cy/uex_mask_video/uex_class4_mask_image'
        for file in sorted(os.listdir(root_dir)):
            file_path = os.path.join(root_dir, file)
            for name in sorted(os.listdir(file_path)):
                filedata_names.append(os.path.join(file_path, name))
                filelabel_names.append(file)
        # 将label名称转化为数字
        label2index = {label: index for index, label in enumerate(sorted(set(filelabel_names)))}
        label_array = np.array([label2index[label] for label in filelabel_names], dtype=int)
    else:
        # 读取所有样本
        filedata_names = []
        filelabel_names = []
        root_dir = '/home/uestc/cy/features_extract/output/'+network
        for file in sorted(os.listdir(root_dir)):
            file_path = os.path.join(root_dir, file)
            for name in sorted(os.listdir(file_path)):
                filedata_names.append(os.path.join(file_path, name))
                filelabel_names.append(file)
        # 将label名称转化为数字
        label2index = {label: index for index, label in enumerate(sorted(set(filelabel_names)))}
        label_array = np.array([label2index[label] for label in filelabel_names], dtype=int)

    # k-fold交叉 -- 且保存每一折结果
    acc = []
    k_index = 0
    skf = StratifiedKFold(n_splits=4)
    for train, test in skf.split(filedata_names, label_array):
        train_name = np.array(filedata_names)[train]
        train_label = np.array(label_array)[train]
        test_name = np.array(filedata_names)[test]
        test_label = np.array(label_array)[test]

        # print(train_name)

        # 确定数据集
        if dataset == "uex_class2":
            num_classes = 2
        elif dataset == "uex_class3":
            num_classes = 3
        elif dataset == "uex_class4":
            num_classes = 4
        else:
            print('We only implemented uex_class2, uex_class3, uex_class4 datasets.')
            raise NotImplementedError

        # 加载模型
        if network=='slowfast':
            model = slowfast_model()
        elif network == 'i3d':
            model = i3d_model()
        elif network == 'c3d':
            model = c3d_model()
        elif network == 's3d':
            model = s3d_model()
        elif network == 'r21d':
            model = r21d_model()
        elif network == 'c3d_simple':
            model = c3d_simple_model()
            # train_params = [{'params': c3d_simple.get_1x_lr_params(model), 'lr': lr},
            #                 {'params': c3d_simple.get_10x_lr_params(model), 'lr': lr * 10}]
        else:
            print('We only implemented slowfast models.')
            raise NotImplementedError


        # 损失函数及优化器
        criterion = nn.CrossEntropyLoss()
        if optimizer_name == "SGD":
            optimizer = optim.SGD(train_params, lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20,gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs
            # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50], gamma=0.1, last_epoch=-1)
        elif optimizer_name == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            print("We only use SGD optimizer and Adam optimizer.")
            raise ValueError()

        # 计算模型参数
        print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

        # 将模型传入设备
        device = torch.device("cuda:"+str(device_list[0]) if torch.cuda.is_available() else "cpu")
        if len(device_list) > 1:
            model = nn.DataParallel(model, device_ids=device_list)
        model.to(device)

        # 将交叉熵损失函数传入设备
        criterion.to(device)

        # 加载数据
        train_dataloader = DataLoader(
            VideoDataset(data_name=train_name, label_array=train_label),
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=4*len(device_list)
            # num_workers = 0
        )
        train_size = len(train_dataloader.dataset)
        print("train_size:",train_size)
        test_dataloader = DataLoader(
            VideoDataset(data_name=test_name, label_array=test_label),
            batch_size=test_batch_size,
            num_workers=4 * len(device_list)
            # num_workers=0
        )
        test_size = len(test_dataloader.dataset)
        print("test_size:",test_size)

        # 日志保存路径
        log_dir = os.path.join(save_dir_root, 'run', 'run_' + dataset)
        writer = SummaryWriter(log_dir=log_dir)

        # 开始训练测试
        max_acc = 0
        k_index += 1
        for epoch in range(num_epoch):
            for phase in ["train", "test"]:
                # 训练阶段
                if phase == "train":
                    start_time = timeit.default_timer()
                    model.train()
                    running_loss = 0.0
                    running_corrects = 0.0
                    for inputs, labels in tqdm(train_dataloader):
                        inputs = Variable(inputs, requires_grad=True).to(torch.float32).to(device)
                        labels = Variable(labels).to(device).long()
                        output = model(inputs)
                        # loss计算
                        # print("output",output.size())
                        # print("labels",labels.size())
                        loss = criterion(output, labels)
                        # 预测结果标签
                        probs = nn.Softmax(dim=1)(output)
                        preds = torch.max(probs, 1)[1]
                        # backward
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        # 计算总体loss和准确率
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)

                    # 更改学习率
                    if optimizer_name == "SGD":
                        scheduler.step()

                    train_loss = running_loss / train_size
                    train_acc = running_corrects.double() / train_size
                    # SummaryWriter写入数据操作
                    writer.add_scalar('data/train_loss_epoch', train_loss, epoch)
                    writer.add_scalar('data/train_acc_epoch', train_acc, epoch)

                    stop_time = timeit.default_timer()
                    print("[train] Fold: {} Epoch: {}/{} lr:{} Loss:{} Top1_acc:{}".format(k_index, epoch + 1, num_epoch,
                                                                                              optimizer.state_dict()[
                                                                                                  'param_groups'][0]['lr'],
                                                                                              train_loss, train_acc))
                    print("Execution time: " + str((stop_time - start_time)))

                # # 储存模型
                # if (epoch + 1) % save_epoch == 0 and phase == "train":
                #     model_save_dir = os.path.join(save_dir_root, "model", dataset,
                #                                   "Epoch{}_".format(epoch + 1) + dataset + ".pt")
                #     state_dict = model.state_dict()
                #     torch.save(state_dict, model_save_dir)
                #     print("Save model at {}".format(model_save_dir))

                # test阶段
                if phase == "test":
                    model.eval()
                    start_time = timeit.default_timer()
                    running_loss = 0.0
                    running_corrects = 0.0
                    for inputs, labels in tqdm(test_dataloader):
                        inputs = Variable(inputs).to(torch.float32).to(device)
                        labels = Variable(labels).to(device).long()
                        with torch.no_grad():
                            output = model(inputs)
                        # loss计算
                        loss = criterion(output, labels)
                        # 预测结果标签
                        probs = nn.Softmax(dim=1)(output)
                        preds = torch.max(probs, 1)[1]
                        # 计算总体loss和准确率
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)

                    test_loss = running_loss / test_size
                    test_acc = running_corrects.double() / test_size

                    # 储存模型
                    if test_acc > max_acc:
                        max_acc = test_acc
                        # model_save_dir = os.path.join(save_dir_root, "model", dataset,
                        #                               "Epoch{}_".format(epoch + 1) + dataset + ".pt")
                        model_save_dir = os.path.join(save_dir_root, "model", dataset,
                                                      network + "-fold"+str(k_index)+".pt")
                        state_dict = model.state_dict()
                        torch.save(state_dict, model_save_dir)
                        print("Save model at {}".format(model_save_dir))

                    # SummaryWriter写入数据操作
                    writer.add_scalar('data/test_loss_epoch', test_loss, epoch)
                    writer.add_scalar('data/test_acc_epoch', test_acc, epoch)

                    stop_time = timeit.default_timer()
                    print("[test] Fold: {} Epoch: {}/{} lr:{} Loss:{} Top1_acc:{} Top1_max_acc:{}".format(k_index, epoch + 1, num_epoch,
                                                                                  optimizer.state_dict()[
                                                                                      'param_groups'][0]['lr'],
                                                                                  test_loss, test_acc, max_acc))
                    print("Execution time: " + str((stop_time - start_time)))
                    print("average time:",str((stop_time - start_time)/test_size)+ "\n")
        writer.close()
        acc.append(max_acc.cpu().numpy())
    # 计算k-fold的准确率以及标准差
    print("type:", network)
    print("k-fold acc:", acc)
    print("average acc:", np.mean(np.array(acc)))
    acc_std = np.std(np.array(acc))
    print("k-fold acc std: ", acc_std)
    # 计算模型参数
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))


if __name__ == "__main__":
    dataset = 'uex_class4'
    network = 'c3d_simple'
    optimizer_name = 'Adam'
    lr = 0.0001    # adam - 0.001, sgd- 1e-3
    weight_decay = 0.0001
    device = [0]
    train_batch_size = 16
    test_batch_size = 16
    num_epoch = 25
    # 储存路径 -- 当前根目录
    save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    # save_epoch表示训练每多少轮存一次模型
    save_epoch = 100
    processor(dataset=dataset, network=network, optimizer_name=optimizer_name, lr=lr, weight_decay=weight_decay, device_list=device,
              train_batch_size=train_batch_size, test_batch_size=test_batch_size, num_epoch=num_epoch,
              save_dir_root=save_dir_root, save_epoch=save_epoch)





