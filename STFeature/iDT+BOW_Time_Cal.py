import os
import numpy as np
import random
from random import sample
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import gzip
import time
import cv2
from scipy.spatial import KDTree

"""将单个gz文件解压为txt文件"""
def unzip_file(input_file, output_file):
    gfile = gzip.GzipFile(input_file)
    open(output_file,"wb").write(gfile.read())
    gfile.close()


def read(dictionary,input_feature_path,output_feature_path):
    unzip_file(input_feature_path, output_feature_path)  # 解压缩
    kdtree = KDTree(dictionary)
    bins = dictionary.shape[0]
    # 读取
    data = np.loadtxt(output_feature_path)
    # 前10列为轨迹信息，30维轨迹，96维HOG，108维HOF，96维MBHX，96维MBHY
    feature_data = data[:, 40:244]
    dis, indices = kdtree.query(feature_data)
    hist, _ = np.histogram(indices, bins=bins)
    hist = hist / len(indices)
    return hist


"""step3:计算时间"""
def time_cal_svm_knn(clf_num, method):
    bow_feature_path = "./idt特征bow"
    idt_feature_zip = "./idt_feature"
    unzip_path = "./idt特征"
    labels_path = "./标签信息"
    videos_path = "./裁剪后视频段"
    test_save_dir = "./test_filename"
    mp4_filename_sample = [[], [], [], []]
    npy_filename_sample = [[], [], [], []]
    zip_filename_sample = [[], [], [], []]
    txt_filename_sample = [[], [], [], []]
    for videoname in sorted(os.listdir(videos_path)):
        label_path = os.path.join(labels_path, videoname + ".txt")  # 标签路径
        # 读取标签文件
        label = []
        with open(label_path, "r", encoding='utf-8') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = line.split('，')
                label.append(eval(line[1]))
        # 根据label读取每个clip的特征文件
        for i in range(len(label)):
            clipname_mp4 = "clip" + str(i + 1) + ".mp4"
            clipname_npy = "clip" + str(i + 1) + ".txt.npy"
            clipname_zip = "clip" + str(i + 1) + ".gz"
            clipname_txt = "clip" + str(i + 1) + ".txt"
            mp4_tmp_path = videoname + "/" + clipname_mp4
            npy_tmp_path = videoname + "/" + clipname_npy
            zip_tmp_path = videoname + "/" + clipname_zip
            txt_tmp_path = videoname + "/" + clipname_txt
            mp4_filename_sample[label[i] - 1].append(mp4_tmp_path)
            npy_filename_sample[label[i] - 1].append(npy_tmp_path)
            zip_filename_sample[label[i] - 1].append(zip_tmp_path)
            txt_filename_sample[label[i] - 1].append(txt_tmp_path)
    print("类别1文件数量：", len(mp4_filename_sample[0]))
    print("类别2文件数量：", len(mp4_filename_sample[1]))
    print("类别3文件数量：", len(mp4_filename_sample[2]))
    print("类别4文件数量：", len(mp4_filename_sample[3]))

    # 选择随机种子
    random.seed(0)
    if clf_num == 2:
        # 二分类
        label_array = [0] * 361 + [1] * 361
        mp4_filename_sample_array = sample(mp4_filename_sample[0], 361) + mp4_filename_sample[1] + \
                                    mp4_filename_sample[
                                        3]
        npy_filename_sample_array = sample(npy_filename_sample[0], 361) + npy_filename_sample[1] + \
                                    npy_filename_sample[
                                        3]
        zip_filename_sample_array = sample(zip_filename_sample[0], 361) + zip_filename_sample[1] + \
                                    zip_filename_sample[
                                        3]
        txt_filename_sample_array = sample(txt_filename_sample[0], 361) + txt_filename_sample[1] + \
                                    txt_filename_sample[
                                        3]
    elif clf_num == 3:
        # 三分类
        label_array = [0] * 159 + [1] * 159 + [2] * 159
        mp4_filename_sample_array = sample(mp4_filename_sample[0], 159) + sample(mp4_filename_sample[1], 159) + \
                                    mp4_filename_sample[3]
        npy_filename_sample_array = sample(npy_filename_sample[0], 159) + sample(npy_filename_sample[1], 159) + \
                                    npy_filename_sample[3]
        zip_filename_sample_array = sample(zip_filename_sample[0], 159) + sample(zip_filename_sample[1], 159) + \
                                    zip_filename_sample[3]
        txt_filename_sample_array = sample(txt_filename_sample[0], 159) + sample(txt_filename_sample[1], 159) + \
                                    txt_filename_sample[3]
    elif clf_num == 4:
        # 四分类
        label_array = [0] * 104 + [1] * 104 + [2] * 104 + [3] * 104
        mp4_filename_sample_array = sample(mp4_filename_sample[0], 104) + sample(mp4_filename_sample[1], 104) + \
                                    mp4_filename_sample[2] + sample(mp4_filename_sample[3], 104)
        npy_filename_sample_array = sample(npy_filename_sample[0], 104) + sample(npy_filename_sample[1], 104) + \
                                    npy_filename_sample[2] + sample(npy_filename_sample[3], 104)
        zip_filename_sample_array = sample(zip_filename_sample[0], 104) + sample(zip_filename_sample[1], 104) + \
                                    zip_filename_sample[2] + sample(zip_filename_sample[3], 104)
        txt_filename_sample_array = sample(txt_filename_sample[0], 104) + sample(txt_filename_sample[1], 104) + \
                                    txt_filename_sample[2] + sample(txt_filename_sample[3], 104)

    # 分层k折交叉验证
    skf = StratifiedKFold(n_splits=10)
    kindex=0
    # 储存每一轮的结果
    time_list = []
    train_score_list = []
    test_score_list = []
    precision_score_list = []
    recall_score_list = []
    f1_score_list = []
    for train, test in skf.split(mp4_filename_sample_array, label_array):
        # 拿到训练集
        train_data = []
        train_label = np.array(label_array)[train]
        for index in train.tolist():
            feature_path = os.path.join(bow_feature_path, npy_filename_sample_array[index])
            feature = np.load(feature_path)  # 读取feature文件
            train_data.append(feature.tolist())
        train_data = np.array(train_data)
        train_data = preprocessing.scale(train_data)  # processing预处理

        # 1.构建测试集时间计算
        # 读取字典
        dictionary = np.loadtxt("idt_bow_dict.txt")
        current_fold_start_time_prepare_data = time.time()
        # 拿到当前折下的测试集
        test_clip_path = []
        test_data = []
        test_label = np.array(label_array)[test]
        for index in test.tolist():
            test_clip_path.append(mp4_filename_sample_array[index])
            input_file_path = os.path.join(idt_feature_zip, zip_filename_sample_array[index])
            output_file_path = os.path.join(unzip_path, txt_filename_sample_array[index])
            feature = read(dictionary,input_file_path,output_file_path)
            test_data.append(feature.tolist())
        test_data = np.array(test_data)
        test_data = preprocessing.scale(test_data)  # processing预处理
        current_fold_end_time_prepare_data = time.time()
        current_fold_prepare_data = current_fold_end_time_prepare_data - current_fold_start_time_prepare_data
        store_path = os.path.join(test_save_dir,str(kindex)+"-fold.txt")
        np.savetxt(test_clip_path,store_path)

        if method == "SVM":
            # 利用svm进行多分类
            clf = svm.SVC(kernel='linear', decision_function_shape='ovr', max_iter=1e7, probability=True)
            clf.fit(train_data, train_label)
        if method == "KNN":
            clf = KNeighborsClassifier(n_neighbors=10)
            clf.fit(train_data, train_label)
        # 采用混淆矩阵计算各种评价指标
        predicted_train_label = clf.predict(train_data)
        # 2.推理预测时间
        current_fold_start_time_predict = time.time()
        predicted_test_label = clf.predict(test_data)
        current_fold_end_time_predict = time.time()
        current_fold_predict = current_fold_end_time_predict - current_fold_start_time_predict

        #         print("当前折训练集准确率：",metrics.accuracy_score(train_label,predicted_train_label))
        train_score_list.append(metrics.accuracy_score(train_label, predicted_train_label))
        #         print("当前折测试集准确率：",metrics.accuracy_score(test_label,predicted_test_label))
        test_score_list.append(metrics.accuracy_score(test_label, predicted_test_label))
        #         print("当前折精准值：",metrics.precision_score(test_label,predicted_test_label,average='weighted'))
        precision_score_list.append(metrics.precision_score(test_label, predicted_test_label, average='weighted'))
        #         print("当前折召回率：",metrics.recall_score(test_label,predicted_test_label,average='weighted'))
        recall_score_list.append(metrics.recall_score(test_label, predicted_test_label, average='weighted'))
        #         print("当前折F1：",metrics.f1_score(test_label,predicted_test_label,average='weighted'))
        f1_score_list.append(metrics.f1_score(test_label, predicted_test_label, average='weighted'))
        #         print("Done!*****************************************************")

        # 3.总时间累计
        current_fold_time = current_fold_prepare_data + current_fold_predict
        time_list.append(current_fold_time / len(test_label))

    print("Method:", method, " Clf_num：", clf_num)
    print("训练集准确率：", np.array(train_score_list).mean())
    print("测试集准确率：", np.array(test_score_list).mean())
    print("精确率：", np.array(precision_score_list).mean())
    print("召回率：", np.array(recall_score_list).mean())
    print("F1得分：", np.array(f1_score_list).mean())
    print("Time:", np.array(time_list).mean())

def time_cal_dt(clf_num):
    bow_feature_path = "./idt特征bow"
    idt_feature_zip = "./idt_feature"
    unzip_path = "./idt特征"
    labels_path = "./标签信息"
    videos_path = "./裁剪后视频段"
    test_save_dir = "./test路径"
    mp4_filename_sample = [[], [], [], []]
    npy_filename_sample = [[], [], [], []]
    zip_filename_sample = [[], [], [], []]
    txt_filename_sample = [[], [], [], []]
    for videoname in sorted(os.listdir(videos_path)):
        label_path = os.path.join(labels_path, videoname + ".txt")  # 标签路径
        # 读取标签文件
        label = []
        with open(label_path, "r", encoding='utf-8') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = line.split('，')
                label.append(eval(line[1]))
        # 根据label读取每个clip的特征文件
        for i in range(len(label)):
            clipname_mp4 = "clip" + str(i + 1) + ".mp4"
            clipname_npy = "clip" + str(i + 1) + ".txt.npy"
            clipname_zip = "clip" + str(i + 1) + ".gz"
            clipname_txt = "clip" + str(i + 1) + ".txt"
            mp4_tmp_path = videoname + "/" + clipname_mp4
            npy_tmp_path = videoname + "/" + clipname_npy
            zip_tmp_path = videoname + "/" + clipname_zip
            txt_tmp_path = videoname + "/" + clipname_txt
            mp4_filename_sample[label[i] - 1].append(mp4_tmp_path)
            npy_filename_sample[label[i] - 1].append(npy_tmp_path)
            zip_filename_sample[label[i] - 1].append(zip_tmp_path)
            txt_filename_sample[label[i] - 1].append(txt_tmp_path)
    print("类别1文件数量：", len(mp4_filename_sample[0]))
    print("类别2文件数量：", len(mp4_filename_sample[1]))
    print("类别3文件数量：", len(mp4_filename_sample[2]))
    print("类别4文件数量：", len(mp4_filename_sample[3]))

    # 选择随机种子
    random.seed(0)
    if clf_num == 2:
        # 二分类
        label_array = [0] * 361 + [1] * 361
        mp4_filename_sample_array = sample(mp4_filename_sample[0], 361) + mp4_filename_sample[1] + \
                                    mp4_filename_sample[
                                        3]
        npy_filename_sample_array = sample(npy_filename_sample[0], 361) + npy_filename_sample[1] + \
                                    npy_filename_sample[
                                        3]
        zip_filename_sample_array = sample(zip_filename_sample[0], 361) + zip_filename_sample[1] + \
                                    zip_filename_sample[
                                        3]
        txt_filename_sample_array = sample(txt_filename_sample[0], 361) + txt_filename_sample[1] + \
                                    txt_filename_sample[
                                        3]
    elif clf_num == 3:
        # 三分类
        label_array = [0] * 159 + [1] * 159 + [2] * 159
        mp4_filename_sample_array = sample(mp4_filename_sample[0], 159) + sample(mp4_filename_sample[1], 159) + \
                                    mp4_filename_sample[3]
        npy_filename_sample_array = sample(npy_filename_sample[0], 159) + sample(npy_filename_sample[1], 159) + \
                                    npy_filename_sample[3]
        zip_filename_sample_array = sample(zip_filename_sample[0], 159) + sample(zip_filename_sample[1], 159) + \
                                    zip_filename_sample[3]
        txt_filename_sample_array = sample(txt_filename_sample[0], 159) + sample(txt_filename_sample[1], 159) + \
                                    txt_filename_sample[3]
    elif clf_num == 4:
        # 四分类
        label_array = [0] * 104 + [1] * 104 + [2] * 104 + [3] * 104
        mp4_filename_sample_array = sample(mp4_filename_sample[0], 104) + sample(mp4_filename_sample[1], 104) + \
                                    mp4_filename_sample[2] + sample(mp4_filename_sample[3], 104)
        npy_filename_sample_array = sample(npy_filename_sample[0], 104) + sample(npy_filename_sample[1], 104) + \
                                    npy_filename_sample[2] + sample(npy_filename_sample[3], 104)
        zip_filename_sample_array = sample(zip_filename_sample[0], 104) + sample(zip_filename_sample[1], 104) + \
                                    zip_filename_sample[2] + sample(zip_filename_sample[3], 104)
        txt_filename_sample_array = sample(txt_filename_sample[0], 104) + sample(txt_filename_sample[1], 104) + \
                                    txt_filename_sample[2] + sample(txt_filename_sample[3], 104)

    # 分层k折交叉验证
    skf = StratifiedKFold(n_splits=10)
    kindex=0
    # 储存每一轮的结果
    time_list = []
    train_score_list = []
    test_score_list = []
    precision_score_list = []
    recall_score_list = []
    f1_score_list = []
    for train, test in skf.split(mp4_filename_sample_array, label_array):
        # 拿到训练集
        train_data = []
        train_label = np.array(label_array)[train]
        for index in train.tolist():
            feature_path = os.path.join(bow_feature_path, npy_filename_sample_array[index])
            feature = np.load(feature_path)  # 读取feature文件
            train_data.append(feature.tolist())
        train_data = np.array(train_data)
        train_data = preprocessing.scale(train_data)  # processing预处理

        # 1.构建测试集时间计算
        # 读取字典
        dictionary = np.loadtxt("idt_bow_dict.txt")
        current_fold_start_time_prepare_data = time.time()
        # 拿到当前折下的测试集
        test_clip_path = []
        test_data = []
        test_label = np.array(label_array)[test]
        for index in test.tolist():
            clip_path = os.path.join(videos_path, mp4_filename_sample_array[index])  # clip视频路径
            test_clip_path.append(clip_path)
            input_file_path = os.path.join(idt_feature_zip, zip_filename_sample_array[index])
            output_file_path = os.path.join(unzip_path, txt_filename_sample_array[index])
            feature = read(dictionary,input_file_path,output_file_path)
            test_data.append(feature.tolist())
        test_data = np.array(test_data)
        test_data = preprocessing.scale(test_data)  # processing预处理
        current_fold_end_time_prepare_data = time.time()
        current_fold_prepare_data = current_fold_end_time_prepare_data - current_fold_start_time_prepare_data
        store_path = os.path.join(test_save_dir,str(kindex)+"-fold")
        np.save(test_clip_path,store_path)

        clf = DecisionTreeClassifier(random_state=0, max_depth=3)
        clf.fit(train_data, train_label)

        # 采用混淆矩阵计算各种评价指标
        predicted_train_label = clf.predict(train_data)
        # 2.推理预测时间
        current_fold_start_time_predict = time.time()
        predicted_test_label = clf.predict(test_data)
        current_fold_end_time_predict = time.time()
        current_fold_predict = current_fold_end_time_predict - current_fold_start_time_predict

        #         print("当前折训练集准确率：",metrics.accuracy_score(train_label,predicted_train_label))
        train_score_list.append(metrics.accuracy_score(train_label, predicted_train_label))
        #         print("当前折测试集准确率：",metrics.accuracy_score(test_label,predicted_test_label))
        test_score_list.append(metrics.accuracy_score(test_label, predicted_test_label))
        #         print("当前折精准值：",metrics.precision_score(test_label,predicted_test_label,average='weighted'))
        precision_score_list.append(metrics.precision_score(test_label, predicted_test_label, average='weighted'))
        #         print("当前折召回率：",metrics.recall_score(test_label,predicted_test_label,average='weighted'))
        recall_score_list.append(metrics.recall_score(test_label, predicted_test_label, average='weighted'))
        #         print("当前折F1：",metrics.f1_score(test_label,predicted_test_label,average='weighted'))
        f1_score_list.append(metrics.f1_score(test_label, predicted_test_label, average='weighted'))
        #         print("Done!*****************************************************")

        # 3.总时间累计
        current_fold_time = current_fold_prepare_data + current_fold_predict
        time_list.append(current_fold_time / len(test_label))

    print("Clf_num：", clf_num)
    print("训练集准确率：", np.array(train_score_list).mean())
    print("测试集准确率：", np.array(test_score_list).mean())
    print("精确率：", np.array(precision_score_list).mean())
    print("召回率：", np.array(recall_score_list).mean())
    print("F1得分：", np.array(f1_score_list).mean())
    print("Time:", np.array(time_list).mean())

def time_cal_supplement(clf_num, method):
    bow_feature_path = "./idt特征bow"
    idt_feature_zip = "./idt_feature"
    unzip_path = "./idt特征"
    labels_path = "./标签信息"
    videos_path = "./裁剪后视频段"
    test_save_dir = "./test_filename"
    mp4_filename_sample = [[], [], [], []]
    npy_filename_sample = [[], [], [], []]
    zip_filename_sample = [[], [], [], []]
    txt_filename_sample = [[], [], [], []]
    for videoname in sorted(os.listdir(videos_path)):
        label_path = os.path.join(labels_path, videoname + ".txt")  # 标签路径
        # 读取标签文件
        label = []
        with open(label_path, "r", encoding='utf-8') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = line.split('，')
                label.append(eval(line[1]))
        # 根据label读取每个clip的特征文件
        for i in range(len(label)):
            clipname_mp4 = "clip" + str(i + 1) + ".mp4"
            clipname_npy = "clip" + str(i + 1) + ".txt.npy"
            clipname_zip = "clip" + str(i + 1) + ".gz"
            clipname_txt = "clip" + str(i + 1) + ".txt"
            mp4_tmp_path = videoname + "/" + clipname_mp4
            npy_tmp_path = videoname + "/" + clipname_npy
            zip_tmp_path = videoname + "/" + clipname_zip
            txt_tmp_path = videoname + "/" + clipname_txt
            mp4_filename_sample[label[i] - 1].append(mp4_tmp_path)
            npy_filename_sample[label[i] - 1].append(npy_tmp_path)
            zip_filename_sample[label[i] - 1].append(zip_tmp_path)
            txt_filename_sample[label[i] - 1].append(txt_tmp_path)
    print("类别1文件数量：", len(mp4_filename_sample[0]))
    print("类别2文件数量：", len(mp4_filename_sample[1]))
    print("类别3文件数量：", len(mp4_filename_sample[2]))
    print("类别4文件数量：", len(mp4_filename_sample[3]))

    # 选择随机种子
    random.seed(0)
    if clf_num == 2:
        # 二分类
        label_array = [0] * 361 + [1] * 361
        mp4_filename_sample_array = sample(mp4_filename_sample[0], 361) + mp4_filename_sample[1] + \
                                    mp4_filename_sample[
                                        3]
        npy_filename_sample_array = sample(npy_filename_sample[0], 361) + npy_filename_sample[1] + \
                                    npy_filename_sample[
                                        3]
        zip_filename_sample_array = sample(zip_filename_sample[0], 361) + zip_filename_sample[1] + \
                                    zip_filename_sample[
                                        3]
        txt_filename_sample_array = sample(txt_filename_sample[0], 361) + txt_filename_sample[1] + \
                                    txt_filename_sample[
                                        3]
    elif clf_num == 3:
        # 三分类
        label_array = [0] * 159 + [1] * 159 + [2] * 159
        mp4_filename_sample_array = sample(mp4_filename_sample[0], 159) + sample(mp4_filename_sample[1], 159) + \
                                    mp4_filename_sample[3]
        npy_filename_sample_array = sample(npy_filename_sample[0], 159) + sample(npy_filename_sample[1], 159) + \
                                    npy_filename_sample[3]
        zip_filename_sample_array = sample(zip_filename_sample[0], 159) + sample(zip_filename_sample[1], 159) + \
                                    zip_filename_sample[3]
        txt_filename_sample_array = sample(txt_filename_sample[0], 159) + sample(txt_filename_sample[1], 159) + \
                                    txt_filename_sample[3]
    elif clf_num == 4:
        # 四分类
        label_array = [0] * 104 + [1] * 104 + [2] * 104 + [3] * 104
        mp4_filename_sample_array = sample(mp4_filename_sample[0], 104) + sample(mp4_filename_sample[1], 104) + \
                                    mp4_filename_sample[2] + sample(mp4_filename_sample[3], 104)
        npy_filename_sample_array = sample(npy_filename_sample[0], 104) + sample(npy_filename_sample[1], 104) + \
                                    npy_filename_sample[2] + sample(npy_filename_sample[3], 104)
        zip_filename_sample_array = sample(zip_filename_sample[0], 104) + sample(zip_filename_sample[1], 104) + \
                                    zip_filename_sample[2] + sample(zip_filename_sample[3], 104)
        txt_filename_sample_array = sample(txt_filename_sample[0], 104) + sample(txt_filename_sample[1], 104) + \
                                    txt_filename_sample[2] + sample(txt_filename_sample[3], 104)

    # 分层k折交叉验证
    skf = StratifiedKFold(n_splits=10)
    kindex=0
    # 储存每一轮的结果
    time_list = []
    train_score_list = []
    test_score_list = []
    precision_score_list = []
    recall_score_list = []
    f1_score_list = []
    for train, test in skf.split(mp4_filename_sample_array, label_array):
        # 拿到训练集
        train_data = []
        train_label = np.array(label_array)[train]
        for index in train.tolist():
            feature_path = os.path.join(bow_feature_path, npy_filename_sample_array[index])
            feature = np.load(feature_path)  # 读取feature文件
            train_data.append(feature.tolist())
        train_data = np.array(train_data)
        train_data = preprocessing.scale(train_data)  # processing预处理

        # 1.构建测试集时间计算
        # 读取字典
        dictionary = np.loadtxt("idt_bow_dict.txt")
        current_fold_start_time_prepare_data = time.time()
        # 拿到当前折下的测试集
        test_clip_path = []
        test_data = []
        test_label = np.array(label_array)[test]
        for index in test.tolist():
            test_clip_path.append(mp4_filename_sample_array[index])
            input_file_path = os.path.join(idt_feature_zip, zip_filename_sample_array[index])
            output_file_path = os.path.join(unzip_path, txt_filename_sample_array[index])
            feature = read(dictionary,input_file_path,output_file_path)
            test_data.append(feature.tolist())
        test_data = np.array(test_data)
        test_data = preprocessing.scale(test_data)  # processing预处理
        current_fold_end_time_prepare_data = time.time()
        current_fold_prepare_data = current_fold_end_time_prepare_data - current_fold_start_time_prepare_data
        store_path = os.path.join(test_save_dir,str(kindex)+"-fold.txt")
        np.savetxt(test_clip_path,store_path)

        if method == "MLP":
            # 利用mlp进行多分类
            from sklearn.neural_network import MLPClassifier
            #             clf = MLPClassifier(solver='lbfgs',alpha=1e-6,hidden_layer_sizes=(64,3))
            #             clf = MLPClassifier(solver='lbfgs',alpha=1e-6,hidden_layer_sizes=(128,6))
            clf = MLPClassifier(solver='lbfgs', alpha=1e-6, hidden_layer_sizes=(256, 9))
            clf.fit(train_data, train_label)
        if method == "RVM":
            from sklearn_rvm import EMRVC
            clf = EMRVC(kernel='rbf', max_iter=10)
            clf.fit(train_data, train_label)
        if method == "RandomForest":
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(max_depth=2)
            clf.fit(train_data, train_label)
        # 采用混淆矩阵计算各种评价指标
        predicted_train_label = clf.predict(train_data)
        # 2.推理预测时间
        current_fold_start_time_predict = time.time()
        predicted_test_label = clf.predict(test_data)
        current_fold_end_time_predict = time.time()
        current_fold_predict = current_fold_end_time_predict - current_fold_start_time_predict

        #         print("当前折训练集准确率：",metrics.accuracy_score(train_label,predicted_train_label))
        train_score_list.append(metrics.accuracy_score(train_label, predicted_train_label))
        #         print("当前折测试集准确率：",metrics.accuracy_score(test_label,predicted_test_label))
        test_score_list.append(metrics.accuracy_score(test_label, predicted_test_label))
        #         print("当前折精准值：",metrics.precision_score(test_label,predicted_test_label,average='weighted'))
        precision_score_list.append(metrics.precision_score(test_label, predicted_test_label, average='weighted'))
        #         print("当前折召回率：",metrics.recall_score(test_label,predicted_test_label,average='weighted'))
        recall_score_list.append(metrics.recall_score(test_label, predicted_test_label, average='weighted'))
        #         print("当前折F1：",metrics.f1_score(test_label,predicted_test_label,average='weighted'))
        f1_score_list.append(metrics.f1_score(test_label, predicted_test_label, average='weighted'))
        #         print("Done!*****************************************************")

        # 3.总时间累计
        current_fold_time = current_fold_prepare_data + current_fold_predict
        time_list.append(current_fold_time / len(test_label))

    print("Method:", method, " Clf_num：", clf_num)
    print("训练集准确率：", np.array(train_score_list).mean())
    print("测试集准确率：", np.array(test_score_list).mean())
    print("精确率：", np.array(precision_score_list).mean())
    print("召回率：", np.array(recall_score_list).mean())
    print("F1得分：", np.array(f1_score_list).mean())
    print("Time:", np.array(time_list).mean())


if __name__ == "__main__":
    """进行2，3，4分类，并利用svm，knn为分类器做对比实验"""
    cls_num = [2, 3, 4]
    methods = ["MLP", "RandomForest"]
    for num in cls_num:
        for method in methods:
            time_cal_supplement(num, method)

    """进行2，3，4分类，并利用dt为分类器做对比实验"""
    # cls_num = [2, 3, 4]
    # for num in cls_num:
    #     time_cal_dt(num)


