import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import os
import time
import numpy as np
from scipy.spatial import KDTree
import pickle
from random import sample
import random
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

"""step1:对视频进行处理，得到每5s的视频段sift特征"""
# 对图片进行SIFT特征提取
def compute_sift(image):
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    # sift特征提取
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=30)
    kp, des = sift.detectAndCompute(gray, None)  # keypoints, descriptors
    return des

# 处理视频
def detect_sift(video_input_path,box_coordinate_path):
    video = cv2.VideoCapture(video_input_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_index = 0
    # 对产生的坐标信息进行保存
    feature_clip = []
    # 打开坐标文件
    with open(box_coordinate_path,"r") as f:
            line1 = f.readline().strip().replace("[","")
            line1 = line1.replace("]","")
            line1 = line1.split(",")
            box_coordinate = list(map(int,line1))
    while(video.isOpened()):
        ret, frame = video.read()   # 读取视频帧
        if ret == True:
            # 根据box大小剪裁视频
            frame = frame[box_coordinate[1]:box_coordinate[3],box_coordinate[0]:box_coordinate[2]]
            des = compute_sift(frame)   # sift特征计算
            if des is not None:
                feature_clip.append(des)
            # 保存每段clip的全部帧的sift特征
            frame_index += 1
        else:
            break
        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break
    feature_clip_array = np.array(feature_clip)
    return feature_clip_array

"""step2：构建字典，构建向量"""
# 统计所有视频的sift向量，构建并训练bow词袋模型
def create_dictionary(features):
    dictionary_size=20
    bow = cv2.BOWKMeansTrainer(dictionary_size)
    # 将所有视频的所有clip添加到bow里
    for clip_idx, clip in enumerate(features):
        for j, frame in enumerate(clip):
            bow.add(frame)
    # 聚类
    dictionary = bow.cluster()
    return dictionary

# 重构词向量
def create_bow(dictionary, clip_features, method):
    kdtree = KDTree(dictionary)
    bins = dictionary.shape[0]
    clip_list = []
    for j, frame in enumerate(clip_features):
        dis, indices = kdtree.query(frame)
        hist, _ = np.histogram(indices, bins=bins)
        hist = hist / len(indices)
        clip_list.append(hist)
    clip_list = np.array(clip_list)

    if method == "mean":
        clip_list = np.mean(clip_list, 0)
    elif method == "max":
        clip_list = np.max(clip_list, 0)
    else:
        raise Exception('Method not found')

    return clip_list

"""step3:计算时间"""
def time_cal_svm_knn(clf_num, method):
    bow_feature_path = "./sift统计"
    labels_path = "./标签信息"
    videos_path = "./裁剪后视频段"
    boxes_dir = "./检测框信息"
    mp4_filename_sample = [[], [], [], []]
    npy_filename_sample = [[], [], [], []]
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
            clipname_npy = "clip" + str(i + 1) + ".npy"
            mp4_tmp_path = videoname + "/" + clipname_mp4
            npy_tmp_path = videoname + "/" + clipname_npy
            mp4_filename_sample[label[i] - 1].append(mp4_tmp_path)
            npy_filename_sample[label[i] - 1].append(npy_tmp_path)
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
    elif clf_num == 3:
        # 三分类
        label_array = [0] * 159 + [1] * 159 + [2] * 159
        mp4_filename_sample_array = sample(mp4_filename_sample[0], 159) + sample(mp4_filename_sample[1], 159) + \
                                    mp4_filename_sample[3]
        npy_filename_sample_array = sample(npy_filename_sample[0], 159) + sample(npy_filename_sample[1], 159) + \
                                    npy_filename_sample[3]
    elif clf_num == 4:
        # 四分类
        label_array = [0] * 104 + [1] * 104 + [2] * 104 + [3] * 104
        mp4_filename_sample_array = sample(mp4_filename_sample[0], 104) + sample(mp4_filename_sample[1], 104) + \
                                    mp4_filename_sample[2] + sample(mp4_filename_sample[3], 104)
        npy_filename_sample_array = sample(npy_filename_sample[0], 104) + sample(npy_filename_sample[1], 104) + \
                                    npy_filename_sample[2] + sample(npy_filename_sample[3], 104)

    # 分层k折交叉验证
    skf = StratifiedKFold(n_splits=10)
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
        dictionary = np.loadtxt("sift_bow_dict.txt")
        current_fold_start_time_prepare_data = time.time()
        # 拿到当前折下的测试集
        test_data = []
        test_label = np.array(label_array)[test]
        for index in test.tolist():
            clip_path = os.path.join(videos_path, mp4_filename_sample_array[index])  # clip视频路径
            box_path = os.path.join(boxes_dir, mp4_filename_sample_array[index].split("/")[0] + ".txt")  # box路径
            feature_clip_array = detect_sift(clip_path, box_path)
            feature = create_bow(dictionary, feature_clip_array, "mean")
            test_data.append(feature.tolist())
        test_data = np.array(test_data)
        test_data = preprocessing.scale(test_data)  # processing预处理
        current_fold_end_time_prepare_data = time.time()
        current_fold_prepare_data = current_fold_end_time_prepare_data - current_fold_start_time_prepare_data

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
    bow_feature_path = "./sift统计"
    labels_path = "./标签信息"
    videos_path = "./裁剪后视频段"
    boxes_dir = "./检测框信息"
    mp4_filename_sample = [[], [], [], []]
    npy_filename_sample = [[], [], [], []]
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
            clipname_npy = "clip" + str(i + 1) + ".npy"
            mp4_tmp_path = videoname + "/" + clipname_mp4
            npy_tmp_path = videoname + "/" + clipname_npy
            mp4_filename_sample[label[i] - 1].append(mp4_tmp_path)
            npy_filename_sample[label[i] - 1].append(npy_tmp_path)
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
    elif clf_num == 3:
        # 三分类
        label_array = [0] * 159 + [1] * 159 + [2] * 159
        mp4_filename_sample_array = sample(mp4_filename_sample[0], 159) + sample(mp4_filename_sample[1], 159) + \
                                    mp4_filename_sample[3]
        npy_filename_sample_array = sample(npy_filename_sample[0], 159) + sample(npy_filename_sample[1], 159) + \
                                    npy_filename_sample[3]
    elif clf_num == 4:
        # 四分类
        label_array = [0] * 104 + [1] * 104 + [2] * 104 + [3] * 104
        mp4_filename_sample_array = sample(mp4_filename_sample[0], 104) + sample(mp4_filename_sample[1], 104) + \
                                    mp4_filename_sample[2] + sample(mp4_filename_sample[3], 104)
        npy_filename_sample_array = sample(npy_filename_sample[0], 104) + sample(npy_filename_sample[1], 104) + \
                                    npy_filename_sample[2] + sample(npy_filename_sample[3], 104)

    # 分层k折交叉验证
    skf = StratifiedKFold(n_splits=10)
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
        dictionary = np.loadtxt("sift_bow_dict.txt")
        print("Done: dictionary 加载并创建完成")
        current_fold_start_time_prepare_data = time.time()
        # 拿到当前折下的测试集
        test_data = []
        test_label = np.array(label_array)[test]
        for index in test.tolist():
            clip_path = os.path.join(videos_path, mp4_filename_sample_array[index])  # clip视频路径
            box_path = os.path.join(boxes_dir, mp4_filename_sample_array[index].split("/")[0] + ".txt")  # box路径
            feature_clip_array = detect_sift(clip_path, box_path)
            feature = create_bow(dictionary, feature_clip_array, "mean")
            test_data.append(feature.tolist())
        test_data = np.array(test_data)
        test_data = preprocessing.scale(test_data)  # processing预处理
        current_fold_end_time_prepare_data = time.time()
        current_fold_prepare_data = current_fold_end_time_prepare_data - current_fold_start_time_prepare_data

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

    print(" Clf_num：", clf_num)
    print("训练集准确率：", np.array(train_score_list).mean())
    print("测试集准确率：", np.array(test_score_list).mean())
    print("精确率：", np.array(precision_score_list).mean())
    print("召回率：", np.array(recall_score_list).mean())
    print("F1得分：", np.array(f1_score_list).mean())
    print("Time:", np.array(time_list).mean())

def time_cal_supplement(clf_num, method):
    bow_feature_path = "./sift统计"
    labels_path = "./标签信息"
    videos_path = "./裁剪后视频段"
    boxes_dir = "./检测框信息"
    mp4_filename_sample = [[], [], [], []]
    npy_filename_sample = [[], [], [], []]
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
            clipname_npy = "clip" + str(i + 1) + ".npy"
            mp4_tmp_path = videoname + "/" + clipname_mp4
            npy_tmp_path = videoname + "/" + clipname_npy
            mp4_filename_sample[label[i] - 1].append(mp4_tmp_path)
            npy_filename_sample[label[i] - 1].append(npy_tmp_path)
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
    elif clf_num == 3:
        # 三分类
        label_array = [0] * 159 + [1] * 159 + [2] * 159
        mp4_filename_sample_array = sample(mp4_filename_sample[0], 159) + sample(mp4_filename_sample[1], 159) + \
                                    mp4_filename_sample[3]
        npy_filename_sample_array = sample(npy_filename_sample[0], 159) + sample(npy_filename_sample[1], 159) + \
                                    npy_filename_sample[3]
    elif clf_num == 4:
        # 四分类
        label_array = [0] * 104 + [1] * 104 + [2] * 104 + [3] * 104
        mp4_filename_sample_array = sample(mp4_filename_sample[0], 104) + sample(mp4_filename_sample[1], 104) + \
                                    mp4_filename_sample[2] + sample(mp4_filename_sample[3], 104)
        npy_filename_sample_array = sample(npy_filename_sample[0], 104) + sample(npy_filename_sample[1], 104) + \
                                    npy_filename_sample[2] + sample(npy_filename_sample[3], 104)

    # 分层k折交叉验证
    skf = StratifiedKFold(n_splits=10)
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
        dictionary = np.loadtxt("sift_bow_dict.txt")
        current_fold_start_time_prepare_data = time.time()
        # 拿到当前折下的测试集
        test_data = []
        test_label = np.array(label_array)[test]
        for index in test.tolist():
            clip_path = os.path.join(videos_path, mp4_filename_sample_array[index])  # clip视频路径
            box_path = os.path.join(boxes_dir, mp4_filename_sample_array[index].split("/")[0] + ".txt")  # box路径
            feature_clip_array = detect_sift(clip_path, box_path)
            feature = create_bow(dictionary, feature_clip_array, "mean")
            test_data.append(feature.tolist())
        test_data = np.array(test_data)
        test_data = preprocessing.scale(test_data)  # processing预处理
        current_fold_end_time_prepare_data = time.time()
        current_fold_prepare_data = current_fold_end_time_prepare_data - current_fold_start_time_prepare_data

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





