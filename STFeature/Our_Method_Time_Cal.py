import os
import time
import numpy as np
import pywt
import cv2
from random import sample
import random
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import label_binarize

"""step1:对已经分段好的视频进行特征提取并保存"""
# 检测单张图片的角点特征坐标
def detect_feature(pic):
    feature_params = dict(maxCorners=500, qualityLevel=0.3, minDistance=7, blockSize=7)  # 参数设置
    mask = np.zeros_like(pic)  # 初始化和视频大小相同的图像
    mask[:] = 255  # 将mask赋值255也就是算全部图像的角点
    tracks = []  # 用于记录该图片的角点坐标
    p = cv2.goodFeaturesToTrack(pic, mask=mask, **feature_params)
    if p is not None:
        for x, y in np.float32(p).reshape(-1, 2):
            tracks.append([(x, y)])
    return tracks

# 追踪5s检测角点的轨迹坐标并保存
def detect_track(video_input_path, box_coordinate_path):
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))  # 参数设置
    video = cv2.VideoCapture(video_input_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    track_len = 5 * fps  # 每5s检测的轨迹长度
    frame_index = 0
    tracks = []  # 记录角点特征轨迹
    frame_index_list = []
    tracks_len_list = []
    # 打开坐标文件
    with open(box_coordinate_path, "r") as f:
        line1 = f.readline().strip().replace("[", "")
        line1 = line1.replace("]", "")
        line1 = line1.split(",")
        box_coordinate = list(map(int, line1))
    # 对产生的坐标信息进行保存
    while (video.isOpened()):
        ret, frame = video.read()  # 读取视频帧
        if ret == True:
            # 根据box大小剪裁视频
            frame = frame[box_coordinate[1]:box_coordinate[3], box_coordinate[0]:box_coordinate[2]]
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转化为灰度虚图像
            vis = frame.copy()

            # 储存每一帧tracks的特征角点数目
            frame_index_list.append(frame_index)
            tracks_len_list.append(len(tracks))

            # 检测到角点后进行光流跟踪
            if len(tracks) > 0:
                img0, img1 = pre_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None,
                                                       **lk_params)  # 前一帧的角点和当前帧的图像作为输入来得到角点在当前帧的位置
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)  # 当前帧跟踪到的角点和前一帧进行比较
                dis = abs(p0 - p0r).reshape(-1, 2).max(-1)
                good = dis < 1  # 判断d内的值是否小于1，大于1跟踪被认为是错误的跟踪点
                new_tracks = []
                # 将跟踪正确的点列入成功跟踪点
                for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > track_len + 1:
                        print("too long")
                        del tr[0]
                    new_tracks.append(tr)  # 得到新的轨迹
                    # cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)  # 给角点坐标画圆
                tracks = new_tracks
                # cv2.polylines(vis, [np.int32(tr) for tr in tracks], False, (0, 0, 255))  # 画变化轨迹

            # 对首帧图像进行角点检测
            if frame_index == 0:
                tracks = detect_feature(frame_gray)
            frame_index += 1
            pre_gray = frame_gray
            # cv2.imshow('lk_track', vis)

        else:
            break
        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break

    tracks = [[list(xy) for xy in tr] for tr in tracks]
    return tracks, frame_index_list, tracks_len_list


"""step2:三类特征提取：坐标信息统计及每帧的特征角点数目统计,并进行小波变换"""
# 坐标信息统计：输入轨迹坐标文件，返回每个clip文件的轨迹移动距离
def dis_calculate(tracks):
    feature_num = len(tracks)
    tracks_sum = 0
    if feature_num == 0:
        return 0, 0
    for track in tracks:
        xy_num = len(track)
        track_sum = 0
        for i in range(xy_num-1):
            # 求相邻帧间角点坐标的距离，并累加
            dis = (track[i+1][0]-track[i][0])**2+(track[i+1][1]-track[i][1])**2
            track_sum += dis
        tracks_sum += track_sum
    average_dis = tracks_sum/feature_num
    return feature_num,average_dis

# 特征角点数目统计：输入特征角点数目文件，返回每个clip的帧数列表以及每帧的特征角点数目
# 小波变换：对clip视频进行小波变换
# 输入：每个clip的帧数列表以及每帧的特征角点数目；输出：每个clip的小波变换（25维向量）
def wtAnalysis(frame_index_list,feature_num_list):
    fps = int(len(frame_index_list)/5)
    # 进行小波变化
    cwtmatr, freqs = pywt.cwt(feature_num_list, np.arange(1, fps+1),'gaus8')  # cwtmart代表小波变换后每个时刻每个尺度下的幅度值
    # 对cwtmatr进行计算
    cwtmatr_array = np.array(cwtmatr)
    cwtmatr_sum = np.max(cwtmatr_array,axis=1)
    cwtmatr_sum = cwtmatr_sum.tolist()
    return cwtmatr_sum


"""step3:选取样本，划分训练集和验证集，采用k折交叉验证的方式"""
# 利用svm,knn分类器进行分类实验
def time_cal_svm_knn(clf_num, method):
    videos_path = "./裁剪后视频段"
    labels_path = "./标签信息"
    boxes_dir = "./检测框信息"
    mp4_filename_sample = [[], [], [], []]
    txt_filename_sample = [[], [], [], []]
    for videoname in sorted(os.listdir(videos_path)):
        label_path = os.path.join(labels_path, videoname + ".txt")
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
            clipname_txt = "clip" + str(i + 1) + ".txt"
            mp4_tmp_path = videoname + "/" + clipname_mp4
            txt_tmp_path = videoname + "/" + clipname_txt
            mp4_filename_sample[label[i] - 1].append(mp4_tmp_path)
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
        mp4_filename_sample_array = sample(mp4_filename_sample[0], 361) + mp4_filename_sample[1] + mp4_filename_sample[
            3]
        txt_filename_sample_array = sample(txt_filename_sample[0], 361) + txt_filename_sample[1] + txt_filename_sample[
            3]
    elif clf_num == 3:
        # 三分类
        label_array = [0] * 159 + [1] * 159 + [2] * 159
        mp4_filename_sample_array = sample(mp4_filename_sample[0], 159) + sample(mp4_filename_sample[1], 159) + \
                                    mp4_filename_sample[3]
        txt_filename_sample_array = sample(txt_filename_sample[0], 159) + sample(txt_filename_sample[1], 159) + \
                                    txt_filename_sample[3]
    elif clf_num == 4:
        # 四分类
        label_array = [0] * 104 + [1] * 104 + [2] * 104 + [3] * 104
        mp4_filename_sample_array = sample(mp4_filename_sample[0], 104) + sample(mp4_filename_sample[1], 104) + \
                                    mp4_filename_sample[2] + sample(mp4_filename_sample[3], 104)
        txt_filename_sample_array = sample(txt_filename_sample[0], 104) + sample(txt_filename_sample[1], 104) + \
                                    txt_filename_sample[2] + sample(txt_filename_sample[3], 104)

    # 分层k折交叉验证
    skf = StratifiedKFold(n_splits=10)
    # 储存每一轮的结果
    time_list = []
    train_score_list = []
    test_score_list = []
    precision_score_list = []
    recall_score_list = []
    f1_score_list = []
    #     feature_sample_array = preprocessing.scale(feature_sample_array)   # processing预处理
    for train, test in skf.split(mp4_filename_sample_array, label_array):
        # 拿到训练集
        train_data = []
        train_label = np.array(label_array)[train]
        for index in train.tolist():
            clip_path = mp4_filename_sample_array[index]
            feature_path = os.path.join("./三类特征统计", txt_filename_sample_array[index])
            feature = np.loadtxt(feature_path, delimiter=',')  # 读取feature文件
            train_data.append(feature.tolist())

        # 1.构建测试集时间计算
        current_fold_start_time_prepare_data = time.time()
        # 拿到当前折下的测试集
        test_data = []
        test_label = np.array(label_array)[test]
        for index in test.tolist():
            clip_path = os.path.join(videos_path, mp4_filename_sample_array[index])  # clip视频路径
            box_path = os.path.join(boxes_dir, mp4_filename_sample_array[index].split("/")[0] + ".txt")  # box路径
            tracks, frame_index_list, feature_num_list = detect_track(clip_path, box_path)  # 检测视频
            feature_num, average_dis = dis_calculate(tracks)  # 计算前两类特征
            cwtmatr = wtAnalysis(frame_index_list, feature_num_list)  # 计算小波特征
            feature = [feature_num, average_dis] + cwtmatr
            test_data.append(feature)
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

# dt分类器计算
def time_cal_dt(clf_num):
    videos_path = "./裁剪后视频段"
    labels_path = "./标签信息"
    boxes_dir = "./检测框信息"
    mp4_filename_sample = [[], [], [], []]
    txt_filename_sample = [[], [], [], []]
    for videoname in sorted(os.listdir(videos_path)):
        label_path = os.path.join(labels_path, videoname + ".txt")
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
            clipname_txt = "clip" + str(i + 1) + ".txt"
            mp4_tmp_path = videoname + "/" + clipname_mp4
            txt_tmp_path = videoname + "/" + clipname_txt
            mp4_filename_sample[label[i] - 1].append(mp4_tmp_path)
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
                                    mp4_filename_sample[3]
        txt_filename_sample_array = sample(txt_filename_sample[0], 361) + txt_filename_sample[1] + \
                                    txt_filename_sample[3]
    elif clf_num == 3:
        # 三分类
        label_array = [0] * 159 + [1] * 159 + [2] * 159
        mp4_filename_sample_array = sample(mp4_filename_sample[0], 159) + sample(mp4_filename_sample[1], 159) + \
                                    mp4_filename_sample[3]
        txt_filename_sample_array = sample(txt_filename_sample[0], 159) + sample(txt_filename_sample[1], 159) + \
                                    txt_filename_sample[3]
    elif clf_num == 4:
        # 四分类
        label_array = [0] * 104 + [1] * 104 + [2] * 104 + [3] * 104
        mp4_filename_sample_array = sample(mp4_filename_sample[0], 104) + sample(mp4_filename_sample[1], 104) + \
                                    mp4_filename_sample[2] + sample(mp4_filename_sample[3], 104)
        txt_filename_sample_array = sample(txt_filename_sample[0], 104) + sample(txt_filename_sample[1], 104) + \
                                    txt_filename_sample[2] + sample(txt_filename_sample[3], 104)

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
            feature_path = os.path.join("./三类特征统计", txt_filename_sample_array[index])
            feature = np.loadtxt(feature_path, delimiter=',')  # 读取feature文件
            train_data.append(feature.tolist())

        # 1.构建测试集时间计算
        current_fold_start_time_prepare_data = time.time()
        # 拿到当前折下的测试集
        test_data = []
        test_label = np.array(label_array)[test]
        for index in test.tolist():
            clip_path = os.path.join(videos_path, mp4_filename_sample_array[index])  # clip视频路径
            box_path = os.path.join(boxes_dir, mp4_filename_sample_array[index].split("/")[0] + ".txt")  # box路径
            tracks, frame_index_list, feature_num_list = detect_track(clip_path, box_path)  # 检测视频
            feature_num, average_dis = dis_calculate(tracks)  # 计算前两类特征
            cwtmatr = wtAnalysis(frame_index_list, feature_num_list)  # 计算小波特征
            feature = [feature_num, average_dis] + cwtmatr
            test_data.append(feature)
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

    print("Clf_num：", clf_num)
    print("训练集准确率：", np.array(train_score_list).mean())
    print("测试集准确率：", np.array(test_score_list).mean())
    print("精确率：", np.array(precision_score_list).mean())
    print("召回率：", np.array(recall_score_list).mean())
    print("F1得分：", np.array(f1_score_list).mean())
    print("Time:", np.array(time_list).mean())


"""step3:选取样本，划分训练集和验证集，采用k折交叉验证的方式"""
"""利用mlp、rvm、random forest分类器进行分类实验"""
"""补充实验"""
def time_cal_supplement(clf_num, method):
    videos_path = "./裁剪后视频段"
    labels_path = "./标签信息"
    boxes_dir = "./检测框信息"
    mp4_filename_sample = [[], [], [], []]
    txt_filename_sample = [[], [], [], []]
    for videoname in sorted(os.listdir(videos_path)):
        label_path = os.path.join(labels_path, videoname + ".txt")
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
            clipname_txt = "clip" + str(i + 1) + ".txt"
            mp4_tmp_path = videoname + "/" + clipname_mp4
            txt_tmp_path = videoname + "/" + clipname_txt
            mp4_filename_sample[label[i] - 1].append(mp4_tmp_path)
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
        mp4_filename_sample_array = sample(mp4_filename_sample[0], 361) + mp4_filename_sample[1] + mp4_filename_sample[
            3]
        txt_filename_sample_array = sample(txt_filename_sample[0], 361) + txt_filename_sample[1] + txt_filename_sample[
            3]
    elif clf_num == 3:
        # 三分类
        label_array = [0] * 159 + [1] * 159 + [2] * 159
        mp4_filename_sample_array = sample(mp4_filename_sample[0], 159) + sample(mp4_filename_sample[1], 159) + \
                                    mp4_filename_sample[3]
        txt_filename_sample_array = sample(txt_filename_sample[0], 159) + sample(txt_filename_sample[1], 159) + \
                                    txt_filename_sample[3]
    elif clf_num == 4:
        # 四分类
        label_array = [0] * 104 + [1] * 104 + [2] * 104 + [3] * 104
        mp4_filename_sample_array = sample(mp4_filename_sample[0], 104) + sample(mp4_filename_sample[1], 104) + \
                                    mp4_filename_sample[2] + sample(mp4_filename_sample[3], 104)
        txt_filename_sample_array = sample(txt_filename_sample[0], 104) + sample(txt_filename_sample[1], 104) + \
                                    txt_filename_sample[2] + sample(txt_filename_sample[3], 104)

    # 分层k折交叉验证
    skf = StratifiedKFold(n_splits=10)
    # 储存每一轮的结果
    time_list = []
    train_score_list = []
    test_score_list = []
    precision_score_list = []
    recall_score_list = []
    f1_score_list = []
    #     feature_sample_array = preprocessing.scale(feature_sample_array)   # processing预处理
    for train, test in skf.split(mp4_filename_sample_array, label_array):
        # 拿到训练集
        train_data = []
        train_label = np.array(label_array)[train]
        for index in train.tolist():
            clip_path = mp4_filename_sample_array[index]
            feature_path = os.path.join("./三类特征统计", txt_filename_sample_array[index])
            feature = np.loadtxt(feature_path, delimiter=',')  # 读取feature文件
            train_data.append(feature.tolist())

        # 1.构建测试集时间计算
        current_fold_start_time_prepare_data = time.time()
        # 拿到当前折下的测试集
        test_data = []
        test_label = np.array(label_array)[test]
        for index in test.tolist():
            clip_path = os.path.join(videos_path, mp4_filename_sample_array[index])  # clip视频路径
            box_path = os.path.join(boxes_dir, mp4_filename_sample_array[index].split("/")[0] + ".txt")  # box路径
            tracks, frame_index_list, feature_num_list = detect_track(clip_path, box_path)  # 检测视频
            feature_num, average_dis = dis_calculate(tracks)  # 计算前两类特征
            cwtmatr = wtAnalysis(frame_index_list, feature_num_list)  # 计算小波特征
            feature = [feature_num, average_dis] + cwtmatr
            test_data.append(feature)
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
    cls_num = [2, 3, 4]
    methods = ["MLP", "RandomForest"]
    for num in cls_num:
        for method in methods:
            time_cal_supplement(num, method)
    #
    # cls_num = [2, 3, 4]
    # for num in cls_num:
    #     time_cal_dt(num)
