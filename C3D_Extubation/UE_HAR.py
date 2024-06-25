import cv2
import numpy as np
import pywt


class UEHAR:
    def __init__(self, input_video_path, box_coordinate_path, output_feature_path):
        self.input_video_path = input_video_path
        self.box_coordinate_path = box_coordinate_path
        self.output_feature_path = output_feature_path

    def video_process(self):
        # 对单个视频进行l-k光流特征提取， 并进行三类特征保存
        feature_params = dict(maxCorners=500, qualityLevel=0.3, minDistance=7, blockSize=7)   # Shi单帧角点检测参数设置
        lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))  # l-k光流参数设置
        # 读取视频
        self.cap = cv2.VideoCapture(self.input_video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.track_len = 5*self.fps   # 以5s为间隔检测
        # 打开坐标文件
        with open(self.box_coordinate_path,"r") as f:
            line = f.readline().strip().replace("[", "")
            line = line.replace("]", "")
            line = line.split(",")
            self.box_coordinate = list(map(int, line))
        # 开始逐帧检测，并记录特征信息
        frame_index = 0
        tracks = []
        time_domain_corner_num = []
        while(self.cap.isOpened()):
            ret, frame = self.cap.read()
            if ret == True:
                # 根据检测框大小裁剪视频
                frame = frame[self.box_coordinate[1]:self.box_coordinate[3],self.box_coordinate[0]:self.box_coordinate[2]]
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                time_domain_corner_num.append(len(tracks))  # 时域角点数特征
                # 光流追踪
                if len(tracks)>0:
                    img0, img1 = pre_gray, frame_gray
                    p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
                    p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)  # 前一帧的角点和当前帧的图像作为输入来得到角点在当前帧的位置
                    p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)  # 当前帧跟踪到的角点和前一帧进行比较
                    dis = abs(p0 - p0r).reshape(-1, 2).max(-1)
                    good = dis < 1  # 判断d内的值是否小于1，大于1跟踪被认为是错误的跟踪点
                    new_tracks = []
                    # 将跟踪正确的点列入成功跟踪点
                    for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
                        if not good_flag:
                            continue
                        tr.append((x, y))
                        if len(tr) > self.track_len + 1:
                            print("too long")
                            del tr[0]
                        new_tracks.append(tr)  # 得到新的轨迹
                    tracks = new_tracks

                # 对首帧图像进行角点检测
                if frame_index == 0:
                    mask = np.zeros_like(frame_gray)
                    mask[:] = 255
                    p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
                    if p is not None:
                        for x, y in np.float32(p).reshape(-1, 2):
                            tracks.append([(x, y)])
                frame_index += 1
                pre_gray = frame_gray

            else:
                break
            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break

        # 当完成5s的检测后， 开始计算 特征角点数目特征、轨迹距离特征以及小波变换特征
        tracks = [[list(xy) for xy in tr] for tr in tracks]
        corner_num_feature, trajectory_feature = self.trajectory_dis(tracks)
        # 小波变换
        cwtmatr, freqs = pywt.cwt(time_domain_corner_num, np.arange(1, self.fps + 1), 'gaus8')  # cwtmart代表小波变换后每个时刻每个尺度下的幅度值
        # 对cwtmatr进行计算
        cwtmatr_array = np.array(cwtmatr)
        cwtmatr_sum = np.max(cwtmatr_array, axis=1)
        freq_domain_wavelet_feature = cwtmatr_sum.tolist()
        # 保存三类特征
        hand_crafted_feature = [corner_num_feature,trajectory_feature]+freq_domain_wavelet_feature
        hand_crafted_feature_array = np.array(hand_crafted_feature)
        np.savetxt(self.output_feature_path, hand_crafted_feature_array, delimiter=',', fmt='%.2f')

    def trajectory_dis(self, tracks):
        feature_num = len(tracks)
        tracks_sum = 0
        if feature_num == 0:
            return 0, 0
        for track in tracks:
            xy_num = len(track)
            track_sum = 0
            for i in range(xy_num - 1):
                # 求相邻帧间角点坐标的距离，并累加
                dis = (track[i + 1][0] - track[i][0]) ** 2 + (track[i + 1][1] - track[i][1]) ** 2
                track_sum += dis
            tracks_sum += track_sum
        average_dis = tracks_sum / feature_num
        return feature_num, average_dis












