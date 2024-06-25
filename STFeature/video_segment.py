import cv2
import os


def video_to_clip(video_input_path, clip_output_path, interval=5):
    """
    将长时段的视频裁剪为若干段固定值的视频段
    :param video_input_path: 输入视频路径
    :param clip_output_path: 输出视频文件夹路径
    :param interval: 裁剪后每段视频长度，默认为5s
    :return:
    """
    video = cv2.VideoCapture(video_input_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    detect_interval = interval*fps  # 每隔5s对新的单张图片进行角点特征提取
    frame_index = 0
    clip_idx = 0
    # 对产生的视频分段进行保存
    while(video.isOpened()):
        ret, frame = video.read()   # 读取视频帧
        if ret == True:
            if frame_index%detect_interval==0 or frame_index == 0:
                clip_idx += 1
                clips_output_path = os.path.join(clip_output_path, "clip"+str(clip_idx)+".mp4")
                vout = cv2.VideoWriter(clips_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
            frame_index += 1
            vout.write(frame)
        else:
            break
        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break


def process(videos_dir, clips_dir):
    """
    主函数：处理视频并保存每个clip
    :param videos_dir:
    :param clips_dir:
    :return:
    """
    videos_path = []
    clips_path = []
    # 根据视频路径获得其他路径信息
    for filename in os.listdir(videos_dir):
        video_path = os.path.join(videos_dir,filename)
        videos_path.append(video_path)   # 每个视频路径
        # 构建裁剪后文件夹
        folder_name = filename[0:filename.find('.')]
        clipfolder_path = os.path.join(clips_dir,folder_name+"/")
        clips_path.append(clipfolder_path)
        if not os.path.exists(clipfolder_path):
            os.makedirs(clipfolder_path)
    # 开始处理
    for i in range(len(videos_path)):
        # 确定当前video的box坐标
        print("Start: ",videos_path[i]," to ",clips_path[i])
        video_to_clip(videos_path[i],clips_path[i],interval=5)   # 保存clip视频
        print("Done: ",videos_path[i]," to ",clips_path[i])
    print("All Done!!!!")


if __name__ == "__main__":
    # 相关文件路径
    videos_dir = "clip_test/原视频"
    clips_dir = "clip_test/裁剪后视频"
    process(videos_dir,clips_dir)