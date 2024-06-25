import cv2
import os
import numpy as np

def process_c3d_video(video, save_dir, video_filename):
    # Initialize a VideoCapture object to read video data into a numpy array
    capture = cv2.VideoCapture(video)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    resize_width = 112
    resize_height = 112

    # Make sure splited video has at least 16 frames
    EXTRACT_FREQUENCY = 4
    count = 0
    i = 0
    retaining = True

    # image store
    frame_sample_count = frame_count // EXTRACT_FREQUENCY
    # print(frame_sample_count)
    # buffer = np.empty((frame_sample_count, resize_width, resize_height, 3), np.dtype('float32'))

    while count < frame_count - 1:
        retaining, frame = capture.read()
        if frame is None:
            continue

        # print(count, i)
        if count % EXTRACT_FREQUENCY == 0:
            # resize image to (112, 112)
            frame = cv2.resize(frame, (resize_width, resize_height))
            if not os.path.exists(os.path.join(save_dir, video_filename)):
                os.makedirs(os.path.join(save_dir, video_filename))
            cv2.imwrite(filename=os.path.join(save_dir, video_filename, '0000{}.jpg'.format(str(i))), img=frame)
            # buffer[i] = frame
            i += 1
        count += 1

    # Release the VideoCapture once it is no longer needed
    capture.release()


if __name__ == "__main__":
    # 读取所有样本
    filedata_names = []
    filelabel_names = []
    outdata_names = []
    video_names = []
    root_dir = '/home/mii2/project/cy/uex_mask_video/uex_class4_mask'
    output_dir = "/home/mii2/project/cy/uex_mask_video/uex_class4_mask_image"
    for file in sorted(os.listdir(root_dir)):
        file_path = os.path.join(root_dir, file)
        out_path = os.path.join(output_dir, file)
        for name in sorted(os.listdir(file_path)):
            video_name = name.split('.')[0]
            video_names.append(video_name)
            outdata_names.append(out_path)
            filedata_names.append(os.path.join(file_path, name))
            filelabel_names.append(file)
    # 将label名称转化为数字
    label2index = {label: index for index, label in enumerate(sorted(set(filelabel_names)))}
    label_array = np.array([label2index[label] for label in filelabel_names], dtype=int)
    # 处理
    for i in range(len(outdata_names)):
        process_c3d_video(filedata_names[i], outdata_names[i], video_names[i])

