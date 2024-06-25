from PIL import Image
import os
import numpy as np
import cv2


def main(video_name, npy_file_folder, video_path, mask_video_path):
    # video name
    print("Deal with " + str(video_name))
    # original video path
    video_path = os.path.join(video_path, video_name+'.mp4')
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # store mask video path
    mask_video_path = os.path.join(mask_video_path, video_name + '.mp4')
    vout = cv2.VideoWriter(mask_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    npy_files = [f for f in os.listdir(npy_file_folder) if f.endswith(".npy")]
    for npy_file in npy_files:
        # npy to image
        npy_file = os.path.join(npy_file_folder, npy_file)
        image = Image.fromarray(np.array(np.load(npy_file)).astype(np.uint8)).convert("L")
        image = np.array(image)
        image[image != 0] = 255
        # print(image.shape)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # print(image.shape)

        # image to mask video
        vout.write(image)
    vout.release()
    print("Done!")


if __name__ == "__main__":
    npy_file_folder = 'C:/Users/admin/Desktop/mask'
    video_path = 'C:/Users/admin/Desktop'
    mask_video_path = 'C:/Users/admin/Desktop'
    for i in os.listdir(npy_file_folder):
        npy_path = os.path.join(npy_file_folder, i)
        main(i, npy_path, video_path, mask_video_path)
        break




