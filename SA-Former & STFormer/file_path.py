import os

txt_file = "/home/uestc/cy/features_extract/sample/with_self_extubation_tendency_video_paths.txt"
video_data = "/home/uestc/cy/features_extract/video_data/with_self_extubation_tendency"
wfile = open(txt_file, 'w')
for folder in sorted(os.listdir(video_data)):
    file_path = os.path.join(video_data, folder)
    wfile.writelines(file_path)
    wfile.write('\n')
wfile.close()


