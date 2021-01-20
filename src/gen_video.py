'''
Author: Gao Biao
Date: 2020-12-17 10:49:39
LastEditTime: 2021-01-12 17:32:03
Description: 将patch_cluster.py生成的视频帧语义预测结果的png图片连起来，输出视频
FilePath: /offroad_semantic_cluster/src/gen_video.py
'''

import numpy as np
import cv2
import os

from numpy.core.numeric import full

cap = cv2.VideoCapture("/home/gaobiao/Documents/offroad_semantic_cluster/data/1_cut.mp4")  # 读取待标注数据
video_pred_dir = "/home/gaobiao/Documents/offroad_semantic_cluster/src/results/1225_RGB_fine_anno_testnew/video_pred"
fps=cap.get(cv2.CAP_PROP_FPS)
video_size=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)//2))
tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

cap_out = cv2.VideoWriter("/home/gaobiao/Documents/offroad_semantic_cluster/data/test.avi",cv2.VideoWriter_fourcc('M','J','P','G'), int(fps), video_size)
################  读入视频帧  #################
while True:
    ret, full_img = cap.read() # 读入一帧图像
    frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if not ret: # 读完整段视频，退出
        break
    if (frame_id % 5 != 1):
        continue
    # 读入对应的预测结果
    pred_frame = cv2.imread(os.path.join(video_pred_dir, str(frame_id)+'_pred_all.png'))
    cat_frame = np.concatenate((full_img, pred_frame), axis=1)
    cap_out.write(cv2.resize(cat_frame, video_size))
    # print info
    print('Save video fine segmentation: [{0}]'.format(frame_id))
cap_out.release()