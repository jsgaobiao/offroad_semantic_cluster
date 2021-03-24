'''
Author: Gao Biao
Date: 2020-12-17 10:49:39
LastEditTime: 2021-01-14 22:02:43
Description: 评估对比学习预测出的细粒度语义聚类结果（计算属于同一类别的像素方差等指标）
FilePath: /offroad_semantic_cluster/src/eval_road.py
'''

import numpy as np
import cv2
import os
import multiprocessing

read_skip = 5   # 每隔若干帧取一张图片进行评估
kmeans = 6      # 类别数量
cap = cv2.VideoCapture("/home/gaobiao/Documents/offroad_semantic_cluster/data/2.avi")  # 读取待标注数据
video_pred_dir = "/home/gaobiao/Documents/offroad_semantic_cluster/src/results/0130_dataAug_BG320_test1_to_test2_kmeans6/video_pred"
fps=cap.get(cv2.CAP_PROP_FPS)
tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
all_pixel_bgr_mean = np.zeros((kmeans, 3))  # 每一类rgb均值
all_pixel_bgr_num = np.zeros((kmeans, 3))   # 每一类像素数量
all_pixel_bgr_std = np.zeros((kmeans, 3))   # 每一类rgb方差
anchor_color = [(0,0,255), (0,255,0), (255,0,0), (0,255,255), (255,0,255), (255,255,0), (255, 191, 0), (0, 191, 255), (128, 0, 255)]

cores = 8 #multiprocessing.cpu_count()
pool = multiprocessing.Pool(processes=cores)

# 构造anchor_color的逆，作为字典
color_dict = {}
for idx, c in enumerate(anchor_color):
    color_dict[str(c[0])+str(c[1])+str(c[2])] = idx

################  读入视频帧 (先计算均值) #################
while True:
    ret, full_img = cap.read() # 读入一帧图像
    if not ret: # 读完整段视频，退出
        break
    frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if (frame_id % read_skip != 1):
        continue    
    # 读入full_img对应的预测结果
    pred_frame = cv2.imread(os.path.join(video_pred_dir, str(frame_id)+'_pred_all.mask.png'))
    # 如果预测结果不存在就continue
    if pred_frame is None: continue
    # 只预测了下半张图片，所以只统计下半张图片上的像素
    for i in range(full_img.shape[0]//2, full_img.shape[0]):
        for j in range(0, full_img.shape[1]):
            p_color = pred_frame[i][j]
            color_key = str(p_color[0])+str(p_color[1])+str(p_color[2])
            if color_key in color_dict.keys():
                lab = color_dict[color_key]
                all_pixel_bgr_mean[lab] += full_img[i][j]
                all_pixel_bgr_num[lab] += [1, 1, 1]
    # print info
    print('Read frame: [{0}]'.format(frame_id))
cap.release()  

# 计算均值
all_pixel_bgr_mean = all_pixel_bgr_mean / all_pixel_bgr_num

################  读入视频帧 (再计算方差) #################
cap = cv2.VideoCapture("/home/gaobiao/Documents/offroad_semantic_cluster/data/2.avi")  # 读取待标注数据
while True:
    ret, full_img = cap.read() # 读入一帧图像
    if not ret: # 读完整段视频，退出
        break
    frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if (frame_id % read_skip != 1):
        continue    
    # 读入full_img对应的预测结果
    pred_frame = cv2.imread(os.path.join(video_pred_dir, str(frame_id)+'_pred_all.mask.png'))
    # 如果预测结果不存在就continue
    if pred_frame is None: continue
    # 只预测了下半张图片，所以只统计下半张图片上的像素
    for i in range(full_img.shape[0]//2, full_img.shape[0]):
        for j in range(0, full_img.shape[1]):
            p_color = pred_frame[i][j]
            color_key = str(p_color[0])+str(p_color[1])+str(p_color[2])
            if color_key in color_dict.keys():
                lab = color_dict[color_key]
                all_pixel_bgr_std[lab] += (full_img[i][j] - all_pixel_bgr_mean[lab])**2
    # print info
    print('Read frame: [{0}]'.format(frame_id))
cap.release()

# 计算均值
all_pixel_bgr_std = all_pixel_bgr_std / all_pixel_bgr_num
#################  统计每类像素BGR值的均值方差 ###################
f_out = open("all_pixel_bgr_stats_BG320_test1_to_test2.csv", "w")
for i in range(kmeans):
    f_out.write(str(list(all_pixel_bgr_mean[i]))[1:-1]+',')
    f_out.write(str(list(all_pixel_bgr_std[i]))[1:-1]+',')
    f_out.write(str(list(all_pixel_bgr_num[i]))[1:-1])
    f_out.write('\n')
f_out.close()
print(all_pixel_bgr_mean)
print(all_pixel_bgr_std)
print(all_pixel_bgr_num)
