'''
    读取每帧的uncertainty分布并: 
        1.把uncertainty可视化到视频图像上  
        2.绘制每帧uncertainty的直方图
        3.顺便统计一下，有锚点标注的视频帧中，各类别anchor patch的不确定性
'''
import numpy as np
import os
import PIL.Image as Image
from numpy.core.fromnumeric import size
import torch
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA,KernelPCA
from sklearn import manifold
import time
from torchvision import transforms
from models.alexnet import MyAlexNetCMC
from offroad_dataset import OffRoadDataset
import joblib
import argparse
import myGaussianProcess

pred_res = 25    # 分类的分辨率：每个patch中间pred_res*pred_res的方块赋予该patch的类别标签
anchor_color = [(0,0,255), (0,255,0), (255,0,0), (0,255,255), (255,0,255), (255,255,0), (220,220,220), (31,102,156), (80,127,255), (140,230,240), (127,255,0), (158,168,3), (255,144,30), (214,112,218)]

def parse_option():
    parser = argparse.ArgumentParser('argument for evaluation')
    parser.add_argument('--data_folder', type=str, default=None, help='path to data')
    parser.add_argument('--pre_video',type=str, default="", help='directory of video for each frame\'s segmentation')
    parser.add_argument('--subset', type=str, default="train_fine_anno", help='subset for using')
    # resume path
    parser.add_argument('--result_path', type=str, default="results", help='path to save result')
    parser.add_argument('--note', type=str, default=None, help='comments to current train settings')
    
    opt, unknowns = parser.parse_known_args()
    if opt.note != None:
        opt.result_path = os.path.join(opt.result_path, opt.note)

    # 将不确定性文件保存为txt，方便C++程序读取
    if not os.path.isdir(os.path.join(opt.result_path, "uncertainty_txt")):
        os.makedirs(os.path.join(opt.result_path, "uncertainty_txt"))
    return opt

def fig2data(fig):
    '''
        fig = plt.figure()
        image = fig2data(fig)
        @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
        @param fig a matplotlib figure
        @return a numpy 3D array of RGBA values
    '''
    # draw the renderer
    fig.canvas.draw()
    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    image = np.asarray(image)
    return image[:,:,:-1]   # 丢弃Alpha通道

if __name__ == "__main__":
    
    # parse argument
    args = parse_option()

    patch_size = 64
    cap = cv2.VideoCapture(args.pre_video)  # 读取视频
    fps=cap.get(cv2.CAP_PROP_FPS)
    videoWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    videoHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_size= (videoWidth + videoHeight, videoHeight)
    tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_avg_var_data = []
    all_avg_var_data_frm_id = []
    # 读入锚点文件
    anchor_dict = np.load(os.path.join(args.data_folder, args.subset, "anchors_annotation.npy"), allow_pickle=True).item()
    # 各锚点类别的不确定性统计,key=anchor_type, value=[uncertainty value]
    anchor_uncertainty_dict = {}
    for i in range(10): anchor_uncertainty_dict[i] = []
    oodVideoWriter = cv2.VideoWriter(os.path.join(args.result_path,'ood.avi'), cv2.VideoWriter_fourcc('M','J','P','G'), int(fps), video_size)

    with torch.no_grad():
        ################  读入视频帧  #################
        while True:
            ret, full_img = cap.read() # 读入一帧图像
            if not ret: # 读完整段视频，退出
                print('Video end!')
                break
            frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            _patch_mask = np.zeros((full_img.shape), dtype=np.uint8)
            #################### 读入保存好的uncertainty分布 ###################
            uncertainty_data = np.load(os.path.join(args.result_path, "uncertainty_hist", str(frame_id)+"_uncertainty.npy"), allow_pickle=True).item()
            means_data = uncertainty_data["mean"]
            var_data = uncertainty_data["var"]
            # var_data: [C, n]  C是类别数量
            avg_var_data = np.mean(var_data, axis=0)
            all_avg_var_data.append(avg_var_data)
            all_avg_var_data_frm_id.append(frame_id)
            
            #################### 绘制uncertainty直方图 ########################
            # 对uncertainty的值进行归一化到0-255
            u_min = 0 #case_study_uncertainty.min()
            u_max = 4.5 #avg_var_data.max()
            print('Uncertainty range of frame{0}: [{1}, {2}]'.format(frame_id, avg_var_data.min(), avg_var_data.max()))
            fig, ax = plt.subplots(figsize=(5,5))
            ax.set_ylim(0, 1)
            # 用weights加权数据，使得y轴为百分比
            ax.hist(avg_var_data, bins=10, range=(0,u_max), weights=np.ones(len(avg_var_data))/len(avg_var_data), log=False)
            hist_img = fig2data(fig)
            hist_img = cv2.resize(hist_img, (videoHeight, videoHeight))

            ##################### 将uncertainty画到视频图像上 #####################
            _uncertainty_mask = np.zeros((full_img.shape), dtype=np.uint8)
            idx = -1
            # 滑动窗处理所有的patch，先不考虑天空，只滑动下半张图片
            for i in range(full_img.shape[0]//2, full_img.shape[0]-pred_res//2, pred_res):
                for j in range(pred_res//2, full_img.shape[1]-pred_res//2, pred_res):
                    idx += 1
                    u_color = int(min(255, max(0, (avg_var_data[idx] - u_min) / (u_max - u_min) * 255)))
                    _uncertainty_mask = cv2.rectangle(_uncertainty_mask, (j-pred_res//2, i-pred_res//2), (j+pred_res//2,i+pred_res//2), (u_color,u_color,u_color), thickness=-1) #thickness=-1 表示矩形框内颜色填充
                    # 判断当前的patch是不是anchor patch
                    if frame_id in anchor_dict.keys():
                        for _p in anchor_dict[frame_id]:
                            # 如果当前patch在某个anchor中，则统计其不确定性
                            if abs(i-_p[0])<patch_size//2 and abs(j-_p[1])<patch_size//2:
                                anchor_uncertainty_dict[_p[2]].append(avg_var_data[idx])
            
            _uncertainty_mask = cv2.applyColorMap(_uncertainty_mask, cv2.COLORMAP_HOT)
            # alpha 为第一张图片的透明度，beta 为第二张图片的透明度 cv2.addWeighted 将原始图片与 mask 融合
            merged_uncertainty_img = cv2.addWeighted(full_img, 1, _uncertainty_mask, 0.25, 0)
            merged_uncertainty_and_hist_img = np.concatenate((merged_uncertainty_img, hist_img), axis=1)
            cv2.putText(merged_uncertainty_and_hist_img, str(frame_id), (60, 60), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 0, 255), 5)
            # cv2.imwrite(os.path.join(args.result_path, str(frame_id)+"_uncertainty_and_hist.png"), merged_uncertainty_and_hist_img)
            if frame_id % 1000 == 0:
                # 将不确定性文件保存为txt，方便C++程序读取
                np.savetxt(os.path.join(args.result_path, "uncertainty_txt", "all_frame_uncertainty.txt"), all_avg_var_data, fmt="%.6f",delimiter="\t")
                np.savetxt(os.path.join(args.result_path, "uncertainty_txt", "all_frame_uncertainty_frameID.txt"), all_avg_var_data_frm_id, fmt="%d",delimiter="\n")
                np.save(os.path.join(args.result_path, "uncertainty_txt", "anchor_uncertainty_dict.npy"), anchor_uncertainty_dict)
            oodVideoWriter.write(merged_uncertainty_and_hist_img)
            plt.close(fig)

    # 将不确定性文件保存为txt，方便C++程序读取
    np.savetxt(os.path.join(args.result_path, "uncertainty_txt", "all_frame_uncertainty.txt"), all_avg_var_data, fmt="%.6f",delimiter="\t")
    np.savetxt(os.path.join(args.result_path, "uncertainty_txt", "all_frame_uncertainty_frameID.txt"), all_avg_var_data_frm_id, fmt="%d",delimiter="\n")
    np.save(os.path.join(args.result_path, "uncertainty_txt", "anchor_uncertainty_dict.npy"), anchor_uncertainty_dict)
    oodVideoWriter.release()