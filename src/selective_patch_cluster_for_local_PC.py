'''
Author: Gao Biao
Date: 2020-05-19
LastEditTime: 2020-05-19 22:11:46
Description: 借鉴selective classification的思路，根据risk-coverage指标找到需要补充标注的聚类类别
FilePath: /offroad_semantic_cluster/src/selective_patch_cluster.py
'''

from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
import numpy as np
import torch
import cv2
import os
import math
import time
from torchvision import transforms
from models.alexnet import MyAlexNetCMC
from offroad_dataset import OffRoadDataset
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import joblib
import argparse

pred_res = 25    # 分类的分辨率：每个patch中间pred_res*pred_res的方块赋予该patch的类别标签
anchor_color = [(0,0,255), (0,255,0), (255,0,0), (0,255,255), (255,0,255), (255,255,0), (220,220,220), (31,102,156), (80,127,255), (140,230,240), (127,255,0), (158,168,3), (255,144,30), (214,112,218)]
anchor_marker = ['.','.','.','.','.','.','x','x','s','s','s','s','*','*']
anchor_label = [u"0:路",u"1:石头",u"2:植物",u"3:路边",u"4:建筑",u"5:碎石",u"6:水泥堆",u"7:木材",u"8:草泥/落叶","9:"]
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
#有中文出现的情况，需要u'内容'
coverage_bins = 100

def parse_option():
    parser = argparse.ArgumentParser('argument for evaluation')
    parser.add_argument('--data_folder', type=str, default=None, help='path to data')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--subset', type=str, default="train", help='subset for training')
    parser.add_argument('--kmeans', type=int, help='kmeans聚类的类别数量')
    parser.add_argument('--pre_video',type=str, default="", help='directory of video for each frame\'s segmentation')
    parser.add_argument('--batch_pred', type=int, default=8000, help='将多个patch放到一个batch中再进行标签预测，加快计算速度')
    parser.add_argument('--background', type=int, default=320, help='size of background patch')
    # resume path
    parser.add_argument('--model_path', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--result_path', type=str, default="results", help='path to save result')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--note', type=str, default=None, help='comments to current train settings')
    parser.add_argument('--softmax', action='store_true', help='using softmax contrastive loss rather than NCE')
    parser.add_argument('--nce_k', type=int, default=1, help='负样本数量')
    parser.add_argument('--nce_t', type=float, default=0.07)
    parser.add_argument('--nce_m', type=float, default=0.5, help='the momentum for dynamically updating the memory.')
    parser.add_argument('--feat_dim', type=int, default=128, help='dim of feat for inner product')
    parser.add_argument('--in_channel', type=int, default=6, help='dim of input image channel (3: RGB, 5: RGBXY, 6: RGB+Background)')

    opt, unknowns = parser.parse_known_args()
    # 要保存每个anchor的特征，所以batch_size必须是1
    opt.batch_size = 1

    if (opt.data_folder is None) or (opt.model_path is None) or (opt.result_path is None):
        raise ValueError('one or more of the folders is None: data_folder | model_path | result_path')
    
    if opt.note != None:
        opt.result_path = os.path.join(opt.result_path, opt.note, "cluster_results")
    if not os.path.isdir(opt.result_path):
        os.makedirs(opt.result_path)
    if not os.path.isdir(opt.result_path.replace("cluster_results", "features")):
        os.makedirs(opt.result_path.replace("cluster_results", "features"))
    if not os.path.isdir(opt.result_path.replace("cluster_results", "uncertainty_hist")):
        os.makedirs(opt.result_path.replace("cluster_results", "uncertainty_hist"))
    if not os.path.isdir(opt.result_path.replace("cluster_results", "cluster_eval")):
        os.makedirs(opt.result_path.replace("cluster_results", "cluster_eval"))
    if not os.path.isdir(opt.result_path.replace("cluster_results", "risk_coverage")):
        os.makedirs(opt.result_path.replace("cluster_results", "risk_coverage"))
    
    # 将使用的配置表保存到文件中
    args_to_save = parser.parse_args()
    print(args_to_save)
    return opt

def set_model(args):
    print('==> loading pre-trained model')
    if torch.cuda.is_available():
        ckpt = torch.load(args.model_path)
    else:
        ckpt = torch.load(args.model_path, map_location=torch.device('cpu'))
    model = MyAlexNetCMC(ckpt['opt'].feat_dim, in_channel=args.in_channel)
    model.load_state_dict(ckpt['model'])
    print("==> loaded checkpoint '{}' (epoch {})".format(args.model_path, ckpt['epoch']))
    print('==> done')
    
    model.eval()
    return model

def get_data_loader(args, subset='train'):
    """get the data loader + 数据增强/归一化"""
    mean = [0.5200442, 0.5257094, 0.517397]
    std = [0.335111, 0.33463535, 0.33491987]

    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    ########## 注意：此data_set提取的正样本数量和负样本数量相等，用于计算锚点到正负样本的平均距离 #########
    sub_dataset = OffRoadDataset(args.data_folder, subset=subset,
                                                pos_sample_num=args.nce_k, 
                                                neg_sample_num=args.nce_k, 
                                                transform=data_transform,
                                                background_size=args.background, 
                                                channels=args.in_channel, 
                                                patch_size=64)
    # data loader
    # TODO: 用sampler或batch_sampler放入tensorboard可视化
    data_loader = torch.utils.data.DataLoader(sub_dataset, batch_size=args.batch_size, 
                                                            shuffle=False,
                                                            num_workers=args.num_workers, 
                                                            pin_memory=True, 
                                                            sampler=None)
    # num of samples
    n_data = len(sub_dataset)
    print('number of samples (subset:{}): {}'.format(subset, n_data))
    return data_loader, n_data, sub_dataset

def calcAllFeature(args, model, data_loader, n_data, recalc_feature=True):
    ''' 计算所有anchor patch的特征向量，并保存 '''
    # 如果存在已经保存的结果，就载入
    if os.path.isfile(os.path.join(args.result_path, "all_patch_features.npy")) and recalc_feature==False:
        all_features = np.load(os.path.join(args.result_path, "all_patch_features.npy"))
        all_features_type = np.load(os.path.join(args.result_path, "all_patch_features_type.npy"))
    else:
        # 否则，计算所有patch的特征
        all_features = np.zeros((n_data, args.feat_dim))
        all_features_type = np.zeros(n_data)
        with torch.no_grad():
            for idx, (anchor, _, _, frame_id, full_img, anchor_xy, _, _, anchor_type) in enumerate(data_loader):
                # anchor shape: [batch_size, 1, channel, H, W] # neg_sample,pos_sample shape: [batch_size, K, channel, H, W]
                batch_size = anchor.size(0)
                # batch_size必须是1，因为要逐个保存每个anchor的特征
                assert(batch_size == 1)
                # inputs shape --> [batch_size, (1), channel, H, W]
                inputs = anchor
                inputs_shape = list(inputs.size())
                # print('inputs_shape:{}'.format(inputs_shape))
                # inputs shape --> [batch_size*(1), channel, H, W]
                inputs = inputs.view((inputs_shape[0]*inputs_shape[1], inputs_shape[2], inputs_shape[3], inputs_shape[4]))
                inputs = inputs.float()
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                # ===================forward=====================
                feature = model(inputs)     # [batch_size*(1), feature_dim]
                for _bs in range(batch_size):
                    all_features[idx] = feature[_bs].cpu().numpy()
                    all_features_type[idx] = anchor_type
                # ===============================================
                # print info
                if (idx + 1) % args.print_freq == 0:
                    print('Calculate feature: [{0}/{1}]'.format(idx + 1, len(data_loader)))
            np.save(os.path.join(args.result_path, "all_patch_features.npy"), all_features)
            np.save(os.path.join(args.result_path, "all_patch_features_type.npy"), all_features_type)
    return all_features, all_features_type

def getPatchXY(_img, _x, _y, anchor_width):
    ''' 
        从图像img中获取中心坐标为(_x, _y)的patch左上右下角坐标 
        _x: 宽， _y: 高
    '''
    h, w = _img.shape[0:2]
    ps = anchor_width // 2
    p_left_top = [max(_x-ps, 0), max(_y-ps, 0)]
    p_right_down = [min(_x+ps, w), min(_y+ps, h)]   # 右侧开区间
    # 如果在图像边缘处要特殊处理
    if (p_right_down[0] - p_left_top[0] < anchor_width):
        if (p_left_top[0] == 0): p_right_down[0] = p_left_top[0] + anchor_width
        if (p_right_down[0] == w): p_left_top[0] = p_right_down[0] - anchor_width
    if (p_right_down[1] - p_left_top[1] < anchor_width):
        if (p_left_top[1] == 0): p_right_down[1] = p_left_top[1] + anchor_width
        if (p_right_down[1] == h): p_left_top[1] = p_right_down[1] - anchor_width
    return p_left_top, p_right_down

def predict_video(args, cluster_model, rbf_threshold):
    '''
        [处理所有视频帧] 预测视频中每一帧的分割结果,并把每个patch对应的特征值保存下来
                        在本地可视化的时候，实时计算rbf值和聚类类别，可以随时调整阈值
    '''
    patch_size = 64
    cap = cv2.VideoCapture(args.pre_video)  # 读取待标注数据
    fps=cap.get(cv2.CAP_PROP_FPS)
    video_size=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)//2))
    tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    videoWriter = cv2.VideoWriter(os.path.join(args.result_path.replace("cluster_results", "."),'rbf_response_risk.avi'), cv2.VideoWriter_fourcc('M','J','P','G'), int(30), video_size)
    tot_patches = 0
    selected_patches =0 
    with torch.no_grad():
        ################  读入视频帧  #################
        while True:
            ret, full_img = cap.read() # 读入一帧图像
            if not ret: # 读完整段视频，退出
                print('Video end!')
                break
            frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            features_for_save = np.array([])
            # 检查是否有已经计算好的特征文件，如果有就不用重新计算
            if os.path.isfile(os.path.join(args.result_path.replace("cluster_results", "features"), str(frame_id)+".npy")):
                # [注意: features_for_save 共2+128维，前2维是patch对应的坐标，后128维是特征向量]
                features_for_save = np.load(os.path.join(args.result_path.replace("cluster_results", "features"), str(frame_id)+".npy"))
            else:
                continue
            
            # 将risk（rbf-based）可视化到图像上
            _patch_mask = np.zeros((full_img.shape), dtype=np.uint8)
            for i in range(features_for_save.shape[0]):
                _i, _j = int(features_for_save[i,0]), int(features_for_save[i,1])
                # 对每个patch进行聚类类别预测
                _patch_label = cluster_model.predict(np.expand_dims(features_for_save[i,2:], axis=0))
                cluster_center_feat = cluster_model.cluster_centers_[_patch_label]
                # 计算到聚类中心的欧几里得距离
                euclidean_dis = np.linalg.norm(features_for_save[i, 2:] - cluster_center_feat)
                # 计算到聚类中心的RBF距离作为概率值
                _sigma = 0.2   # RBF kernel 的参数 length scale
                rbf_dis = np.exp(-euclidean_dis**2/(2*_sigma*_sigma))
                rbf_response = 1-rbf_dis    # 值域是0-1
                # 将patch分类得到的类别标签绘制到图像上（只绘制i,j为中心，pred_res*pred_res的方块）
                # 只有超出rbf_threshold的patch需要被丢弃，根据距离远近可视化它们
                tot_patches += 1
                if (rbf_response > rbf_threshold):                    
                    u_color = int(min(255, max(0, (rbf_response - rbf_threshold) / (1 - rbf_threshold) * 255)))
                    _patch_mask = cv2.rectangle(_patch_mask, (_j-pred_res//2,_i-pred_res//2), (_j+pred_res//2,_i+pred_res//2), (u_color,u_color,u_color), thickness=-1) #thickness=-1 表示矩形框内颜色填充
                else:
                    selected_patches += 1
            
            _patch_mask = cv2.applyColorMap(_patch_mask, cv2.COLORMAP_HOT)
            # alpha 为第一张图片的透明度，beta 为第二张图片的透明度 cv2.addWeighted 将原始图片与 mask 融合
            merged_full_img = cv2.addWeighted(full_img, 0.8, _patch_mask, 0.2, 0)
            merged_img = np.concatenate((merged_full_img, full_img), axis=1)
            videoWriter.write(cv2.resize(merged_img, video_size))
            print("frame {}, real coverage: {}".format(frame_id, selected_patches / tot_patches))
    videoWriter.release()
            

def evalClusterResult(args, cluster_model, cluster_method, anchor_dict):
    '''
        计算聚类结果和锚点标注的吻合度:
        每张图上有n个锚点，对应了n*(n-1)对约束（互为正样本/负样本），评价聚类结果能满足多少约束对
        type_of_anchor: 聚类出来的锚点标签
    '''
    tot = 0
    cluster_precision_sum = 0
    cluster_precision = {}
    error_of_same_type = np.zeros(20)    # 估算一下每个类别锚点预测的错误数(同anchor type被预测为不同类别)
    error_of_diff_type = np.zeros(20)    # 估算一下每个类别锚点预测的错误数(不同anchor type被预测为相同类别)
    for _frm_id in sorted(anchor_dict.keys()):
        # 计算当前帧中正负样本约束和聚类标签的吻合度
            cnt = 0
            _n = len(anchor_dict[_frm_id])
            for i in range(_n):
                for j in range(i):
                    # 锚点i和j互为正样本
                    cnt = cnt + (1 if (anchor_dict[_frm_id][i][2]==anchor_dict[_frm_id][j][2] and cluster_model.labels_[tot+i]==cluster_model.labels_[tot+j]) else 0)
                    # 锚点i和j互为负样本
                    cnt = cnt + (1 if (anchor_dict[_frm_id][i][2] != anchor_dict[_frm_id][j][2]) and (cluster_model.labels_[tot+i] != cluster_model.labels_[tot+j]) else 0)
                    # 同类锚点预测为不同类别
                    if (anchor_dict[_frm_id][i][2] == anchor_dict[_frm_id][j][2]) and (cluster_model.labels_[tot+i] != cluster_model.labels_[tot+j]):
                        error_of_same_type[anchor_dict[_frm_id][i][2]] += 2
                    # 不同类锚点预测为同类别
                    if (anchor_dict[_frm_id][i][2] != anchor_dict[_frm_id][j][2]) and (cluster_model.labels_[tot+i] == cluster_model.labels_[tot+j]):
                        error_of_diff_type[anchor_dict[_frm_id][i][2]] += 1
                        error_of_diff_type[anchor_dict[_frm_id][j][2]] += 1
            # 当前帧的吻合度 cnt*2/n(n-1)
            cluster_precision[_frm_id] = cnt * 2 / (_n * (_n - 1))
            cluster_precision_sum += cnt * 2 / (_n * (_n - 1))
            tot += _n

    print("Average cluster precision: {}".format(cluster_precision_sum / len(anchor_dict.keys())))
    # with open(os.path.join(args.result_path, 'average_cluster_precision.txt'), 'a+') as f_cp:
        # f_cp.write("\nModel path: {}\n{}={}, Average cluster precision: {}\n\n".format(args.model_path, cluster_method, args.kmeans, cluster_precision_sum / len(anchor_dict.keys())))
    return cluster_precision
 
def load_anno(args):
    '''
    读入人工标注的锚点数据
    '''
    file_name = os.path.join(args.data_folder, args.subset, "anchors_annotation.npy")
    anchor_dict = np.load((file_name), allow_pickle=True).item()
    anchor_list = []
    for _f_id in sorted(anchor_dict.keys()):
        for _ac_id in range(len(anchor_dict[_f_id])):
            anchor_list.append([_f_id] + anchor_dict[_f_id][_ac_id])  # [frame_id, anchor_x, anchor_y, anchor_type]

    return anchor_dict, anchor_list

def calcRiskCoverageOfAnchors(args, cluster_method, cluster_model, anchor_feature_list, type_of_anchor_feature_list, selected_coverage=0.8):
    ''' 根据锚点聚类结果，计算risk-coverage曲线并得到coverage对应的阈值 '''
    risk_coverage = []  # [max_rbf_prob, pseudo_rbf_prob(到每个聚类中心的距离)]
    for i in range(anchor_feature_list.shape[0]):
        pseudo_rbf_prob = np.zeros(args.kmeans)

        for _k in range(args.kmeans):
            cluster_center_feat = cluster_model.cluster_centers_[_k]
            # 计算到聚类中心的欧几里得距离
            euclidean_dis = np.linalg.norm(anchor_feature_list[i, :] - cluster_center_feat)
            # 计算到聚类中心的RBF距离作为概率值
            _sigma = 0.2   # RBF kernel 的参数 length scale
            rbf_dis = np.exp(-euclidean_dis**2/(2*_sigma*_sigma))
            pseudo_rbf_prob[_k] = rbf_dis
        # 取1-最大值，这样离聚类中心越近，值越小
        risk_coverage.append([1-pseudo_rbf_prob.max(), cluster_model.labels_[i], type_of_anchor_feature_list[i] ])

    ''' 计算Risk_Coverage曲线，Risk使用到聚类中心的RBF_Prob计算'''
    # risk_coverage: [1-RBF, 聚类类别, anchor类别]
    risk_coverage = sorted(risk_coverage, key=lambda x:x[0])    # 按照response排序，逐步增加coverage并绘risk_coverage图
    rbf_threshold = risk_coverage[int(selected_coverage * len(risk_coverage))]
    # 得到selected_coverage所对应的rbf阈值
    return rbf_threshold[0]

def main():# 供直接运行本脚本

    # parse argument
    args = parse_option()
    # 读入标注数据
    anchor_dict, anchor_list = load_anno(args)

    # 计算所有patch对应的特征向量
    anchor_feature_list = np.load(os.path.join(args.result_path, "all_patch_features.npy"), )
    type_of_anchor_feature_list = np.load(os.path.join(args.result_path, "all_patch_features_type.npy"))

    cluster_method = "kmeans" # "kmeans" or "kmedoids" or "Agglomerative"
    if cluster_method == "kmeans":
        # K-means聚类
        if (os.path.isfile(os.path.join(args.result_path, "kmeans{}.pkl".format(args.kmeans)))):
            cluster_model = joblib.load(os.path.join(args.result_path, "kmeans{}.pkl".format(args.kmeans)))
        else:
            cluster_model = KMeans(n_clusters=args.kmeans).fit(anchor_feature_list)
            joblib.dump(cluster_model, os.path.join(args.result_path, "kmeans{}.pkl".format(args.kmeans)))
        print("K-means {} cluster over!".format(args.kmeans))
    elif cluster_method == "kmedoids":
        # K-medoids聚类
        if (os.path.isfile(os.path.join(args.result_path, "kmedoids{}.pkl".format(args.kmeans)))):
            cluster_model = joblib.load(os.path.join(args.result_path, "kmedoids{}.pkl".format(args.kmeans)))
        else:
            cluster_model = KMedoids(n_clusters=args.kmeans, metric="cosine").fit(anchor_feature_list)
            joblib.dump(cluster_model, os.path.join(args.result_path, "kmedoids{}.pkl".format(args.kmeans)))
        print("K-Medoids {} cluster over!".format(args.kmeans))
    elif cluster_method == "Agglomerative":       
        # 层次聚类
        cluster_model = AgglomerativeClustering(n_clusters=args.kmeans)
        cluster_model = cluster_model.fit(anchor_feature_list)
        print("AgglomerativeClustering {} cluster over!".format(args.kmeans))
    
    ''' 评价聚类结果 '''
    # 0. 计算聚类结果和锚点标注的吻合度 [可选]
    # cluster_precision = evalClusterResult(args, cluster_model, cluster_method, anchor_dict)

    # 1. 根据锚点聚类结果，计算risk-coverage曲线并得到coverage对应的阈值
    rbf_threshold = calcRiskCoverageOfAnchors(args, cluster_method, cluster_model, anchor_feature_list, type_of_anchor_feature_list, selected_coverage=0.9)

    # 2. 对所有视频帧进行语义分割结果的预测(读取已经保存好的features)
    predict_video(args, cluster_model, rbf_threshold)

if __name__ == '__main__':
    main()
