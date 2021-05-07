'''
    根据不确定性指标，评价聚类结果的合理性
'''
from sklearn_extra.cluster import KMedoids
import numpy as np
import os
from numpy.core.fromnumeric import size
from sklearn import cluster
import math
from sklearn.metrics.pairwise import euclidean_distances
import torch
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
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
anchor_marker = ['.','.','.','.','.','.','x','x','s','s','s','s','*','*']

def parse_option():
    parser = argparse.ArgumentParser('argument for evaluation')
    parser.add_argument('--data_folder', type=str, default=None, help='path to data')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--subset', type=str, default="train", help='subset for training')
    parser.add_argument('--kmeans', type=int, help='kmeans聚类的类别数量')
    parser.add_argument('--show_kmeans_center', type=bool, default=True, help='可视化kmeans聚类中心')
    parser.add_argument('--pre_video',type=str, default="", help='directory of video for each frame\'s segmentation')
    parser.add_argument('--batch_pred', type=int, default=8000, help='将多个patch放到一个batch中再进行标签预测，加快计算速度')
    parser.add_argument('--background', type=int, default=192, help='size of background patch')
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
    parser.add_argument('--in_channel', type=int, default=3, help='dim of input image channel (3: RGB, 5: RGBXY, 6: RGB+Background)')

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

def my_softmax(x):
    """Compute the softmax in a numerically stable way."""
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

def my_entropy(probs, base=None):
    """ Computes entropy of label distribution. """
    ent = 0.
    # Compute entropy
    base = math.e if base is None else base
    for i in probs:
        ent -= i * math.log(i, base)
    return ent

def predict_video_frame_entropy(args, model, cluster_model, features_for_save):
    '''
    根据到聚类中心的距离，预测patch到每个类别的概率（类似Softmax），然后计算熵
    [注意: features_for_save 共2+128维，前2维是patch对应的坐标，后128维是特征向量]
    '''
    entropy_list_cosine = []
    entropy_list_euclidean = []
    for i in range(features_for_save.shape[0]):
        pseudo_cosine_prob = np.zeros(args.kmeans)
        pseudo_euclidean_prob = np.zeros(args.kmeans)
        for _k in range(args.kmeans):
            cluster_center_feat = cluster_model.cluster_centers_[_k]
            # 计算到聚类中心的余弦距离作为概率值
            cos_dis = np.dot(features_for_save[i, 2:], cluster_center_feat.T) / (np.linalg.norm(features_for_save[i, 2:]) * np.linalg.norm(cluster_center_feat))
            # 计算到聚类中心的欧几里得距离作为概率值
            euclidean_dis = np.linalg.norm(features_for_save[i, 2:] - cluster_center_feat)
            # 计算到聚类中心的RBF距离作为概率值
            # _sigma = 0.1    # RBF kernel 的参数 length scale
            # rbf_dis = np.exp(-euclidean_dis/(2*_sigma*_sigma))
            pseudo_cosine_prob[_k] = cos_dis
            pseudo_euclidean_prob[_k] = euclidean_dis

        # 使用Softmax归一化,并计算熵
        pseudo_cosine_prob = my_softmax(pseudo_cosine_prob)
        entropy_list_cosine.append(my_entropy(pseudo_cosine_prob))

        pseudo_euclidean_prob = my_softmax(pseudo_euclidean_prob)
        entropy_list_euclidean.append(my_entropy(pseudo_euclidean_prob))
    return np.array(entropy_list_cosine), np.array(entropy_list_euclidean)

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

def my_softmax(x):
    """Compute the softmax in a numerically stable way."""
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

def my_entropy(probs, base=None):
    """ Computes entropy of label distribution. """
    ent = 0.
    # Compute entropy
    base = math.e if base is None else base
    for i in probs:
        ent -= i * math.log(i, base)
    return ent

def evalUncertaintyOfCluster(args, cluster_model, anchor_feature_list):
    '''根据不确定性评估聚类的结果'''
    entropy_list_cosine = []
    entropy_list_euc = []
    entropy_list_rbf = []
    for i in range(anchor_feature_list.shape[0]):
        pseudo_cosine_prob = np.zeros(args.kmeans)
        pseudo_rbf_prob = np.zeros(args.kmeans)
        pseudo_euc_prob = np.zeros(args.kmeans)

        for _k in range(args.kmeans):
            cluster_center_feat = cluster_model.cluster_centers_[_k]
            # 计算到聚类中心的余弦距离作为概率值
            cos_dis = np.dot(anchor_feature_list[i, :], cluster_center_feat.T) / (np.linalg.norm(anchor_feature_list[i, :]) * np.linalg.norm(cluster_center_feat))
            # 计算到聚类中心的欧几里得距离
            euclidean_dis = np.linalg.norm(anchor_feature_list[i, :] - cluster_center_feat)
            # 计算到聚类中心的RBF距离作为概率值
            _sigma = 0.2   # RBF kernel 的参数 length scale
            rbf_dis = np.exp(-euclidean_dis**2/(2*_sigma*_sigma))
            pseudo_cosine_prob[_k] = cos_dis
            pseudo_rbf_prob[_k] = rbf_dis
            pseudo_euc_prob[_k] = 1.0 / euclidean_dis
            
        # 使用Softmax归一化,并计算熵
        pseudo_cosine_prob = my_softmax(pseudo_cosine_prob)
        entropy_list_cosine.append(my_entropy(pseudo_cosine_prob))

        # pseudo_rbf_prob = my_softmax(pseudo_rbf_prob)
        # entropy_list_rbf.append(my_entropy(pseudo_rbf_prob))
        entropy_list_rbf.append(pseudo_rbf_prob.max())

        pseudo_euc_prob = my_softmax(pseudo_euc_prob)
        entropy_list_euc.append(my_entropy(pseudo_euc_prob))

    print("cosine_entropy:", np.array(entropy_list_cosine).mean())
    print("rbf_entropy:", np.array(entropy_list_rbf).mean())
    print("euclidean_entropy:", np.array(entropy_list_euc).mean())

def vis_clustered_feature(args, cluster_method, cluster_model, _feature_list, _type_of_feature_list=None, mode='pca'):
    ''' 将聚类后的anchor特征可视化 '''

    ''' 降维特征向量 '''
    if mode == 'pca':
        # Kernel PCA降成2维特征后，可视化类别簇
        pca = PCA(n_components=2)
        # pca = KernelPCA(n_components=2, kernel='rbf')
        x_low_dim = pca.fit(_feature_list).transform(_feature_list)
    elif mode == 't-sne':
        # perpexity	混乱度，表示t-SNE优化过程中考虑邻近点的多少，默认为30，建议取值在5到50之间
        # early_exaggeration 表示嵌入空间簇间距的大小，默认为12，该值越大，可视化后的簇间距越大
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=1, perplexity=30, early_exaggeration=20)
        x_low_dim = tsne.fit_transform(_feature_list)

    ''' 可视化锚点的真值 '''
    if _type_of_feature_list is not None: 
        assert _feature_list.shape[0] == _type_of_feature_list.shape[0]
        fig, ax = plt.subplots(figsize=(8, 8))
        # 先画anchor features
        for i in range(_type_of_feature_list.shape[0]):
            c = anchor_color[int(_type_of_feature_list[i])]
            hex_c = '#%02x%02x%02x' % (c[2], c[1], c[0])
            ax.scatter(x_low_dim[i, 0], x_low_dim[i, 1], c=hex_c, marker=anchor_marker[int(_type_of_feature_list[i])], label=int(_type_of_feature_list[i]), alpha = 0)
            ax.text(x_low_dim[i, 0], 
                    x_low_dim[i, 1], 
                    str(int(_type_of_feature_list[i])), 
                    c=hex_c,
                    fontdict={'weight': 'bold', 'size': 9},
                    alpha = 0.5)
        fig.savefig(os.path.join(args.result_path, '../cluster_eval', '{}_anchor_type_{}.png'.format(args.subset, mode)), dpi=600)

    plt.figure(figsize=(8, 8))

    # 按照聚类结果的颜色可视化anchor features
    # 先画anchor features
    for c, lab in zip(anchor_color, range(len(anchor_color))):
        if (cluster_model.labels_==lab).any():
            hex_c = '#%02x%02x%02x' % (c[2], c[1], c[0])
            plt.scatter(x_low_dim[cluster_model.labels_==lab, 0], x_low_dim[cluster_model.labels_==lab, 1], c=hex_c, marker=anchor_marker[lab], label=lab, alpha = 0.5)
    plt.legend(bbox_to_anchor=(-0.15, 1), loc='upper left')
    # args.kmeans 其实是聚类数量
    plt.savefig(os.path.join(args.result_path, '../cluster_eval', '{}_{}_{}_{}.png'.format(args.subset, cluster_method, args.kmeans, mode)), dpi=600)
    plt.cla()
    plt.close("all")

if __name__ == "__main__":           

    # parse argument
    args = parse_option()
    # 读入标注数据
    anchor_dict, anchor_list = load_anno(args)
    # set the data loader (n_data: dataset size)
    data_loader, n_data, sub_dataset = get_data_loader(args, subset=args.subset)
    # load model
    model = set_model(args)
    # 计算所有anchor patch对应的特征向量
    anchor_feature_list, type_of_anchor_feature_list = calcAllFeature(args, model, data_loader, n_data, recalc_feature=True)

    cluster_method = "kmeans" # "kmeans" or "kmedoids" or "Agglomerative"
    # 枚举聚类数量
    max_k = args.kmeans
    for K in range(1, max_k+1):
        args.kmeans = K
        if cluster_method == "kmeans":
            # K-means聚类
            if (os.path.isfile(os.path.join(args.result_path, "kmeans{}.pkl".format(args.kmeans))) and False):
                cluster_model = joblib.load(os.path.join(args.result_path, "kmeans{}.pkl".format(args.kmeans)))
            else:
                cluster_model = KMeans(n_clusters=args.kmeans).fit(anchor_feature_list)
                joblib.dump(cluster_model, os.path.join(args.result_path, "kmeans{}.pkl".format(args.kmeans)))
            print("K-means {} cluster over!".format(args.kmeans))
        elif cluster_method == "kmedoids":
            # K-medoids聚类
            if (os.path.isfile(os.path.join(args.result_path, "kmedoids{}.pkl".format(args.kmeans))) and False):
                cluster_model = joblib.load(os.path.join(args.result_path, "kmedoids{}.pkl".format(args.kmeans)))
            else:
                cluster_model = KMedoids(n_clusters=args.kmeans, metric="cosine").fit(anchor_feature_list)
                joblib.dump(cluster_model, os.path.join(args.result_path, "kmedoids{}.pkl".format(args.kmeans)))
            print("K-Medoids {} cluster over!".format(args.kmeans))
        elif cluster_method == "Agglomerative":       
            # 层次聚类
            cluster_model = AgglomerativeClustering(n_clusters=K)
            cluster_model = cluster_model.fit(anchor_feature_list)
            print("AgglomerativeClustering {} cluster over!".format(args.kmeans))

        ''' 评价聚类结果 '''
        # 1. 计算聚类结果和锚点标注的吻合度
        cluster_precision = evalClusterResult(args, cluster_model, cluster_method, anchor_dict)
        # 2. 计算不确定性（entropy）
        # evalUncertaintyOfCluster(args, cluster_model, anchor_feature_list)
        # 3. 把锚点分类结果可视化画出来(先降维，再根据聚类结果画颜色)
        vis_clustered_feature(args, cluster_method, cluster_model, anchor_feature_list, _type_of_feature_list=type_of_anchor_feature_list, mode='t-sne')
