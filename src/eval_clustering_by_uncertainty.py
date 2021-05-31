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
import matplotlib as mpl
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
    parser.add_argument('--anchor_lab_mask', type=str, default='', help='需要屏蔽的anchor标签')

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
    
    opt.anchor_lab_mask = [int(item) for item in opt.anchor_lab_mask.split(',')]
    print("\nmasked anchor label: {}\n".format(opt.anchor_lab_mask))

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
        for _ac_id in range(len(anchor_dict[_f_id])-1, -1, -1):
            # 如果有anchor_lab_mask，就要屏蔽一部分锚点标注
            if anchor_dict[_f_id][_ac_id][2] in args.anchor_lab_mask:
                del anchor_dict[_f_id][_ac_id]
        # 判断一下，如果_f_id帧只剩下一种类型的锚点或者没有锚点，就丢弃这一帧
        if len(anchor_dict[_f_id]) == 0:
            print('Discard frame {} for empty anchor dict\n'.format(_f_id))
            del anchor_dict[_f_id]
            continue
        if min(np.array(anchor_dict[_f_id])[:,2]) == max(np.array(anchor_dict[_f_id])[:,2]):
            print('Discard frame {} which only has one type anchors {}\n'.format(_f_id, anchor_dict[_f_id][0][2]))
            del anchor_dict[_f_id]
            continue
        for _ac_id in range(len(anchor_dict[_f_id])-1, -1, -1):
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

def to_percent(y, position):
    # 把y轴转化为百分比
    return str(100 * y) + '%'

def evalUncertaintyOfCluster(args, cluster_method, cluster_model, anchor_feature_list, type_of_anchor_feature_list, x_low_dim):
    '''根据不确定性评估聚类的结果, 并绘制所有样本不确定性的直方图'''
    entropy_list_cosine = []
    entropy_list_euc = []
    entropy_list_rbf = []
    risk_coverage = []  # [max_rbf_prob, pseudo_rbf_prob(到每个聚类中心的距离)]
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
        
        entropy_list_rbf.append(1-pseudo_rbf_prob.max())    # 其实没有计算熵，而是取1-最大值，这样离聚类中心越近，值越小
        risk_coverage.append([1-pseudo_rbf_prob.max(), cluster_model.labels_[i], type_of_anchor_feature_list[i] ])

        pseudo_euc_prob = my_softmax(pseudo_euc_prob)
        entropy_list_euc.append(my_entropy(pseudo_euc_prob))

    print("cosine_entropy:", np.array(entropy_list_cosine).mean())
    print("rbf_entropy:", np.array(entropy_list_rbf).mean())
    print("euclidean_entropy:", np.array(entropy_list_euc).mean())

    ''' 按类别统计锚点的entropy(uncertainty)分布折线图'''
    hex_c = []
    for c in anchor_color: hex_c.append('#%02x%02x%02x' % (c[2], c[1], c[0]))
    fig, ax = plt.subplots(figsize=(8, 6))
    num_bins = 10
    min_unc = np.array(entropy_list_euc).min()
    max_unc = 2.5 #np.array(entropy_list_euc).max()
    x_axis = np.arange(min_unc, max_unc, (max_unc-min_unc)/num_bins)
    # 各类别锚点的不确定性：在不同区间不确定性的集合中的占比(以直方图x轴区间为单位进行统计)
    stack_uncertainty = np.zeros((int(type_of_anchor_feature_list.max()+1), num_bins))
    # 各类别锚点的不确定性：在本类别中的占比
    accumulate_uncertainty_rate = np.zeros((int(type_of_anchor_feature_list.max()+1), num_bins))
    for i in range(len(type_of_anchor_feature_list)):
        # 遍历所有uncertainty的值，统计各区间段的分布
        val = entropy_list_euc[i]
        anchor_type = int(type_of_anchor_feature_list[i])
        _block = min(int((val - min_unc) / (max_unc - min_unc + 1e-5) * num_bins), num_bins-1)
        stack_uncertainty[anchor_type][_block] += 1
    # 绘制曲线图
    for i in range(stack_uncertainty.shape[0]):
        for j in range(stack_uncertainty.shape[1]):
            accumulate_uncertainty_rate[i][j] = np.sum(stack_uncertainty[i][:j]) / np.sum(stack_uncertainty[i])
        ax.plot(x_axis, accumulate_uncertainty_rate[i], color=hex_c[i], label=anchor_label[i])
    ax.legend(bbox_to_anchor=(-0.2, 1), loc='upper left')
    ax.set_xlim([0, 2.5])
    ax.set_title("Euclidean entropy of different anchor types")
    ax.set_xlabel("euclidean entropy")
    ax.set_ylabel("percent")
    fig.savefig(os.path.join(args.result_path.replace("cluster_results", "uncertainty_hist"), 'anchor_euclidean_entropy_by_class__{}_{}.png'.format(cluster_method, args.kmeans)), dpi=600)
    plt.cla()
    plt.close("all")

    ''' 所有样本整体的entropy分布直方图 (选用RBF distance)'''
    # fig, ax = plt.subplots(figsize=(8, 6))
    # ax.hist(entropy_list_euc, bins=50)
    # ax.set_xlim([0, 2.5])
    # ax.set_title("Euclidean entropy distribution of all anchors")
    # ax.set_xlabel("euclidean entropy")
    # ax.set_ylabel("anchor samples")
    # fig.savefig(os.path.join(args.result_path.replace("cluster_results", "uncertainty_hist"), 'anchor_euclidean_entropy__{}_{}.png'.format(cluster_method, args.kmeans)), dpi=600)
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.hist(entropy_list_rbf, bins=50)
    ax1.set_title("RBF entropy distribution of all anchors")
    ax1.set_xlabel("RBF entropy")
    ax1.set_ylabel("anchor samples")
    fig1.savefig(os.path.join(args.result_path.replace("cluster_results", "uncertainty_hist"), 'anchor_RBF_entropy__{}_{}.png'.format(cluster_method, args.kmeans)), dpi=600)
    plt.cla()
    plt.close("all")

    ''' 计算融合Loss曲线（所有样本的entropy+聚类数量惩罚，聚类数量太多要惩罚）'''
    prob_cluster = np.zeros(args.kmeans)    # 每个类别出现的概率（各类别数量占比）
    for k in range(args.kmeans):
        prob_cluster[k] = np.count_nonzero(cluster_model.labels_ == k) / cluster_model.labels_.shape[0]
    loss_cluster = my_entropy(prob_cluster)
    print("loss_cluster:", loss_cluster)

    ''' 画出Risk_Coverage曲线，Risk使用到聚类中心的RBF_Prob计算'''
    # risk_coverage: [1-RBF, 聚类类别, anchor类别]
    risk_coverage = sorted(risk_coverage, key=lambda x:x[0])    # 按照response排序，逐步增加coverage并绘risk_coverage图
    risk_coverage_curve = []
    risk_coverage_curve_of_class = []
    cur_risk_coverage_of_class = [0]*9 # int(type_of_anchor_feature_list.max()+1)
    num_coverage_curve_of_class = []    # 统计样本数量随coverage的变化
    cur_num_coverage_of_class = [1e-6]*15
    cluster_risk_coverage_curve_of_class = []    # 聚类类别下的risk分布
    cur_cluster_risk_coverage_of_class = [0]*15
    cur_coverage_step = 1e-10
    tot_risk = 0
    for i in range(len(risk_coverage)):
        cur_coverage = (i+1) / len(risk_coverage)
        if cur_coverage >= cur_coverage_step:
            risk_coverage_curve.append((tot_risk / len(risk_coverage)) / cur_coverage)
            risk_coverage_curve_of_class.append(np.array(cur_risk_coverage_of_class) / len(risk_coverage) / cur_coverage)
            num_coverage_curve_of_class.append(np.array(cur_num_coverage_of_class))
            cluster_risk_coverage_curve_of_class.append(np.array(cur_cluster_risk_coverage_of_class) / len(risk_coverage) / cur_coverage)
            cur_coverage_step += 1.0 / coverage_bins
        # np.eye构造one-hot向量
        # cluster_lab = risk_coverage[i][1]
        # tot_risk += np.linalg.norm(risk_coverage[i][0] - np.eye(args.kmeans)[cluster_lab])
        tot_risk += np.linalg.norm(risk_coverage[i][0])
        cur_risk_coverage_of_class[int(risk_coverage[i][2])] += np.linalg.norm(risk_coverage[i][0])
        cur_num_coverage_of_class[int(risk_coverage[i][1])] += 1
        cur_cluster_risk_coverage_of_class[int(risk_coverage[i][1])] += np.linalg.norm(risk_coverage[i][0])

    # 所有聚类数量的曲线放到一张图上画
    risk_coverage_curve_of_all_K.append(risk_coverage_curve)
    # 绘制当前聚类数量下，各个类别的曲线(ax0绘制risk曲线，ax1绘制样本数量曲线)
    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    x_axis = np.arange(0, 1, 1.0 / coverage_bins)
    ax[0][0].plot(x_axis, risk_coverage_curve, c='k', linewidth=3)
    risk_coverage_curve_of_class = np.array(risk_coverage_curve_of_class)
    ax[0][0].stackplot(x_axis, risk_coverage_curve_of_class[:,0], risk_coverage_curve_of_class[:,1], risk_coverage_curve_of_class[:,2], risk_coverage_curve_of_class[:,3],
                         risk_coverage_curve_of_class[:,4], risk_coverage_curve_of_class[:,5], risk_coverage_curve_of_class[:,6], risk_coverage_curve_of_class[:,7],
                         risk_coverage_curve_of_class[:,8], colors=hex_c[:9], labels=anchor_label[:9], alpha=0.8)
    ax[0][0].legend()
    ax[0][0].set_xlabel("Coverage")
    ax[0][0].set_ylabel("Risk coverage curve")
    cluster_risk_coverage_curve_of_class = np.array(cluster_risk_coverage_curve_of_class)
    ax[0][1].stackplot(x_axis, cluster_risk_coverage_curve_of_class[:,0], cluster_risk_coverage_curve_of_class[:,1], cluster_risk_coverage_curve_of_class[:,2], cluster_risk_coverage_curve_of_class[:,3],
                         cluster_risk_coverage_curve_of_class[:,4], cluster_risk_coverage_curve_of_class[:,5], cluster_risk_coverage_curve_of_class[:,6], cluster_risk_coverage_curve_of_class[:,7],
                         cluster_risk_coverage_curve_of_class[:,8], cluster_risk_coverage_curve_of_class[:,9], cluster_risk_coverage_curve_of_class[:,10], cluster_risk_coverage_curve_of_class[:,11], 
                         cluster_risk_coverage_curve_of_class[:,12], cluster_risk_coverage_curve_of_class[:,13], alpha=0.65)
    # ax[1].legend()
    ax[0][1].set_xlabel("Coverage")
    ax[0][1].set_ylabel("Clustered risk coverage curve")
    num_coverage_curve_of_class = np.array(num_coverage_curve_of_class)
    ax[0][2].stackplot(x_axis, num_coverage_curve_of_class[:,0], num_coverage_curve_of_class[:,1], num_coverage_curve_of_class[:,2], num_coverage_curve_of_class[:,3],
                         num_coverage_curve_of_class[:,4], num_coverage_curve_of_class[:,5], num_coverage_curve_of_class[:,6], num_coverage_curve_of_class[:,7],
                         num_coverage_curve_of_class[:,8], num_coverage_curve_of_class[:,9], num_coverage_curve_of_class[:,10], num_coverage_curve_of_class[:,11], 
                         num_coverage_curve_of_class[:,12], num_coverage_curve_of_class[:,13], alpha=0.65)
    # ax[1].legend()
    ax[0][2].set_xlabel("Coverage")
    ax[0][2].set_ylabel("Sample number & coverage curve")
    # 绘制各聚类类别Risk分布曲线
    weighted_risk = np.zeros((args.kmeans,2))
    for i in range(args.kmeans):
        cluster_risk_percent_curve_of_class = cluster_risk_coverage_curve_of_class[:,i] / num_coverage_curve_of_class[:,i] * len(risk_coverage)
        s_mask = (cluster_risk_percent_curve_of_class != 0) # risk为0的点不画
        ax[1][1].plot(x_axis[s_mask], cluster_risk_percent_curve_of_class[s_mask], label=i, alpha=0.65)
        # 计算risk曲线的均值点
        mean_point = 0
        sum_risk = np.sum(cluster_risk_percent_curve_of_class)
        for j in range(cluster_risk_percent_curve_of_class.shape[0]):
           mean_point += j * (cluster_risk_percent_curve_of_class[j] / sum_risk)
        weighted_risk[i, :] = mean_point, cluster_risk_percent_curve_of_class[int(mean_point)]
        ax[1][1].scatter((mean_point/100), weighted_risk[i][1], marker='*', s=40)
    ax[1][1].legend(ncol=4)
    ax[1][1].set_xlabel("Coverage")
    ax[1][1].set_ylabel("Clustered risk coverage curve")
    # 绘制降维后的锚点(分别用锚点类别真值和聚类类别可视化)
    for i in range(int(type_of_anchor_feature_list.max()+1)):
        ax[1][0].scatter(x_low_dim[type_of_anchor_feature_list == i, 0], x_low_dim[type_of_anchor_feature_list == i, 1], color=hex_c[i], marker=anchor_marker[i], alpha=0.2)
    for i in range(args.kmeans):
        ax[1][2].scatter(x_low_dim[cluster_model.labels_ == i, 0], x_low_dim[cluster_model.labels_ == i, 1], marker=anchor_marker[i], alpha=0.2)
    # 把数据保存到csv中
    with open(os.path.join(args.result_path.replace("cluster_results", "risk_coverage"), 'risk_coverage_curve_of_all_K.csv'.format(cluster_method, args.kmeans)), 'a+') as f:
        f.write(u"\n聚类数量,{}\n".format(args.kmeans))
        # 保存各聚类类别的样本占比
        f.write(u"各类样本数量,")
        for i in range(args.kmeans):
            f.write("{},".format(num_coverage_curve_of_class[-1, i]))
        f.write('\n')
        # 保存risk数据
        # for i in range(args.kmeans):
        #     sum_risk = np.sum(cluster_risk_coverage_curve_of_class[:, i])
        #     f.write("cluster {},".format(i))
        #     for j in range(cluster_risk_coverage_curve_of_class.shape[0]):
        #         f.write("{},".format(cluster_risk_coverage_curve_of_class[j, i]))
        #         weighted_risk[i] += (cluster_risk_coverage_curve_of_class[j, i] / sum_risk) * j
        #     f.write('\n')
        f.write(u"均值点coverage,")
        for i in range(args.kmeans):
            f.write("{},".format(weighted_risk[i][0]))
        f.write("\n")
        f.write(u"均值点Risk,")
        for i in range(args.kmeans):
            f.write("{},".format(weighted_risk[i][1]))
        f.write("\n")
        f.write(u"终点Risk占比,")
        for i in range(args.kmeans):
            f.write("{},".format(cluster_risk_coverage_curve_of_class[-1,i]/np.sum(cluster_risk_coverage_curve_of_class[-1,:])))
        f.write("\n")
    fig.savefig(os.path.join(args.result_path.replace("cluster_results", "risk_coverage"), 'risk_coverage_curve__{}_{}.png'.format(cluster_method, args.kmeans)), dpi=600)
    plt.cla()
    plt.close("all")
    

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
    return x_low_dim

def predict_patch(args, data_loader, cluster_method, cluster_model, cluster_precision):
    ''' 
        可视化聚类后的锚点，并将它们绘制在图像上 (同时绘制锚点标注的结果、聚类标签的结果和聚类吻合度)
    '''
    anchor_width = 64 
    last_frame_id = -1
    dir_name = "clustered_anchor_on_image/{}_{}_{}".format(args.subset, cluster_method, args.kmeans)
    if not os.path.isdir(args.result_path.replace("cluster_results", dir_name)):
        os.makedirs(args.result_path.replace("cluster_results", dir_name))
    for idx, (anchor, _, _, frame_id, _full_img, _anchor_xy, _, _, anchor_type) in enumerate(data_loader):
        # 需要将同一frame图片上的若干锚点都画上后，再保存图片；如果frame_id没变化，则不刷新图片full_img
        if last_frame_id != frame_id.numpy()[0]:
            if last_frame_id != -1:
                cv2.imwrite(os.path.join(args.result_path.replace("cluster_results", dir_name), str(last_frame_id)+".png"), np.concatenate((full_img, full_img_anchor), axis=1))
            full_img = _full_img[0,:,:,:3].numpy().astype(np.uint8)
            full_img_anchor = _full_img[0,:,:,:3].numpy().astype(np.uint8)
            last_frame_id = frame_id.numpy()[0]

        assert(full_img.any())
        anchor_xy = _anchor_xy[0].numpy()
        # 绘制anchor
        for _p, _anchor_t in zip(anchor_xy, anchor_type.cpu().numpy()):
            p_left_top, p_right_down = getPatchXY(full_img, _p[0], _p[1], anchor_width)
            # patch对应的聚类类别
            p_label = cluster_model.labels_[idx]
            # 按照聚类标签绘制patch
            cv2.rectangle(full_img, tuple(p_left_top), tuple(p_right_down), anchor_color[p_label], thickness=4)
            cv2.putText(full_img, "Label {:d}".format(p_label), tuple(p_right_down), cv2.FONT_HERSHEY_SIMPLEX, 1.5, anchor_color[p_label], 2)
            # 按照anchor type绘制patch
            cv2.rectangle(full_img_anchor, tuple(p_left_top), tuple(p_right_down), anchor_color[_anchor_t], thickness=4)
            # cv2.putText(full_img_anchor, "{:d}".format(_anchor_t), tuple(p_right_down), cv2.FONT_HERSHEY_SIMPLEX, 1.5, anchor_color[_anchor_t], 2)
        # 将当前帧对应的聚类标签吻合度画到图上
        cv2.putText(full_img, "Prec: {:.4f}".format(cluster_precision[frame_id.numpy()[0]]), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
        # print info
        if (idx + 1) % args.print_freq == 0:
            print('Save clustered anchors: [{0}/{1}]'.format(idx + 1, len(data_loader)))
    # 保存最后一张图
    cv2.imwrite(os.path.join(args.result_path.replace("cluster_results", dir_name), str(frame_id.numpy()[0])+".png"), np.concatenate((full_img, full_img_anchor), axis=1))

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
    anchor_feature_list, type_of_anchor_feature_list = calcAllFeature(args, model, data_loader, n_data, recalc_feature=False)

    risk_coverage_curve_of_all_K = []
    cluster_method = "kmeans" # "kmeans" or "kmedoids" or "Agglomerative"
    # 枚举聚类数量
    max_k = args.kmeans
    for K in range(3, max_k+1):
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
            if (os.path.isfile(os.path.join(args.result_path, "kmedoids{}.pkl".format(args.kmeans)))):
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

        # 2. 把锚点分类结果可视化画出来(先降维，再根据聚类结果画颜色)
        x_low_dim = vis_clustered_feature(args, cluster_method, cluster_model, anchor_feature_list, _type_of_feature_list=type_of_anchor_feature_list, mode='t-sne')
        print()
        
        # 3. 计算聚类结果整体的不确定性（entropy）
        if args.subset == "train_fine_anno":
            evalUncertaintyOfCluster(args, cluster_method, cluster_model, anchor_feature_list, type_of_anchor_feature_list, x_low_dim)

        # 4. 可视化聚类后的锚点，并将它们绘制在图像上
        # predict_patch(args, data_loader, cluster_method, cluster_model, cluster_precision)
    
    ''' 画出所有聚类情况下的risk_coverage_curve '''
    fig, ax = plt.subplots()
    x_axis = np.arange(0, 1, 1.0 / coverage_bins)
    for i in range(len(risk_coverage_curve_of_all_K)):
        ax.plot(x_axis, risk_coverage_curve_of_all_K[i], label=i+3) # 聚类从3个开始
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk coverage curve")
    ax.set_title("Risk coverage curve of different clustering numbers")
    ax.legend()
    fig.savefig(os.path.join(args.result_path.replace("cluster_results", "risk_coverage"), 'risk_coverage_curve_of_all_K.png'), dpi=600)
    plt.cla()
    plt.close("all")