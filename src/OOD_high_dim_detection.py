'''
    使用人工标注的锚点构建高斯过程，并计算每个测试样本的不确定性（方差）
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
from sklearn.decomposition import PCA,KernelPCA
from sklearn import manifold
import time
from torchvision import transforms
from models.alexnet import MyAlexNetCMC
from offroad_dataset import OffRoadDataset
import joblib
import argparse
import myGaussianProcess

# subset = "train_with_label"
# data_folder = "/home/gaobiao/Documents/offroad_semantic_cluster/data"
# result_folder = "/home/gaobiao/Documents/offroad_semantic_cluster/src/results/0324_OOD_train_to_all/train"
pred_res = 25    # 分类的分辨率：每个patch中间pred_res*pred_res的方块赋予该patch的类别标签
anchor_color = [(0,0,255), (0,255,0), (255,0,0), (0,255,255), (255,0,255), (255,255,0), (220,220,220), (31,102,156), (80,127,255), (140,230,240), (127,255,0), (158,168,3), (255,144,30), (214,112,218)]
anchor_marker = ['.','.','.','.','.','.','x','x','s','s','s','s','*','*']
case_study_frame_id = 200

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

    opt = parser.parse_args()
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
    argsDict = args_to_save.__dict__
    with open(os.path.join(opt.result_path.replace("cluster_results", ""), 'args.txt'), 'w') as f:
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')

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

def vis_anchors():
    '''
    将人工标注的anchor在聚类结果上可视化出来，观察聚类中心和OOD样本的分布
    '''

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

def load_anchor_feature(feature_path, anchor_dict):
    '''
    读入之前保存的图片上有标注的anchor对应patch的feature
    '''
    file_names = os.listdir(feature_path)
    anchor_feature_list = []    # 锚点标注的feature
    type_of_anchor_feature_list = []   # 锚点标注的feature对应的锚点类别
    _half_pred_res = pred_res // 2
    file_names.sort()

    for file_name in file_names:
        # feature_of_one_img (1672, 130), 第一维度是一张图片上patch的数量，第二维度是patch坐标和对应的128维feature [x,y,feature]
        feature_of_one_img = np.load(os.path.join(feature_path, file_name), allow_pickle=True)

        # 找一找有没有和它匹配的anchor patch
        _frame_id = int(file_name.split('.')[0])
        if anchor_dict.get(_frame_id, -1) != -1:
            # 在对应帧的anchor patch中找到和它们对应的feature
            _curr_frm_anchor_list = anchor_dict[_frame_id]
            for _anchor_id in range(len(_curr_frm_anchor_list)):
                a_x, a_y, a_type = anchor_dict[_frame_id][_anchor_id]    # [anchor_x, anchor_y, anchor_type]
                for _item in feature_of_one_img:
                    f_y, f_x = _item[0:2]
                    # 如果锚点中心和feature patch的中心在一定范围内，则认为它们是匹配的
                    if abs(a_x - f_x) <= _half_pred_res and abs(a_y - f_y) <= _half_pred_res:
                        anchor_feature_list.append(_item[2:])   # 128 dim feature
                        type_of_anchor_feature_list.append(a_type)
                        break
            print(_frame_id)
        # if len(anchor_feature_list) > 300:    # DEBUG
            # break

    return np.array(anchor_feature_list), np.array(type_of_anchor_feature_list)

def label_special_features(args, k_means_model, mdis_to_center, feature_list_all):
    ''' 标记部分样本（例如：到聚类中心超过平均距离的样本点）'''
    _feature_flag = np.zeros(feature_list_all.shape[0])
    _kmeans_label = k_means_model.predict(feature_list_all)
    for i in range(feature_list_all.shape[0]):
        _dis = np.linalg.norm(feature_list_all[i] - k_means_model.cluster_centers_[_kmeans_label[i]])
        if _dis > mdis_to_center[_kmeans_label[i]]:
            _feature_flag[i] = 1
    return _feature_flag

def vis_features(args, k_means_model, mdis_to_center, _feature_list, type_of_feature_list, mode='pca', features_for_case_study=None, features_type_for_case_study=None):
    '''
        将输入的特征进行降维可视化
        【注意】：降维的时候应该只用锚点特征学习降维模型，否则会受到case stduy特征的影响，而t-sne做不到这一点，只能所有特征放进去一起降维
        _feature_list, type_of_feature_list: 锚点的特征和类别
    '''
    
    print("feature list size: {}".format(_feature_list.shape))
    _feature_list = np.concatenate((k_means_model.cluster_centers_, _feature_list))
    type_of_feature_list = np.concatenate((np.arange(0, args.kmeans), type_of_feature_list))

    # 如果有case study的视频帧,其特征要和锚点的特征放在一起降维可视化
    if features_for_case_study is not None:
        # features_for_case_study 每行是 x, y, features   1+1+128=130
        feature_list_all = np.concatenate((_feature_list, features_for_case_study[:,2:]))
    else:
        feature_list_all = _feature_list
    
    feature_flag = np.zeros(_feature_list.shape[0])
    # 标记部分样本（例如：到聚类中心超过平均距离的样本点）
    # feature_flag = label_special_features(args, k_means_model, mdis_to_center, _feature_list)

    if mode == 'pca':
        # Kernel PCA降成2维特征后，可视化类别簇
        pca = PCA(n_components=2)
        # pca = KernelPCA(n_components=2, kernel='rbf')
        x_pca = pca.fit(_feature_list).transform(feature_list_all)
        # print("PCA explained_variance_ratio : {}".format(pca.explained_variance_ratio_))

        plt.figure(figsize=(16, 8))
        x_pca_anchors = x_pca[:_feature_list.shape[0]]
        if features_for_case_study is not None:
            x_pca_case_study = x_pca[_feature_list.shape[0]:]
            plt.subplot(1,2,1)

        # 按照聚类结果的颜色可视化anchor features
        # 先画anchor features
        x_pca_anchors_without_centers = x_pca_anchors[args.kmeans:_feature_list.shape[0]]
        for c, lab in zip(anchor_color, range(len(anchor_color))):
            if (k_means_model.labels_==lab).any():
                hex_c = '#%02x%02x%02x' % (c[2], c[1], c[0])
                plt.scatter(x_pca_anchors_without_centers[k_means_model.labels_==lab, 0], x_pca_anchors_without_centers[k_means_model.labels_==lab, 1], c=hex_c, marker=anchor_marker[lab], label=lab, alpha = 0.3)
                plt.scatter(x_pca_anchors_without_centers[(k_means_model.labels_==lab) & (feature_flag[args.kmeans:_feature_list.shape[0]]==1), 0], 
                            x_pca_anchors_without_centers[(k_means_model.labels_==lab) & (feature_flag[args.kmeans:_feature_list.shape[0]]==1), 1], 
                            c='k', marker=anchor_marker[lab], label=lab, alpha = 0.3)
                # 画出聚类簇的中心
                if args.show_kmeans_center:
                    plt.scatter(x_pca_anchors[lab, 0], x_pca_anchors[lab, 1], c=hex_c, marker=anchor_marker[lab], label=lab, alpha = 1, s=200, edgecolors='k')                
        plt.legend(bbox_to_anchor=(-0.15, 1), loc='upper left')

        if features_for_case_study is not None:
            plt.subplot(1,2,2)
            # 按照聚类结果的颜色可视化anchor features
            # 先画anchor features
            for c, lab in zip(anchor_color, range(len(anchor_color))):
                if (type_of_feature_list==lab).any():
                    hex_c = '#%02x%02x%02x' % (c[2], c[1], c[0])
                    plt.scatter(x_pca_anchors[type_of_feature_list==lab, 0], x_pca_anchors[type_of_feature_list==lab, 1], c=hex_c, marker=anchor_marker[lab], label=lab, alpha = 0.3)
            plt.legend()
            # 再画case features
            for i in range(features_type_for_case_study.shape[0]):
                c = anchor_color[features_type_for_case_study[i]]
                hex_c = '#%02x%02x%02x' % (c[2], c[1], c[0])
                plt.text(x_pca_case_study[i, 0], 
                        x_pca_case_study[i, 1], 
                        str(features_type_for_case_study[i]), 
                        c=hex_c,
                        fontdict={'weight': 'bold', 'size': 9},
                        alpha = 0.3)
        plt.legend(bbox_to_anchor=(1.15, 1), loc='upper left')
        plt.savefig(os.path.join(args.result_path, '../', 'cluster_vis_pca.png'), dpi=600)
        
        return x_pca, _feature_list, type_of_feature_list

    elif mode == 't-sne':
        '''
        perpexity	混乱度，表示t-SNE优化过程中考虑邻近点的多少，默认为30，建议取值在5到50之间
        early_exaggeration	表示嵌入空间簇间距的大小，默认为12，该值越大，可视化后的簇间距越大
        '''
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=1, perplexity=30, early_exaggeration=12)
        X_tsne = tsne.fit_transform(feature_list_all)
        # 可视化
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        X_norm_anchors = X_norm[:_feature_list.shape[0]]

        if features_for_case_study is not None:
            X_norm_case_study = X_norm[_feature_list.shape[0]:]
            # plt.subplot(1,2,1)

        X_norm_anchors_without_centers = X_norm[args.kmeans:_feature_list.shape[0]]
        # 按照聚类结果的颜色可视化 anchor features
        for c, lab in zip(anchor_color, range(len(anchor_color))):
            if (k_means_model.labels_==lab).any():
                hex_c = '#%02x%02x%02x' % (c[2], c[1], c[0])                
                ax[0].scatter(X_norm_anchors_without_centers[k_means_model.labels_==lab, 0], X_norm_anchors_without_centers[k_means_model.labels_==lab, 1], c=hex_c, marker=anchor_marker[lab], label=lab, alpha = 0.3)
                ax[0].scatter(X_norm_anchors_without_centers[(k_means_model.labels_==lab) & (feature_flag[args.kmeans:_feature_list.shape[0]]==1), 0], 
                            X_norm_anchors_without_centers[(k_means_model.labels_==lab) & (feature_flag[args.kmeans:_feature_list.shape[0]]==1), 1], 
                            c='k', marker=anchor_marker[lab], label=lab, alpha = 0.3)
                # 画出聚类簇的中心
                if args.show_kmeans_center:
                    ax[0].scatter(X_norm_anchors[lab, 0], X_norm_anchors[lab, 1], c=hex_c, marker=anchor_marker[lab], label=lab, alpha = 1, s=200, edgecolors='k')
        # ax[0].xticks([])
        # ax[0].yticks([])
        ax[0].legend(bbox_to_anchor=(-0.15, 1), loc='upper left')
        
        if features_for_case_study is not None:
            # plt.subplot(1,2,2)
            # 按照聚类结果的颜色可视化 case study(指定帧的features)
            # 先画anchor features
            for c, lab in zip(anchor_color, range(len(anchor_color))):
                if (type_of_feature_list==lab).any():
                    hex_c = '#%02x%02x%02x' % (c[2], c[1], c[0])
                    ax[1].scatter(X_norm_anchors[type_of_feature_list==lab, 0], X_norm_anchors[type_of_feature_list==lab, 1], c=hex_c, marker=anchor_marker[lab], label=lab, alpha = 0.3)
            # 再画case features
            for i in range(features_type_for_case_study.shape[0]):
                c = anchor_color[features_type_for_case_study[i]]
                hex_c = '#%02x%02x%02x' % (c[2], c[1], c[0])
                ax[1].text(X_norm_case_study[i, 0], 
                        X_norm_case_study[i, 1], 
                        str(features_type_for_case_study[i]), 
                        c=hex_c,
                        fontdict={'weight': 'bold', 'size': 9},
                        alpha = 0.3)
        ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        fig.savefig(os.path.join(args.result_path, '../', 'cluster_vis_t-sne.png'), dpi=600)

        return X_norm, _feature_list, type_of_feature_list, 


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

def calc_dis_to_cluster_center(args, k_means_model, anchor_feature_list):
    # 计算所有类的样本到聚类中心的平均距离
    mdis_to_center = np.zeros((args.kmeans))
    # 计算距离之和
    for i in range(anchor_feature_list.shape[0]):
        mdis_to_center[k_means_model.labels_[i]] += np.linalg.norm(anchor_feature_list[i] - k_means_model.cluster_centers_[k_means_model.labels_[i]])
    # 除以每类样本数，得到平均距离
    for i in range(args.kmeans):
        mdis_to_center[i] /= np.count_nonzero(k_means_model.labels_==i)
    return mdis_to_center

def show_uncertainty_img(args, case_study_frame_id, case_study_uncertainty):
    '''
        [处理指定视频帧] 将uncertainty的结果可视化到指定的视频帧上
    '''    
    if not os.path.isdir(args.result_path.replace("cluster_results", "case_study")):
        os.makedirs(args.result_path.replace("cluster_results", "case_study"))
    
    patch_size = 64
    cap = cv2.VideoCapture(args.pre_video)  # 读取待标注数据
    fps=cap.get(cv2.CAP_PROP_FPS)
    video_size=(cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 定位到需要做case study的视频帧case_study_frame_id
    cap.set(cv2.CAP_PROP_POS_FRAMES, case_study_frame_id)
    # 对uncertainty的值进行归一化到0-255
    u_min = 0 #case_study_uncertainty.min()
    u_max = 0.15 #case_study_uncertainty.max()
    print('Uncertainty range of frame{0}: [{1}, {2}]'.format(case_study_frame_id, case_study_uncertainty.min(), case_study_uncertainty.max()))
    fig, ax = plt.subplots()
    ax.set_ylim(0, 1e3)
    ax.hist(case_study_uncertainty, bins=100, range=(0,0.5), log=False)
    fig.savefig(os.path.join(args.result_path.replace("cluster_results", "case_study"), 'frame{}_threshold.png'.format(case_study_frame_id+1)), dpi=600)

    with torch.no_grad():
        ################  读入视频帧  #################
        while True:
            ret, full_img = cap.read() # 读入一帧图像
            if not ret: # 读完整段视频，退出
                print('Video end!')
                break
            frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            _uncertainty_mask = np.zeros((full_img.shape), dtype=np.uint8)
            idx = -1
            # 滑动窗处理所有的patch，先不考虑天空，只滑动下半张图片
            for i in range(full_img.shape[0]//2, full_img.shape[0]-pred_res//2, pred_res):
                for j in range(pred_res//2, full_img.shape[1]-pred_res//2, pred_res):
                    idx += 1
                    p_left_top, p_right_down = getPatchXY(full_img, j, i, patch_size)
                    # 将patch分类得到的类别标签绘制到图像上（只绘制i,j为中心，pred_res*pred_res的方块）
                    u_color = int(min(255, max(0, (case_study_uncertainty[idx] - u_min) / (u_max - u_min) * 255)))
                    _uncertainty_mask = cv2.rectangle(_uncertainty_mask, (j-pred_res//2, i-pred_res//2), (j+pred_res//2,i+pred_res//2), (u_color,u_color,u_color), thickness=-1) #thickness=-1 表示矩形框内颜色填充
            
            _uncertainty_mask = cv2.applyColorMap(_uncertainty_mask, cv2.COLORMAP_WINTER)
            # alpha 为第一张图片的透明度，beta 为第二张图片的透明度 cv2.addWeighted 将原始图片与 mask 融合
            full_img = cv2.addWeighted(full_img, 1, _uncertainty_mask, 0.4, 0)
            cv2.imwrite(os.path.join(args.result_path.replace("cluster_results", "case_study"), str(frame_id)+"_uncertainty.png"), full_img)      
            cv2.imwrite(os.path.join(args.result_path.replace("cluster_results", "case_study"), str(frame_id)+"_uncertainty.mask.png"), _uncertainty_mask)        
            
            print('Save video uncertainty: frame[{0}] {1}'.format(frame_id, os.path.join(args.result_path.replace("cluster_results", "case_study"), str(frame_id)+"_uncertainty.png")))
            # 只处理1帧就退出 DEBUG
            break
    # return features_for_save, np.array(features_type_for_save)[:,0]

def predict_video_frame_OOD(args, model, k_means_model, GPC_model, GPC_likelihood):
    '''
        利用高斯过程模型，计算所有视频帧的不确定性
    '''
    if not os.path.isdir(args.result_path.replace("cluster_results", "case_study")):
        os.makedirs(args.result_path.replace("cluster_results", "case_study"))        
    mean = [0.5200442, 0.5257094, 0.517397]
    std = [0.335111, 0.33463535, 0.33491987]
    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    all_frm_entropy_dict = {}
    all_frm_euclidean_entropy_dict = {}
    patch_size = 64
    # pred_res = 25    # 分类的分辨率：每个patch中间pred_resss*pred_res的方块赋予该patch的类别标签
    cap = cv2.VideoCapture(args.pre_video)  # 读取待标注数据
    if not os.path.isdir(args.result_path.replace("cluster_results", "video_pred")):
        os.makedirs(args.result_path.replace("cluster_results", "video_pred"))
    fps=cap.get(cv2.CAP_PROP_FPS)
    video_size=(cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with torch.no_grad():
        ################  读入视频帧  #################
        while True:
            ret, full_img = cap.read() # 读入一帧图像
            if not ret: # 读完整段视频，退出
                print('Video end!')
                break
            frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            _patch_mask = np.zeros((full_img.shape), dtype=np.uint8)
            batch_cnt = 0   # 将args.batch_pred个patch放入一个batch中再计算特征
            _patch_batch = []
            i_batch = []
            j_batch = []
            features_for_save = np.array([])
            features_type_for_save = []
            # 检查是否有已经计算好的特征文件，如果有就不用重新计算
            if os.path.isfile(os.path.join(args.result_path.replace("cluster_results", "features"), str(frame_id)+".npy")):
                features_for_save = np.load(os.path.join(args.result_path.replace("cluster_results", "features"), str(frame_id)+".npy"))
            # 否则，滑动窗计算视频帧中的patch对应的特征
            else:
                # 滑动窗处理所有的patch，先不考虑天空，只滑动下半张图片
                for i in range(full_img.shape[0]//2, full_img.shape[0]-pred_res//2, pred_res):
                    for j in range(pred_res//2, full_img.shape[1]-pred_res//2, pred_res):
                        p_left_top, p_right_down = getPatchXY(full_img, j, i, patch_size)
                        # 滑动窗得到的小patch
                        _patch = full_img[p_left_top[1]:p_right_down[1], p_left_top[0]:p_right_down[0]]
                        _patch = data_transform(_patch)
                        # 如果channel==6 前背景patch
                        if args.in_channel==6:
                            p_left_top, p_right_down = getPatchXY(full_img, j, i, args.background)
                            _bg_patch = full_img[p_left_top[1]:p_right_down[1], p_left_top[0]:p_right_down[0]]
                            _bg_patch = data_transform(_bg_patch)
                            _patch = torch.cat((_patch, _bg_patch), 0)
                        # 对patch进行transform
                        _patch_batch.append(_patch.cpu().numpy())
                        i_batch.append(i)
                        j_batch.append(j)
                        batch_cnt += 1
                        # 如果凑够了args.batch_pred个patch，则一起计算特征
                        if (batch_cnt % args.batch_pred == 0) or (i+pred_res >= full_img.shape[0]-pred_res//2 and j+pred_res >= full_img.shape[1]-pred_res//2):
                            # if (i+pred_res >= full_img.shape[0]-pred_res//2 and j+pred_res >= full_img.shape[1]-pred_res//2):
                                # print('last batch:{}'.format(batch_cnt))
                            time0 = time.time()
                            # inputs shape --> [batch_size, (1), channel, H, W]
                            inputs = torch.Tensor(_patch_batch)
                            inputs_shape = list(inputs.size())
                            # inputs shape --> [batch_size*(1), channel, H, W]
                            inputs = inputs.view((inputs_shape[0], inputs_shape[1], inputs_shape[2], inputs_shape[3]))
                            inputs = inputs.float()
                            if torch.cuda.is_available():
                                inputs = inputs.cuda()
                            # ===================forward=====================
                            _patch_feature_batch = model(inputs)     # [batch_size*(1), feature_dim]
                            time1 = time.time()
                            # 逐个预测类别并可视化
                            for _patch_feature, _i, _j in zip(_patch_feature_batch, i_batch, j_batch):
                                _patch_label = k_means_model.predict(np.expand_dims(_patch_feature.cpu().numpy().astype(np.float), axis=0))
                                # 将patch分类得到的类别标签绘制到图像上（只绘制i,j为中心，pred_res*pred_res的方块）
                                _patch_mask = cv2.rectangle(_patch_mask, (_j-pred_res,_i-pred_res), (_j+pred_res,_i+pred_res), anchor_color[_patch_label[0]], thickness=-1) #thickness=-1 表示矩形框内颜色填充
                                # 将计算好的patch对应的feature保存到文件里
                                _patch_ij_feature = np.expand_dims(np.concatenate(([_i], [_j], _patch_feature.cpu().numpy())), axis=0)
                                features_type_for_save.append(_patch_label)
                                if features_for_save.shape[0] == 0:
                                    features_for_save = _patch_ij_feature
                                else:
                                    features_for_save = np.concatenate((features_for_save, _patch_ij_feature), axis=0)
                            # 清空上一个batch
                            batch_cnt = 0
                            _patch_batch = []
                            i_batch = []
                            j_batch = []
                            time2 = time.time()
                            print("time cost: {:.3f} + {:.3f}".format(time1-time0, time2-time1))
            
            # 根据计算好的特征，统计该帧视频的uncertainty分布 [注意: features_for_save 共2+128维，前2维是patch对应的坐标，后128维是特征向量]
            # 高斯过程预测特征的均值/方差
            _, pred_means, pred_vars = myGaussianProcess.eval_GP_classification(torch.Tensor(features_for_save[:,2:]), GPC_model, GPC_likelihood)
            
            # 根据到聚类中心的距离，预测patch到每个类别的概率（类似Softmax），然后计算熵
            entropy_list, entropy_list_euclidean = predict_video_frame_entropy(args, model, k_means_model, features_for_save)
            all_frm_entropy_dict[frame_id] = entropy_list
            all_frm_euclidean_entropy_dict[frame_id] = entropy_list_euclidean
            
            # 将均值/方差保存下来
            uncertainty_for_save = {}
            uncertainty_for_save["mean"] = pred_means
            uncertainty_for_save["var"] = pred_vars
          #  np.save(os.path.join(args.result_path.replace("cluster_results", "uncertainty_hist"), str(frame_id)+"_uncertainty.npy"), uncertainty_for_save)
            print("Save uncertainty of frame {}".format(frame_id))
    # 把计算好的entropy保存下来 (分别根据cosine距离和欧几里得距离计算到聚类中心的距离，作为概率值)       
    np.save(os.path.join(args.result_path.replace("cluster_results", "uncertainty_txt"), "all_frame_entropy_dict.npy"), all_frm_entropy_dict)
    np.save(os.path.join(args.result_path.replace("cluster_results", "uncertainty_txt"), "all_frame_entropy_euclidean_dict.npy"), all_frm_euclidean_entropy_dict)

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

def predict_video_frame_entropy(args, model, k_means_model, features_for_save):
    '''
    根据到聚类中心的距离，预测patch到每个类别的概率（类似Softmax），然后计算熵
    [注意: features_for_save 共2+128维，前2维是patch对应的坐标，后128维是特征向量]
    '''
    entropy_list = []
    entropy_list_euclidean = []
    for i in range(features_for_save.shape[0]):
        pseudo_prob = np.zeros(args.kmeans)
        pseudo_euclidean_prob = np.zeros(args.kmeans)
        for _k in range(args.kmeans):
            cluster_center_feat = k_means_model.cluster_centers_[_k]
            # 计算到聚类中心的余弦距离作为概率值
            cos_dis = np.dot(features_for_save[i, 2:], cluster_center_feat.T) / (np.linalg.norm(features_for_save[i, 2:]) * np.linalg.norm(cluster_center_feat))
            # 计算到聚类中心的欧几里得距离作为概率值
            euclidean_dis = np.linalg.norm(features_for_save[i, 2:] - cluster_center_feat)
            pseudo_prob[_k] = cos_dis
            pseudo_euclidean_prob[_k] = euclidean_dis

        # 使用Softmax归一化,并计算熵
        pseudo_prob = my_softmax(pseudo_prob)
        pseudo_euclidean_prob = my_softmax(pseudo_euclidean_prob)
        entropy_list.append(my_entropy(pseudo_prob))
        entropy_list_euclidean.append(my_entropy(pseudo_euclidean_prob))
    return np.array(entropy_list), np.array(entropy_list_euclidean)
 
if __name__ == "__main__":           

    # parse argument
    args = parse_option()

    # set the data loader (n_data: dataset size)
    data_loader, n_data, sub_dataset = get_data_loader(args, subset=args.subset)

    # load model
    model = set_model(args)

    # 计算所有anchor patch对应的特征向量
    anchor_feature_list, type_of_anchor_feature_list = calcAllFeature(args, model, data_loader, n_data, recalc_feature=True)

    cluster_method = "kmeans" # "kmeans" or "kmedoids"
    if cluster_method == "kmeans":
        # K-means聚类
        if (os.path.isfile(os.path.join(args.result_path, "kmeans{}.pkl".format(args.kmeans)))):
            k_means_model = joblib.load(os.path.join(args.result_path, "kmeans{}.pkl".format(args.kmeans)))
        else:
            k_means_model = KMeans(n_clusters=args.kmeans).fit(anchor_feature_list)
            joblib.dump(k_means_model, os.path.join(args.result_path, "kmeans{}.pkl".format(args.kmeans)))
        print("K-means cluster over!")
    else:
        # K-medoids聚类
        if (os.path.isfile(os.path.join(args.result_path, "kmedoids{}.pkl".format(args.kmeans)))):
            k_means_model = joblib.load(os.path.join(args.result_path, "kmedoids{}.pkl".format(args.kmeans)))
        else:
            k_means_model = KMedoids(n_clusters=args.kmeans, metric="cosine").fit(anchor_feature_list)
            joblib.dump(k_means_model, os.path.join(args.result_path, "kmedoids{}.pkl".format(args.kmeans)))
        print("K-Medoids cluster over!")

    # 计算所有类的样本到聚类中心的平均距离
    # mdis_to_center = calc_dis_to_cluster_center(args, k_means_model, anchor_feature_list)
    print("K-means cluster over!")

    '''-----------------------------------------------------------------------------------------------'''

    # 读入标注数据
    anchor_dict, anchor_list = load_anno(args)

    # 读入保存的feature,找到其中与有标注锚点对应的feature
    # feature_folder = os.path.join(args.result_path, "features")
    # anchor_feature_list, type_of_anchor_feature_list = load_anchor_feature(feature_folder, anchor_dict)

    # 计算指定的某一帧的预测结果
    # features_for_save, features_type_for_save = predict_video_frame(args, model, k_means_model)

    # 将锚点对应的feature进行降维可视化, 其中
    # low_dim_pos：[args.k_means个聚类中心点+若干锚点+若干case study特征点]， 后面若干case study特征点暂时不用
    # _feature_list, _type_of_feature_list: [args.k_means个聚类中心点+若干锚点]
    # low_dim_pos, _feature_list, _type_of_feature_list = vis_features(args, k_means_model, mdis_to_center,
    #                                                                 anchor_feature_list, 
    #                                                                 type_of_anchor_feature_list, 
    #                                                                 mode='pca', 
    #                                                                 features_for_case_study=features_for_save, 
    #                                                                 features_type_for_case_study=features_type_for_save)

    # 降维之后的features，用高斯过程建模不确定性
    GPC_model, GPC_likelihood = myGaussianProcess.train_GP_classification(anchor_feature_list, type_of_anchor_feature_list)

    # 根据建模好的高斯过程，计算所有视频帧的不确定性(顺便计算熵)
    predict_video_frame_OOD(args, model, k_means_model, GPC_model, GPC_likelihood)

    # 可视化高斯过程的空间范围
    # xx, yy = np.meshgrid(np.arange(-1.0, 1.01, 0.01), np.arange(-1.0, 1.01, 0.01))
    # test_x_mat, test_y_mat = torch.Tensor(xx), torch.Tensor(yy)
    # test_x = torch.cat((test_x_mat.view(-1,1), test_y_mat.view(-1,1)),dim=1)

    # 预测高斯过程的均值方差
    # _, pred_means, pred_vars = myGaussianProcess.eval_GP_classification(test_x, GPC_model, GPC_likelihood)
    # 可视化高斯过程的结果(case study样本)
    # case_study_uncertainty = myGaussianProcess.visualize(args, test_x_mat, test_y_mat, pred_means, pred_vars, case_study_frame_id, case_study_pos=low_dim_pos[_feature_list.shape[0]:], case_study_type=features_type_for_save)
    # 可视化高斯过程的结果(锚点样本)
    # case_study_uncertainty = myGaussianProcess.visualize(args, test_x_mat, test_y_mat, pred_means, pred_vars, case_study_frame_id, case_study_pos=low_dim_pos[args.kmeans:_feature_list.shape[0]], case_study_type=k_means_model.labels_)

    # 将uncertainty可视化到case study的视频帧上面
    # case_study_uncertainty.shape为所有case study的patch排成一维
    # show_uncertainty_img(args, case_study_frame_id, case_study_uncertainty)
