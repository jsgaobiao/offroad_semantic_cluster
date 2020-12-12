from sklearn.cluster import KMeans
import numpy as np
import torch
import cv2
import os
import math
import sys
from torchvision import transforms, datasets
from models.alexnet import MyAlexNetCMC
from offroad_dataset import OffRoadDataset
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import argparse

def parse_option():
    parser = argparse.ArgumentParser('argument for evaluation')
    parser.add_argument('--data_folder', type=str, default=None, help='path to data')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--subset', type=str, default="train", help='subset for training')
    parser.add_argument('--kmeans', type=int, help='kmeans聚类的类别数量')
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
    parser.add_argument('--in_channel', type=int, default=3, help='dim of input image channel (3: RGB, 5: RGBXY)')

    opt = parser.parse_args()
    # 要保存每个anchor的特征，所以batch_size必须是1
    opt.batch_size = 1

    if (opt.data_folder is None) or (opt.model_path is None) or (opt.result_path is None):
        raise ValueError('one or more of the folders is None: data_folder | model_path | result_path')
    
    if opt.note != None:
        opt.result_path = os.path.join(opt.result_path, opt.note, "cluster_results")
    if not os.path.isdir(opt.result_path):
        os.makedirs(opt.result_path)
    
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
    return data_loader, n_data

def calcAllFeature(args, model, data_loader, n_data):
    ''' 计算所有patch的特征向量，并保存 '''
    # 如果存在已经保存的结果，就载入
    if os.path.isfile(os.path.join(args.result_path, "all_patch_features.npy")):
        all_features = np.load(os.path.join(args.result_path, "all_patch_features.npy"))
    else:
        # 否则，计算所有patch的特征
        all_features = np.zeros((n_data, args.feat_dim))
        with torch.no_grad():
            for idx, (anchor, _, _, frame_id, full_img, anchor_xy, _, _, _) in enumerate(data_loader):
                # anchor shape: [batch_size, 1, channel, H, W] # neg_sample,pos_sample shape: [batch_size, K, channel, H, W]
                batch_size = anchor.size(0)
                # batch_size必须是1，因为要逐个保存每个anchor的特征
                assert(batch_size == 1)
                # inputs shape --> [batch_size, (1), channel, H, W]
                inputs = anchor
                inputs_shape = list(inputs.size())
                # inputs shape --> [batch_size*(1), channel, H, W]
                inputs = inputs.view((inputs_shape[0]*inputs_shape[1], inputs_shape[2], inputs_shape[3], inputs_shape[4]))
                inputs = inputs.float()
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                # ===================forward=====================
                feature = model(inputs)     # [batch_size*(1), feature_dim]
                for _bs in range(batch_size):
                    all_features[idx] = feature[_bs].cpu().numpy()
                # ===============================================
                # print info
                if (idx + 1) % args.print_freq == 0:
                    print('Calculate feature: [{0}/{1}]'.format(idx + 1, len(data_loader)))
            np.save(os.path.join(args.result_path, "all_patch_features.npy"), all_features)
    return all_features

def predict_patch(args, data_loader, k_means_model, cluster_precision):
    ''' 
        可视化聚类后的锚点，并将它们绘制在图像上 (同时绘制锚点标注的结果、聚类标签的结果和聚类吻合度)
    '''
    anchor_width = 64
    anchor_color = [(0,0,255), (0,255,0), (255,0,0), (0,255,255), (255,0,255), (255,255,0), (255, 191, 0), (0, 191, 255), (128, 0, 255)]

    def getPatchXY(_img, _x, _y, anchor_width):
        ''' 从图像img中获取中心坐标为(_x, _y)的patch左上右下角坐标 '''
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

    def getColor(_v, color_map):
        # _v的值在[e^(-1), e]之间， 归一化到0-255
        _dv = int(max(min((_v - math.exp(-1)) / (math.exp(1) - math.exp(-1)) * 255, 255), 0))
        return [int(color_map[_dv][0][0]), int(color_map[_dv][0][1]), int(color_map[_dv][0][2])]

    last_frame_id = -1
    for idx, (anchor, _, _, frame_id, _full_img, _anchor_xy, _, _, anchor_type) in enumerate(data_loader):
        # 需要将同一frame图片上的若干锚点都画上后，再保存图片；如果frame_id没变化，则不刷新图片full_img
        if last_frame_id != frame_id.numpy()[0]:
            if last_frame_id != -1:
                cv2.imwrite(os.path.join(args.result_path, str(last_frame_id)+".png"), np.concatenate((full_img, full_img_anchor), axis=1))
            full_img = _full_img[0,:,:,:3].numpy().astype(np.uint8)
            full_img_anchor = _full_img[0,:,:,:3].numpy().astype(np.uint8)
            last_frame_id = frame_id.numpy()[0]

        assert(full_img.any())
        anchor_xy = _anchor_xy[0].numpy()
        # 绘制anchor
        for _p, _anchor_t in zip(anchor_xy, anchor_type.cpu().numpy()):
            p_left_top, p_right_down = getPatchXY(full_img, _p[0], _p[1], anchor_width)
            # patch对应的聚类类别
            p_label = k_means_model.labels_[idx]
            # 按照聚类标签绘制patch
            cv2.rectangle(full_img, tuple(p_left_top), tuple(p_right_down), anchor_color[p_label], thickness=4)
            cv2.putText(full_img, "{:d}".format(p_label), tuple(p_right_down), cv2.FONT_HERSHEY_SIMPLEX, 1.5, anchor_color[p_label], 2)
            # 按照anchor type绘制patch
            cv2.rectangle(full_img_anchor, tuple(p_left_top), tuple(p_right_down), anchor_color[_anchor_t], thickness=4)
            cv2.putText(full_img_anchor, "{:d}".format(_anchor_t), tuple(p_right_down), cv2.FONT_HERSHEY_SIMPLEX, 1.5, anchor_color[_anchor_t], 2)
        # 将当前帧对应的聚类标签吻合度画到图上
        cv2.putText(full_img, "Prec: {:.4f}".format(cluster_precision[frame_id.numpy()[0]]), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
        # print info
        if (idx + 1) % args.print_freq == 0:
            print('Save clustered anchors: [{0}/{1}]'.format(idx + 1, len(data_loader)))
    # 保存最后一张图
    cv2.imwrite(os.path.join(args.result_path, str(frame_id.numpy()[0])+".png"), np.concatenate((full_img, full_img_anchor), axis=1))

def evalClusterResult(args, n_data, data_loader, k_means_model):
    '''
        计算聚类结果和锚点标注的吻合度:
        每张图上有n个锚点，对应了n*(n-1)对约束（互为正样本/负样本），评价聚类结果能满足多少约束对
    '''
    last_frame_id = -1
    anchor_in_frame = []
    cluster_precision = {}
    cluster_precision_sum = 0
    frame_cnt = 0
    for idx, (anchor, _, _, frame_id, _full_img, _anchor_xy, _, _, anchor_type) in enumerate(data_loader):
        # 需要将同一frame图片上的所有锚点的聚类标签取下来，再计算约束吻合度
        if last_frame_id == -1:
            last_frame_id = frame_id.numpy()[0]
        if last_frame_id != frame_id.numpy()[0]:
            # 计算当前帧中正负样本约束和聚类标签的吻合度
            cnt = 0
            _n = len(anchor_in_frame)
            for i in range(_n):
                for j in range(i):
                    # 锚点i和j互为正样本
                    cnt = cnt + (1 if anchor_in_frame[i] == anchor_in_frame[j] else 0)
                    # 锚点i和j互为负样本
                    cnt = cnt + (1 if (anchor_in_frame[i][0] != anchor_in_frame[j][0]) and (anchor_in_frame[i][1] != anchor_in_frame[j][1]) else 0)
            # 当前帧的吻合度 cnt*2/n(n-1)
            cluster_precision[last_frame_id] = cnt * 2 / (_n * (_n - 1))
            cluster_precision_sum += cnt * 2 / (_n * (_n - 1))
            frame_cnt += 1
            anchor_in_frame = []
            last_frame_id = frame_id.numpy()[0]

        # 这个anchor对应的聚类后标签
        p_label = k_means_model.labels_[idx]
        anchor_in_frame.append([anchor_type.cpu().numpy()[0], p_label])
        # print info
        if (idx + 1) % args.print_freq == 0:
            print('Evaluate clustered anchors: [{0}/{1}]'.format(idx + 1, len(data_loader)))
    
    # 循环结束后，计算最后一帧中正负样本约束和聚类标签的吻合度
    cnt = 0
    _n = len(anchor_in_frame)
    for i in range(_n):
        for j in range(i):
            # 锚点i和j互为正样本
            cnt = cnt + (1 if anchor_in_frame[i] == anchor_in_frame[j] else 0)
            # 锚点i和j互为负样本
            cnt = cnt + (1 if (anchor_in_frame[i][0] != anchor_in_frame[j][0]) and (anchor_in_frame[i][1] != anchor_in_frame[j][1]) else 0)
    # 当前帧的吻合度 cnt*2/n(n-1)
    cluster_precision[frame_id.numpy()[0]] = cnt * 2 / (_n * (_n - 1))
    cluster_precision_sum += cnt * 2 / (_n * (_n - 1))
    frame_cnt += 1

    print("Average cluster precision: {}".format(cluster_precision_sum / frame_cnt))
    return cluster_precision
    

def pcaVisualize(args, all_features, k_means_model):
    '''PCA降成2维特征后，可视化类别簇'''
    pca = PCA(n_components=2)
    x_pca = pca.fit(all_features).transform(all_features)
    print("explained_variance_ratio : {}".format(pca.explained_variance_ratio_))
    ax = plt.figure()
    for c, lab in zip('rgbcmykw', range(args.kmeans)):
        plt.scatter(x_pca[k_means_model.labels_==lab, 0], x_pca[k_means_model.labels_==lab, 1], c=c, label=lab)
    plt.xlabel('Dimension1')
    plt.ylabel('Dimension2')
    plt.title('clusters')
    plt.legend()
    plt.savefig(os.path.join(args.result_path, 'cluster_vis.png'))

def main():# 供直接运行本脚本

    # parse argument
    args = parse_option()

    # set the data loader (n_data: dataset size)
    data_loader, n_data = get_data_loader(args, subset=args.subset)

    # load model
    model = set_model(args)

    # 计算所有patch对应的特征向量
    all_features = calcAllFeature(args, model, data_loader, n_data)

    # K-means聚类
    k_means_model = KMeans(n_clusters=args.kmeans).fit(all_features)
    print("K-means cluster over!")
    
    # 计算聚类结果和锚点标注的吻合度
    cluster_precision = evalClusterResult(args, n_data, data_loader, k_means_model)

    # 预测每个patch的类别并保存可视化结果
    print("Start predicting anchor labels...")
    predict_patch(args, data_loader, k_means_model, cluster_precision)

    # PCA可视化类别簇
    print("Start PCA visualization...")
    pcaVisualize(args, all_features, k_means_model)
        


if __name__ == '__main__':
    main()
