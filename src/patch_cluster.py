'''
Author: Gao Biao
Date: 2020-12-08 14:45:45
LastEditTime: 2021-01-16 22:11:46
Description: 根据对比学习学到的特征距离进行K-means聚类，并且根据K-means结果预测patch的语义类别，绘制到图像上
FilePath: /offroad_semantic_cluster/src/patch_cluster.py
'''

from sklearn.cluster import KMeans
import numpy as np
import torch
import cv2
import os
import math
import time
from torchvision import transforms
from models.alexnet import MyAlexNetCMC
from offroad_dataset import OffRoadDataset
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import joblib
import argparse

def parse_option():
    parser = argparse.ArgumentParser('argument for evaluation')
    parser.add_argument('--data_folder', type=str, default=None, help='path to data')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--subset', type=str, default="train", help='subset for training')
    parser.add_argument('--kmeans', type=int, help='kmeans聚类的类别数量')
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
    ''' 计算所有patch的特征向量，并保存 '''
    # 如果存在已经保存的结果，就载入
    if os.path.isfile(os.path.join(args.result_path, "all_patch_features.npy")) and recalc_feature==False:
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
                # ===============================================
                # print info
                if (idx + 1) % args.print_freq == 0:
                    print('Calculate feature: [{0}/{1}]'.format(idx + 1, len(data_loader)))
            np.save(os.path.join(args.result_path, "all_patch_features.npy"), all_features)
    return all_features

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

def getColor(_v, color_map):
    # _v的值在[e^(-1), e]之间， 归一化到0-255
    _dv = int(max(min((_v - math.exp(-1)) / (math.exp(1) - math.exp(-1)) * 255, 255), 0))
    return [int(color_map[_dv][0][0]), int(color_map[_dv][0][1]), int(color_map[_dv][0][2])]

def predict_patch(args, data_loader, k_means_model, cluster_precision):
    ''' 
        可视化聚类后的锚点，并将它们绘制在图像上 (同时绘制锚点标注的结果、聚类标签的结果和聚类吻合度)
    '''
    anchor_width = 64
    anchor_color = [(0,0,255), (0,255,0), (255,0,0), (0,255,255), (255,0,255), (255,255,0), (255, 191, 0), (0, 191, 255), (128, 0, 255)]
 
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
    cv2.imwrite(os.path.join(args.result_path, str(frame_id.numpy()[0])+".png"), np.concatenate((full_img, full_img_anchor), axis=1))


def predict_all_patch(args, data_loader, model, k_means_model):
    ''' [只处理有标注锚点的帧] 滑动窗选取图像上所有patch，预测每个patch归属的类别 '''
    mean = [0.5200442, 0.5257094, 0.517397]
    std = [0.335111, 0.33463535, 0.33491987]
    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    anchor_color = [(0,0,255), (0,255,0), (255,0,0), (0,255,255), (255,0,255), (255,255,0), (255, 191, 0), (0, 191, 255), (128, 0, 255)]
    last_frame_id = -1
    tot_frame = 0
    with torch.no_grad():
        for idx, (anchor, _, _, frame_id, _full_img, _, _, _, _) in enumerate(data_loader):
            print(idx, frame_id)
            # 来了一帧新的frame，处理上面的所有patch
            if last_frame_id != frame_id.numpy()[0]:                
                full_img = _full_img[0,:,:,:3].numpy().astype(np.uint8)
                _patch_mask = np.zeros((full_img.shape), dtype=np.uint8)
                # 滑动窗处理所有的patch，先不考虑天空，只滑动下半张图片
                patch_size = 64
                pred_res = 64    # 分类的分辨率：每个patch中间pred_res*pred_res的方块赋予该patch的类别标签
               
                for i in range(full_img.shape[0]//2, full_img.shape[0]-pred_res//2, pred_res):
                    for j in range(pred_res//2, full_img.shape[1]-pred_res//2, pred_res):
                        p_left_top, p_right_down = getPatchXY(full_img, j, i, patch_size)
                        # 滑动窗得到的小patch
                        _patch = full_img[p_left_top[1]:p_right_down[1], p_left_top[0]:p_right_down[0]]
                        # 对patch进行transform后计算其特征
                        _patch = data_transform(_patch)
                        # 如果channel==6 前背景patch
                        if args.in_channel==6:
                            p_left_top, p_right_down = getPatchXY(full_img, j, i, args.background)
                            _bg_patch = full_img[p_left_top[1]:p_right_down[1], p_left_top[0]:p_right_down[0]]
                            _bg_patch = data_transform(_bg_patch)
                            _patch = torch.cat((_patch, _bg_patch), 0)
                        # inputs shape --> [batch_size, (1), channel, H, W]
                        inputs = _patch
                        inputs_shape = list(inputs.size())
                        # inputs shape --> [batch_size*(1), channel, H, W]
                        inputs = inputs.view((1, inputs_shape[0], inputs_shape[1], inputs_shape[2]))
                        inputs = inputs.float()
                        if torch.cuda.is_available():
                            inputs = inputs.cuda()
                        # ===================forward=====================
                        _patch_feature = model(inputs)     # [batch_size*(1), feature_dim]
                        _patch_label = k_means_model.predict(_patch_feature.cpu().numpy().astype(np.float))
                        # 将patch分类得到的类别标签绘制到图像上（只绘制i,j为中心，pred_res*pred_res的方块）
                        _patch_mask = cv2.rectangle(_patch_mask, (j-pred_res,i-pred_res), (j+pred_res,i+pred_res), anchor_color[_patch_label[0]], thickness=-1) #thickness=-1 表示矩形框内颜色填充

                # alpha 为第一张图片的透明度，beta 为第二张图片的透明度 cv2.addWeighted 将原始图片与 mask 融合
                full_img = cv2.addWeighted(full_img, 1, _patch_mask, 0.2, 0)
                cv2.imwrite(os.path.join(args.result_path, str(frame_id.numpy()[0])+"_pred_all.png"), full_img)        
                last_frame_id = frame_id.numpy()[0]
                # print info
                tot_frame += 1
                print('Save fine segmentation: [{0}]'.format(tot_frame))
            else:
                continue

def predict_vidoe(args, model, k_means_model):
    '''
        [处理所有视频帧] 预测视频中每一帧的分割结果
    '''
    mean = [0.5200442, 0.5257094, 0.517397]
    std = [0.335111, 0.33463535, 0.33491987]
    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    patch_size = 64
    pred_res = 25    # 分类的分辨率：每个patch中间pred_res*pred_res的方块赋予该patch的类别标签
    anchor_color = [(0,0,255), (0,255,0), (255,0,0), (0,255,255), (255,0,255), (255,255,0), (255, 191, 0), (0, 191, 255), (128, 0, 255)]
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
            # if (frame_id % 5 == 1):
                # continue
            # 如果有结果就不重复计算了
            if os.path.isfile(os.path.join(args.result_path.replace("cluster_results", "video_pred"), str(frame_id)+"_pred_all.mask.png")):
                continue
            _patch_mask = np.zeros((full_img.shape), dtype=np.uint8)
            batch_cnt = 0   # 将args.batch_pred个patch放入一个batch中再计算特征
            _patch_batch = []
            i_batch = []
            j_batch = []
            features_for_save = np.array([])
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
            # alpha 为第一张图片的透明度，beta 为第二张图片的透明度 cv2.addWeighted 将原始图片与 mask 融合
            full_img = cv2.addWeighted(full_img, 1, _patch_mask, 0.2, 0)
            cv2.imwrite(os.path.join(args.result_path.replace("cluster_results", "video_pred"), str(frame_id)+"_pred_all.png"), full_img)      
            cv2.imwrite(os.path.join(args.result_path.replace("cluster_results", "video_pred"), str(frame_id)+"_pred_all.mask.png"), _patch_mask)        
            # 将计算好的patch对应的feature保存到文件里
            print(features_for_save.shape)
            np.save(os.path.join(args.result_path.replace("cluster_results", "features"), str(frame_id)+".npy"), features_for_save)
            # print info
            print('Save video fine segmentation: [{0}] {1}'.format(frame_id, os.path.join(args.result_path.replace("cluster_results", "video_pred"), str(frame_id)+"_pred_all.png")))

def getColor(_v, color_map):
    # _v的值在[e^(-1), e]之间， 归一化到0-255
    _dv = int(max(min((_v - math.exp(-1)) / (math.exp(1) - math.exp(-1)) * 255, 255), 0))
    return [int(color_map[_dv][0][0]), int(color_map[_dv][0][1]), int(color_map[_dv][0][2])]

def eval_dis_2_road(args, data_loader, model, k_means_model):
    '''
        假设镜头前为路面区域，度量其他patch到该区域的特征距离
    '''
    mean = [0.5200442, 0.5257094, 0.517397]
    std = [0.335111, 0.33463535, 0.33491987]
    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    patch_size = 64
    pred_res = 25    # 分类的分辨率：每个patch中间pred_res*pred_res的方块赋予该patch的类别标签
    color_map = cv2.applyColorMap(np.arange(0, 256, dtype=np.uint8), cv2.COLORMAP_JET)
    cap = cv2.VideoCapture(args.pre_video)  # 读取待标注数据
    if not os.path.isdir(args.result_path.replace("cluster_results", "drivable_dis")):
        os.makedirs(args.result_path.replace("cluster_results", "drivable_dis"))
    video_size=(cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with torch.no_grad():
        ################  读入视频帧  #################
        while True:
            ret, full_img = cap.read() # 读入一帧图像
            frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if not ret: # 读完整段视频，退出
                break
            _patch_mask = np.zeros((full_img.shape), dtype=np.uint8)
            
            ############## 首先提取图像前的路面区域 ################
            road_patch = full_img[int(video_size[1]-64):int(video_size[1]), int(video_size[0]//2-32):int(video_size[0]//2+32)]
            # 对patch进行transform后计算其特征
            road_patch = data_transform(road_patch)
            # 如果channel==6 前背景patch
            if args.in_channel==6:
                p_left_top, p_right_down = getPatchXY(full_img, int(video_size[1]-32), int(video_size[0]//2), args.background)
                _bg_patch = full_img[p_left_top[1]:p_right_down[1], p_left_top[0]:p_right_down[0]]
                _bg_patch = data_transform(_bg_patch)
                road_patch = torch.cat((road_patch, _bg_patch), 0)
            # inputs shape --> [batch_size, (1), channel, H, W]
            inputs = road_patch
            inputs_shape = list(inputs.size())
            # inputs shape --> [batch_size*(1), channel, H, W]
            inputs = inputs.view((1, inputs_shape[0], inputs_shape[1], inputs_shape[2]))
            inputs = inputs.float()
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            # ===================forward=====================
            road_patch_feature = model(inputs)     # [batch_size*(1), feature_dim]

            ######################## 滑动窗处理所有的patch计算它们与road patch的距离，先不考虑天空，只滑动下半张图片 ########################
            
            for i in range(full_img.shape[0]//2, full_img.shape[0]-pred_res//2, pred_res):
                for j in range(pred_res//2, full_img.shape[1]-pred_res//2, pred_res):
                    time0 = time.time()
                    p_left_top, p_right_down = getPatchXY(full_img, j, i, patch_size)
                    # 滑动窗得到的小patch
                    _patch = full_img[p_left_top[1]:p_right_down[1], p_left_top[0]:p_right_down[0]]
                    # 对patch进行transform后计算其特征
                    _patch = data_transform(_patch)
                    # 如果channel==6 前背景patch
                    if args.in_channel==6:
                        p_left_top, p_right_down = getPatchXY(full_img, j, i, args.background)
                        _bg_patch = full_img[p_left_top[1]:p_right_down[1], p_left_top[0]:p_right_down[0]]
                        _bg_patch = data_transform(_bg_patch)
                        _patch = torch.cat((road_patch, _bg_patch), 0)
                    # inputs shape --> [batch_size, (1), channel, H, W]
                    inputs = _patch
                    inputs_shape = list(inputs.size())
                    # inputs shape --> [batch_size*(1), channel, H, W]
                    inputs = inputs.view((1, inputs_shape[0], inputs_shape[1], inputs_shape[2]))
                    inputs = inputs.float()
                    if torch.cuda.is_available():
                        inputs = inputs.cuda()
                    # ===================forward=====================
                    _patch_feature = model(inputs)     # [batch_size*(1), feature_dim]
                    time1 = time.time()
                    # _patch_label = k_means_model.predict(_patch_feature.cpu().numpy())
                    f_dis = torch.mm(_patch_feature, road_patch_feature.view(args.feat_dim, -1))    # 范围[-1,1]
                    f_dis = torch.exp(f_dis)
                    # 根据patch与road_patch的特征距离可视化颜色，然后绘制到图像上（只绘制i,j为中心，pred_res*pred_res的方块）
                    _patch_mask = cv2.rectangle(_patch_mask, (j-pred_res,i-pred_res), (j+pred_res,i+pred_res), tuple(getColor(f_dis, color_map)), thickness=-1) #thickness=-1 表示矩形框内颜色填充
                    time2 = time.time()
                    print("time cost: {} / {} / {}".format(time1-time0, time2-time1))
            # alpha 为第一张图片的透明度，beta 为第二张图片的透明度 cv2.addWeighted 将原始图片与 mask 融合
            full_img = cv2.addWeighted(full_img, 1, _patch_mask, 0.4, 0)
            cv2.imwrite(os.path.join(args.result_path.replace("cluster_results", "drivable_dis"), str(frame_id)+"_dis2road.png"), full_img)        
            # print info
            print('Save distance to road patch image: [{0}/{1}]'.format(frame_id, tot_frames))

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
    

def pcaVisualize(args, all_features, k_means_model, sub_dataset=None):
    '''PCA降成2维特征后，可视化类别簇'''
    pca = PCA(n_components=2)
    x_pca = pca.fit(all_features).transform(all_features)
    print("explained_variance_ratio : {}".format(pca.explained_variance_ratio_))
    ax = plt.figure()
    # 按照聚类结果的颜色可视化
    for c, lab in zip('rgbymckw', range(args.kmeans)):
        plt.scatter(x_pca[k_means_model.labels_==lab, 0], x_pca[k_means_model.labels_==lab, 1], c=c, label=lab, alpha = 0.2)
    # 按照训练集和测试集可视化 for DEBUG
    # plt.scatter(x_pca[:,0], x_pca[:,1], c='r', label='train', alpha = 0.25)
    # plt.scatter(x_pca[:973,0], x_pca[:973,1], c='r', label='train', alpha = 0.1)
    # plt.scatter(x_pca[973:,0], x_pca[973:,1], c='g', label='test', alpha = 0.1)

    # case study
    if (sub_dataset):
        case_id = 3335 # case study 的frame id
        al = np.array(sub_dataset.anchor_list)
        for c, lab in zip('rgbymck', range(7)):
            flag_case_id = (al[:,0] == case_id)
            flag_anchor_lab = (al[:,3] == lab)
            flags = [i and j for i,j in zip(flag_case_id, flag_anchor_lab)]+[False]*626
            if (any(flags)):
                plt.scatter(x_pca[flags, 0], x_pca[flags, 1], c=c, label=lab, alpha = 1, marker="*", s=100)
        # for idx, i in enumerate(sub_dataset.anchor_dict[case_id]):
            # plt.scatter(x_pca[idx,0], x_pca[idx,1])

    plt.xlabel('Dimension1')
    plt.ylabel('Dimension2')
    plt.title('clusters')
    plt.legend()
    plt.savefig(os.path.join(args.result_path, 'cluster_vis{}.png'.format(args.kmeans)), dpi=600)

def main():# 供直接运行本脚本

    # parse argument
    args = parse_option()

    # set the data loader (n_data: dataset size)
    data_loader, n_data, sub_dataset = get_data_loader(args, subset=args.subset)

    # load model
    model = set_model(args)

    # 计算所有patch对应的特征向量
    all_features = calcAllFeature(args, model, data_loader, n_data)

    # K-means聚类
    if (os.path.isfile(os.path.join(args.result_path, "kmeans{}.pkl".format(args.kmeans)))):
        k_means_model = joblib.load(os.path.join(args.result_path, "kmeans{}.pkl".format(args.kmeans)))
    else:
        k_means_model = KMeans(n_clusters=args.kmeans).fit(all_features)
        joblib.dump(k_means_model, os.path.join(args.result_path, "kmeans{}.pkl".format(args.kmeans)))
    print("K-means cluster over!")
    
    # 计算聚类结果和锚点标注的吻合度
    cluster_precision = evalClusterResult(args, n_data, data_loader, k_means_model)

    # 预测每个anchor(patch)的类别并保存可视化结果
    print("Start predicting anchor labels...")
    # predict_patch(args, data_loader, k_means_model, cluster_precision)

    # PCA可视化类别簇
    print("Start PCA visualization...")
    # 同时可视化训练集和测试集 for DEBUG
    vis_test = False
    if (vis_test):
        data_loader_test, n_data_test, _ = get_data_loader(args, subset="test_new")
        all_features_test = calcAllFeature(args, model, data_loader_test, n_data_test, recalc_feature=True)
        all_features = np.concatenate((all_features, all_features_test), axis=0)
        pcaVisualize(args, all_features, k_means_model, sub_dataset)
    else:
        pcaVisualize(args, all_features, k_means_model)

    # [只处理有标注锚点的帧]滑动窗预测图像上所有patch的类别并保存可视化结果
    # print("Start predicting all patch labels...")
    # predict_all_patch(args, data_loader, model, k_means_model)

    # 假设镜头前为路面区域，度量其他patch到该区域的特征距离
    # eval_dis_2_road(args, data_loader, model, k_means_model)

    # 将所有帧的可视化结果拼成video
    if (args.pre_video):
        print("Start predicting segmentation of video {}".format(args.pre_video))
        predict_vidoe(args, model, k_means_model)

if __name__ == '__main__':
    main()
