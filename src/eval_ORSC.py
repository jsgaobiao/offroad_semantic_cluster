import os
import sys
import time
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn as nn
import argparse
import cv2
import numpy as np
import math

from torchvision import transforms, datasets

from models.alexnet import MyAlexNetCMC
from offroad_dataset import OffRoadDataset
from util import AverageMeter
import matplotlib.pyplot as plt

def parse_option():
    parser = argparse.ArgumentParser('argument for evaluation')
    parser.add_argument('--data_folder', type=str, default=None, help='path to data')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--subset', type=str, default="train", help='subset for training')
    # resume path
    parser.add_argument('--model_path', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--result_path', type=str, default="results", help='path to save result')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--note', type=str, default=None, help='comments to current train settings')
    parser.add_argument('--softmax', action='store_true', help='using softmax contrastive loss rather than NCE')
    parser.add_argument('--nce_k', type=int, default=8, help='负样本数量')
    parser.add_argument('--nce_t', type=float, default=0.07)
    parser.add_argument('--nce_m', type=float, default=0.5, help='the momentum for dynamically updating the memory.')
    parser.add_argument('--feat_dim', type=int, default=128, help='dim of feat for inner product')
    parser.add_argument('--in_channel', type=int, default=3, help='dim of input image channel (3: RGB, 5: RGBXY)')

    opt = parser.parse_args()

    if (opt.data_folder is None) or (opt.model_path is None) or (opt.result_path is None):
        raise ValueError('one or more of the folders is None: data_folder | model_path | result_path')
    
    if opt.note != None:
        opt.result_path = os.path.join(opt.result_path, opt.note)
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

def caseStudy(args, _full_img, _anchor_xy, _pos_sample_xy, _neg_sample_xy, _pos_feat_dis, _neg_feat_dis, color_map):
    ''' 可视化选取的锚点和正负样本，并将它们的特征相似度绘制在图像上 '''
    anchor_width = 64
    anchor_color = [(0,0,255), (0,255,0), (255,0,0), (0,255,255), (255,0,255), (255,255,0), (255, 191, 0), (0, 191, 255), (128, 0, 255)]
    full_img = _full_img[:,:,:3].numpy().astype(np.uint8)
    anchor_xy = _anchor_xy.numpy()
    pos_sample_xy = _pos_sample_xy.numpy()
    neg_sample_xy = _neg_sample_xy.numpy()
    pos_feat_dis = _pos_feat_dis.cpu().numpy()
    neg_feat_dis = _neg_feat_dis.cpu().numpy()
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
    # 绘制anchor
    for _p in anchor_xy:
        p_left_top, p_right_down = getPatchXY(full_img, _p[0], _p[1], anchor_width)
        cv2.rectangle(full_img, tuple(p_left_top), tuple(p_right_down), anchor_color[0], thickness=4)
        cv2.putText(full_img, "anchor", tuple(p_right_down), cv2.FONT_HERSHEY_SIMPLEX, 1, anchor_color[0], 2)
    # 绘制正样本
    for (_p, _feat) in zip(pos_sample_xy, pos_feat_dis):
        p_left_top, p_right_down = getPatchXY(full_img, _p[0], _p[1], anchor_width)
        pos_color = tuple(getColor(_feat[0], color_map))
        cv2.rectangle(full_img, tuple(p_left_top), tuple(p_right_down), pos_color, thickness=4)
        cv2.putText(full_img, "P({:.2f})".format(_feat[0]), tuple(p_right_down), cv2.FONT_HERSHEY_SIMPLEX, 1, pos_color, 2)
    # 绘制负样本
    for (_p, _feat) in zip(neg_sample_xy, neg_feat_dis):
        p_left_top, p_right_down = getPatchXY(full_img, _p[0], _p[1], anchor_width)
        neg_color = tuple(getColor(_feat[0], color_map))
        cv2.rectangle(full_img, tuple(p_left_top), tuple(p_right_down), neg_color, thickness=4)
        cv2.putText(full_img, "N({:.2f})".format(_feat[0]), tuple(p_right_down), cv2.FONT_HERSHEY_SIMPLEX, 1, neg_color, 2)
    
    return full_img

def calcAverageDis(args, model, data_loader):
    ''' 采样计算锚点与正负样本的平均距离 '''
    posDises = AverageMeter()
    negDises = AverageMeter()
    K = args.nce_k
    # 根据样本与锚点的距离，绘制不同的颜色进行可视化
    color_map = cv2.applyColorMap(np.arange(0, 256, dtype=np.uint8), cv2.COLORMAP_JET)
    with torch.no_grad():
        for idx, (anchor, pos_sample, neg_sample, frame_id, full_img, anchor_xy, pos_sample_xy, neg_sample_xy, _) in enumerate(data_loader):
            # anchor shape: [batch_size, 1, channel, H, W] # neg_sample,pos_sample shape: [batch_size, K, channel, H, W]
            batch_size = anchor.size(0)
            # inputs shape --> [batch_size, (1+2K), channel, H, W]
            inputs = torch.cat((anchor, pos_sample, neg_sample), dim=1)
            inputs_shape = list(inputs.size())
            # inputs shape --> [batch_size*(1+2K), channel, H, W]
            inputs = inputs.view((inputs_shape[0]*inputs_shape[1], inputs_shape[2], inputs_shape[3], inputs_shape[4]))
            inputs = inputs.float()
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            # ===================forward=====================
            feature = model(inputs)     # [batch_size*(1+2K), feature_dim]
            # ==============calculate distance===============
            feature_size = feature.size(1) # 获取特征的维度
            for i in range(batch_size): #逐个计算每个锚点与其正负样本的距离
                anchor = feature[i*(K+K+1)].view(feature_size, -1) # 这里相当于扩维+矩阵转置
                pos = feature[i*(K+K+1)+1   : i*(K+K+1)+1+K]
                neg = feature[i*(K+K+1)+1+K : (i+1)*(K+K+1)]
                # 计算锚点到K个正样本的平均距离
                pos_feature = torch.mm(pos, anchor) # [K * 1]
                pos_feat_dis = torch.exp(pos_feature)
                posDises.update(torch.mean(pos_feat_dis))
                # 计算锚点到K个负样本的平均距离
                neg_feature = torch.mm(neg, anchor)
                neg_feat_dis = torch.exp(neg_feature)
                negDises.update(torch.mean(neg_feat_dis))
                # 每个batch抽第一组[anchor和正负样本]，进行可视化（case study）
                if (i == 0):
                    img2save = caseStudy(args, full_img[i], anchor_xy[i], pos_sample_xy[i], neg_sample_xy[i], pos_feat_dis, neg_feat_dis, color_map)
                    cv2.imwrite(os.path.join(args.result_path, str(frame_id[i].numpy())+".png"), img2save)
            # print info
            if (idx + 1) % args.print_freq == 0:
                print('Eval: [{0}/{1}]\t'
                    'posDises {posDises.val:.3f} ({posDises.avg:.3f})\t'
                    'negDises {negDises.val:.3f} ({negDises.avg:.3f})'.format(
                    idx + 1, len(data_loader), posDises=posDises, negDises=negDises))
                # print(out_l.shape)
                sys.stdout.flush()
            
    return posDises.avg, negDises.avg

def main():# 供直接运行本脚本

    # parse argument
    args = parse_option()

    # set the data loader (n_data: dataset size)
    data_loader, n_data = get_data_loader(args, subset=args.subset)

    # load model
    model = set_model(args)

    # 随机抽取正负样本对，并计算和锚点的平均距离
    posDis, negDis = calcAverageDis(args, model, data_loader)
    print('positive sample average distance = {}, negative sample average distance = {}'.format(posDis, negDis))
    fout = open(os.path.join(args.result_path,"average_dis.log"), 'w')
    fout.write('positive sample average distance = {}, negative sample average distance = {}'.format(posDis, negDis))
    fout.close()

if __name__ == '__main__':
    main()
