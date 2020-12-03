import numpy as np

import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import random

class OffRoadDataset(Dataset):
    def __init__(self, root, subset='train', neg_sample_num=32, transform=None, patch_size=32):
        super(OffRoadDataset, self).__init__()
        self.root = os.path.join(root, 'train')
        self.neg_sample_num = neg_sample_num
        self.transform = transform
        self.patch_size = patch_size
        self.anchor_dict = np.load(os.path.join(self.root,"anchors_annotation.npy"), allow_pickle=True).item()
        self.anchor_list = []
        for _f_id in sorted(self.anchor_dict.keys()):
            for _ac_id in range(len(self.anchor_dict[_f_id])):
                self.anchor_list.append([_f_id] + self.anchor_dict[_f_id][_ac_id])  # [frame_id, anchor_x, anchor_y, anchor_type]
            

    def __len__(self):
        return len(self.anchor_list)
    
    def __getPatch__(self, _img, _x, _y):
        ''' 从图像img中获取中心坐标为(_x, _y)的patch '''
        h, w = _img.shape[0:2]
        ps = self.patch_size // 2
        p_left_top = [max(_x-ps, 0), max(_y-ps, 0)]
        p_right_down = [min(_x+ps, w), min(_y+ps, h)]   # 右侧开区间
        # 如果在图像边缘处要特殊处理
        if (p_right_down[0] - p_left_top[0] < self.patch_size):
            if (p_left_top[0] == 0): p_right_down[0] = p_left_top[0] + self.patch_size
            if (p_right_down[0] == w): p_left_top[0] = p_right_down[0] - self.patch_size
        if (p_right_down[1] - p_left_top[1] < self.patch_size):
            if (p_left_top[1] == 0): p_right_down[1] = p_left_top[1] + self.patch_size
            if (p_right_down[1] == h): p_left_top[1] = p_right_down[1] - self.patch_size

        return _img[p_left_top[1]:p_right_down[1], p_left_top[0]:p_right_down[0]]
    
    # 绘制当前帧的已有锚点 [for DEBUG]
    # def drawAnchors(self, img, anchors):
    #     resize_ratio = 2.0
    #     anchor_color = [(0,0,255), (0,255,0), (255,0,0), (0,255,255), (255,0,255), (255,255,0)]
    #     for _p in anchors:
    #         p_left_top = (max(_p[0]-32, 0), max(_p[1]-32, 0))
    #         p_right_down = (min(_p[0]+32, img.shape[1]-1), min(_p[1]+32, img.shape[0]-1))
    #         cv2.rectangle(img, p_left_top, p_right_down, anchor_color[_p[2] % len(anchor_color)], thickness=4)
    #         cv2.putText(img, str(_p[2]), p_left_top, cv2.FONT_HERSHEY_SIMPLEX, 3, anchor_color[_p[2] % len(anchor_color)], 2)
    #     vis_frame = cv2.resize(img, (int(img.shape[1]/resize_ratio), int(img.shape[0]/resize_ratio)))
    #     cv2.imshow('video', vis_frame)
        
    def __getSample__(self, _img, _dat, rand_sample=False, sample_num=1):
        ''' 以(_x,_y)为中心进行patch采样，rand_sample可选择小范围随机patch， sample_num选择采样数量'''
        _x, _y, _lab = _dat[:]
        _patch_list = []
        delta_x, delta_y = 0, 0
        ps = self.patch_size // 2
        for i in range(sample_num):
            if rand_sample:
                delta_x, delta_y = random.randint(-ps, ps), random.randint(-ps, ps)
            _patch_list.append(self.__getPatch__(_img, _x+delta_x, _y+delta_y))
        # self.drawAnchors(_img, [_dat])
        # cv2.imshow("patch", _patch_list[0])
        # cv2.waitKey(0)
        return np.array(_patch_list)

    def __getitem__(self, idx):
        ''' 返回一个锚点，一个正样本，若干负样本 
        :return: anchor, pos_sample, neg_sample
        '''
        frame_id = self.anchor_list[idx][0]  # anchor_list[i]: [frame_id, anchor_x, anchor_y, anchor_type]
        img_file = os.path.join(self.root, str(frame_id)+'.png')
        full_img = cv2.imread(img_file)
        anchor = self.__getSample__(full_img, self.anchor_list[idx][1:], rand_sample=False)

        # 挑选出anchor_type一样的作为正样本
        # pos_sample_list[i]: [x, y, anchor_type] # anchor_list[i]: [frame_id, anchor_x, anchor_y, anchor_type]
        pos_sample_list = [_dat for _dat in self.anchor_dict[frame_id] if _dat[2] == self.anchor_list[idx][3]]
        # 随机选一个正样本
        pos_sample_id = random.randint(0, len(pos_sample_list)-1)
        pos_sample = self.__getSample__(full_img, pos_sample_list[pos_sample_id], rand_sample=True)

        # 挑选anchor_type不同的作为负样本 # neg_sample_list[i]: [x, y, anchor_type]
        neg_sample_list = [_dat for _dat in self.anchor_dict[frame_id] if _dat[2] != self.anchor_list[idx][3]]
        # 在负样本集合中随机选取self.neg_sample_num个负样本id
        neg_sample_id_list = [random.randint(0, len(neg_sample_list)-1) for i in range(self.neg_sample_num)]
        neg_sample = np.zeros((self.neg_sample_num, self.patch_size, self.patch_size, 3), dtype=np.uint8)
        for i, neg_id in enumerate(neg_sample_id_list):
            # 负样本： 在patch[_id]范围内随机选取新的中心点，作为负样本patch中心
            neg_sample[i] = self.__getSample__(full_img, neg_sample_list[neg_id], rand_sample=True)

        anchor_tensor = torch.zeros(1, 3, 224, 224)
        pos_sample_tensor = torch.zeros(1, 3, 224, 224)
        neg_sample_tensor = torch.zeros(neg_sample.shape[0], 3, 224, 224)
        # transform will change shape [num, H, W, channel] --> [num, channel, H, W]
        if self.transform is not None:
            anchor_tensor[0] = self.transform(anchor[0])           # [1, H, W, channel] 
            pos_sample_tensor[0] = self.transform(pos_sample[0])   # [1, H, W, channel]
            for i in range(neg_sample.shape[0]):
                neg_sample_tensor[i] = self.transform(neg_sample[i])     # [K, H, W, channel] 

        return anchor_tensor, pos_sample_tensor, neg_sample_tensor




