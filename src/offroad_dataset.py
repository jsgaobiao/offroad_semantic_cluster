import numpy as np

import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import time
import random

class OffRoadDataset(Dataset):
    def __init__(self, root, subset='train', pos_sample_num=1, neg_sample_num=32, transform=None, channels=3, patch_size=64, background_size=192, use_data_aug_for_bg=False, rand_sample=True):
        super(OffRoadDataset, self).__init__()
        self.root = os.path.join(root, subset)
        self.neg_sample_num = neg_sample_num
        self.pos_sample_num = pos_sample_num
        self.transform = transform
        self.rand_sample = rand_sample
        self.background_size = background_size
        self.use_data_aug_for_bg = use_data_aug_for_bg
        self.channels = channels    # dim of input image channel (3: RGB, 5: RGBXY, 6: RGB+Background)
        self.patch_size = patch_size
        self.anchor_dict = np.load(os.path.join(self.root,"anchors_annotation.npy"), allow_pickle=True).item()
        self.anchor_list = []
        for _f_id in sorted(self.anchor_dict.keys()):
            for _ac_id in range(len(self.anchor_dict[_f_id])):
                self.anchor_list.append([_f_id] + self.anchor_dict[_f_id][_ac_id])  # [frame_id, anchor_x, anchor_y, anchor_type]
            

    def __len__(self):
        return len(self.anchor_list)
    
    def __getPatch__(self, _img, _x, _y, get_bg=False):
        ''' 从图像img中获取中心坐标为(_x, _y)的patch '''
        patch_size = self.patch_size
        # 如果取背景patch的话，尺寸默认是原始patch的3倍
        if get_bg: patch_size = self.background_size
        h, w = _img.shape[0:2]
        ps = patch_size // 2
        p_left_top = [max(_x-ps, 0), max(_y-ps, 0)]
        p_right_down = [min(_x+ps, w), min(_y+ps, h)]   # 右侧开区间
        # 如果在图像边缘处要特殊处理
        if (p_right_down[0] - p_left_top[0] < patch_size):
            if (p_left_top[0] == 0): p_right_down[0] = p_left_top[0] + patch_size
            if (p_right_down[0] == w): p_left_top[0] = p_right_down[0] - patch_size
        if (p_right_down[1] - p_left_top[1] < patch_size):
            if (p_left_top[1] == 0): p_right_down[1] = p_left_top[1] + patch_size
            if (p_right_down[1] == h): p_left_top[1] = p_right_down[1] - patch_size
        # if get_bg:
            # return cv2.resize(_img[p_left_top[1]:p_right_down[1], p_left_top[0]:p_right_down[0]], (self.patch_size, self.patch_size))
        # 【注意】如果取背景的话，返回的patch尺寸可能不一样
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
        
    def __getSample__(self, _img, _dat, rand_sample=False, sample_num=1, get_background=False):
        ''' 以(_x,_y)为中心进行patch采样，rand_sample可选择小范围随机patch， sample_num选择采样数量'''
        _x, _y, _lab = _dat[:]
        _patch_list = []
        _bg_list = []
        _patch_pos_list = []
        delta_x, delta_y = 0, 0
        ps = self.patch_size // 2
        for i in range(sample_num):
            if rand_sample:
                delta_x, delta_y = random.randint(-ps, ps), random.randint(-ps, ps)
            _patch_list.append(self.__getPatch__(_img, _x+delta_x, _y+delta_y))
            _patch_pos_list.append([_x+delta_x, _y+delta_y])
            # 是否采样patch的背景图片(channel==6的情况，单独保存背景patch)
            if get_background:
                _bg_list.append(self.__getPatch__(_img, _x+delta_x, _y+delta_y, get_bg=True))
            else:
                _bg_list.append(np.zeros((self.background_size, self.background_size, self.channels)))
        # self.drawAnchors(_img, [_dat])
        # cv2.imshow("patch", _patch_list[0])
        # cv2.waitKey(0)
        return np.array(_patch_list), np.array(_patch_pos_list), np.array(_bg_list)

    def __getitem__(self, idx):
        ''' 返回一个锚点，若干正样本（默认1个），若干负样本 
        :return: anchor, pos_sample, neg_sample
        '''
        time0 = time.time()
        frame_id = self.anchor_list[idx][0]  # anchor_list[i]: [frame_id, anchor_x, anchor_y, anchor_type]
        img_file = os.path.join(self.root, str(frame_id)+'.png')
        full_img = cv2.imread(img_file)
        # 如果channel参数为5，则通道为RGBXY, 其中x,y通道值域[0,1]
        if (self.channels == 5):
            xy_img = np.zeros((full_img.shape[0], full_img.shape[1], 2), dtype=np.float32)
            xy_img[:,:,0] = np.tile(np.expand_dims(np.arange(0, 1, 1/float(full_img.shape[0])).astype(np.float32), axis=1), (1,full_img.shape[1])) # 将x坐标归一化到0-255
            xy_img[:,:,1] = np.tile(np.arange(0, 1, 1/float(full_img.shape[1])).astype(np.float32), (full_img.shape[0], 1)) # 将x坐标归一化到0-255
            full_img = np.concatenate((full_img, xy_img), 2)
        # 如果channel参数为6，则单独提取背景patch作为额外的3个通道
        is_get_bg = False
        if (self.channels == 6):
            is_get_bg = True
            full_img = np.concatenate((full_img, full_img), axis=2)
        anchor, anchor_xy, anchor_bg = self.__getSample__(full_img, self.anchor_list[idx][1:], rand_sample=False, get_background=is_get_bg)

        # 挑选出anchor_type一样的作为正样本
        # pos_sample_list[i]: [x, y, anchor_type] # anchor_list[i]: [frame_id, anchor_x, anchor_y, anchor_type]
        pos_sample_list = [_dat for _dat in self.anchor_dict[frame_id] if _dat[2] == self.anchor_list[idx][3]]        
        # 在正样本集合中随机选取self.pos_sample_num（默认1）个正样本id
        pos_sample_id_list = [random.randint(0, len(pos_sample_list)-1) for i in range(self.pos_sample_num)]
        pos_sample = np.zeros((self.pos_sample_num, self.patch_size, self.patch_size, self.channels), dtype=np.uint8)
        pos_sample_xy = np.zeros((self.pos_sample_num, 2), dtype=np.int32)   # positive sample position
        pos_sample_bg = np.zeros((self.pos_sample_num, self.background_size, self.background_size, self.channels), dtype=np.uint8)
        for i, pos_id in enumerate(pos_sample_id_list):
            # 正样本： 在patch[_id]范围内随机选取新的中心点，作为正样本patch中心
            pos_sample[i], pos_sample_xy[i], pos_sample_bg[i] = self.__getSample__(full_img, pos_sample_list[pos_id], rand_sample=self.rand_sample, get_background=is_get_bg)

        time1 = time.time()

        # 挑选anchor_type不同的作为负样本 # neg_sample_list[i]: [x, y, anchor_type]
        neg_sample_list = [_dat for _dat in self.anchor_dict[frame_id] if _dat[2] != self.anchor_list[idx][3]]
        # 在负样本集合中随机选取self.neg_sample_num个负样本id
        neg_sample_id_list = [random.randint(0, len(neg_sample_list)-1) for i in range(self.neg_sample_num)]
        neg_sample = np.zeros((self.neg_sample_num, self.patch_size, self.patch_size, self.channels), dtype=np.uint8)
        neg_sample_xy = np.zeros((self.neg_sample_num, 2), dtype=np.int32)
        neg_sample_bg = np.zeros((self.neg_sample_num, self.background_size, self.background_size, self.channels), dtype=np.uint8)
        for i, neg_id in enumerate(neg_sample_id_list):
            # 负样本： 在patch[_id]范围内随机选取新的中心点，作为负样本patch中心
            neg_sample[i], neg_sample_xy[i], neg_sample_bg[i] = self.__getSample__(full_img, neg_sample_list[neg_id], rand_sample=self.rand_sample, get_background=is_get_bg)

        anchor_tensor = torch.zeros(1, self.channels, 224, 224)
        pos_sample_tensor = torch.zeros(pos_sample.shape[0], self.channels, 224, 224)
        neg_sample_tensor = torch.zeros(neg_sample.shape[0], self.channels, 224, 224)

        time2 = time.time()
        #################数据增强###############
        # 对RGB通道的transform
        # transform will change shape [num, H, W, channel] --> [num, channel, H, W]
        if self.transform is not None:
            if (self.channels == 3):
                anchor_tensor[0][:3] = self.transform(anchor[0,:,:,:3].astype(np.uint8))           # [1, H, W, channel] 
                for i in range(pos_sample.shape[0]):
                    pos_sample_tensor[i][:3] = self.transform(pos_sample[i,:,:,:3].astype(np.uint8))     # [P, H, W, channel]  default: P=1
                for i in range(neg_sample.shape[0]):
                    neg_sample_tensor[i][:3] = self.transform(neg_sample[i,:,:,:3].astype(np.uint8))     # [K, H, W, channel] 
            # 如果是5通道（RGBXY）需要对XY通道单独做变换
            if (self.channels == 5):
                mean = [0.5, 0.5]
                std = [0.2887, 0.2887]
                anchor_tensor[0][3:] = torch.from_numpy(np.transpose(cv2.resize(anchor[0,:,:,3:], (224, 224)), (2,0,1))).to(anchor_tensor)
                for i in range(pos_sample.shape[0]):
                    pos_sample_tensor[i][3:] = torch.from_numpy(np.transpose(cv2.resize(pos_sample[i,:,:,3:], (224, 224)), (2,0,1))).to(pos_sample_tensor)     # [P, H, W, channel]  default: P=1
                for i in range(neg_sample.shape[0]):
                    neg_sample_tensor[i][3:] = torch.from_numpy(np.transpose(cv2.resize(neg_sample[i,:,:,3:], (224, 224)), (2,0,1))).to(neg_sample_tensor)     # [K, H, W, channel] 
                xy_transform = transforms.Compose([
                    transforms.Normalize(mean=mean, std=std),
                ])
                anchor_tensor[0][3:] = xy_transform(anchor_tensor[0][3:])
                for i in range(pos_sample.shape[0]):
                    pos_sample_tensor[i][3:] = xy_transform(pos_sample_tensor[i][3:])     # [P, H, W, channel]  default: P=1
                for i in range(neg_sample.shape[0]):
                    neg_sample_tensor[i][3:] = xy_transform(neg_sample_tensor[i][3:])     # [K, H, W, channel] 
            # 如果是6通道（前景patch+背景patch），需要对前背景patch同时处理，保持同步transform
            if (self.channels == 6):
                # 先把前背景resize到相同尺寸，再打包成patch进行数据增强
                resize_transform = transforms.Compose([
                    transforms.Resize((224,224))])
                mean = [0.5200442, 0.5257094, 0.517397]
                std = [0.335111, 0.33463535, 0.33491987]
                # 训练的时候使用前背景同时数据增强，测试的时候不做数据增强
                if self.use_data_aug_for_bg:
                    da_transform = transforms.Compose([
                        transforms.RandomGrayscale(p=0.2),
                        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip()
                    ])
                else:
                    da_transform = transforms.Compose([])

                norm_transform = transforms.Compose([transforms.Normalize(mean=mean, std=std)])
                # [B,C,H,W]
                fg_bg_tensor = torch.cat((resize_transform(torch.from_numpy(anchor[0,:,:,:3].astype(np.uint8).transpose(2,0,1))).view(1,3,224,224),
                                        resize_transform(torch.from_numpy(anchor_bg[0,:,:,:3].astype(np.uint8).transpose(2,0,1))).view(1,3,224,224)), 
                                        axis=0)
                anchor_tensor[0] = norm_transform(da_transform(fg_bg_tensor).float().div(255)).reshape(self.channels,224,224)           # anchor_tensor [1, H, W, channel] 
                
                for i in range(pos_sample.shape[0]):
                    fg_bg_tensor = torch.cat((resize_transform(torch.from_numpy(pos_sample[i,:,:,:3].astype(np.uint8).transpose(2,0,1))).view(1,3,224,224),
                                        resize_transform(torch.from_numpy(pos_sample_bg[i,:,:,:3].astype(np.uint8).transpose(2,0,1))).view(1,3,224,224)), 
                                        axis=0)
                    pos_sample_tensor[i][:] = norm_transform(da_transform(fg_bg_tensor).float().div(255)).reshape(self.channels,224,224)     # pos_sample_tensor [P, H, W, channel]  default: P=1
                
                for i in range(neg_sample.shape[0]):
                    fg_bg_tensor = torch.cat((resize_transform(torch.from_numpy(neg_sample[i,:,:,:3].astype(np.uint8).transpose(2,0,1))).view(1,3,224,224),
                                        resize_transform(torch.from_numpy(neg_sample_bg[i,:,:,:3].astype(np.uint8).transpose(2,0,1))).view(1,3,224,224)), 
                                        axis=0)
                    neg_sample_tensor[i][:] = norm_transform(da_transform(fg_bg_tensor).float().div(255)).reshape(self.channels,224,224)     # neg_sample_tensor [K, H, W, channel] 
                
                time3 = time.time()

        # print("get_item time cost: {} / {} / {}".format(time1-time0, time2-time1, time3-time2))
        anchor_type = self.anchor_list[idx][3]
        return anchor_tensor, pos_sample_tensor, neg_sample_tensor, frame_id, full_img, anchor_xy, pos_sample_xy, neg_sample_xy, anchor_type




