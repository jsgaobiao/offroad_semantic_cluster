import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_gaussian, create_pairwise_bilateral
from sklearn.cluster import KMeans
import numpy as np
import cv2
import os
import math
import time
import matplotlib.pyplot as plt
import joblib
import argparse

pred_res = 25    # 分类的分辨率：每个patch中间pred_res*pred_res的方块赋予该patch的类别标签
anchor_color = [(0,0,255), (0,255,0), (255,0,0), (0,255,255), (255,0,255), (255,255,0), (220,220,220), (31,102,156), (80,127,255), (140,230,240), (127,255,0), (158,168,3), (255,144,30), (214,112,218)]
anchor_marker = ['.','.','.','.','.','.','x','x','s','s','s','s','*','*']
anchor_label = [u"0:路",u"1:石头",u"2:植物",u"3:路边",u"4:建筑",u"5:碎石",u"6:水泥堆",u"7:木材",u"8:草泥/落叶","9:"]
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
#有中文出现的情况，需要u'内容'

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
    if not os.path.isdir(opt.result_path.replace("cluster_results", "crf")):
        os.makedirs(opt.result_path.replace("cluster_results", "crf"))
    
    # 将使用的配置表保存到文件中
    args_to_save = parser.parse_args()
    print(args_to_save)
    return opt

def dense_crf(probs, img, n_labels):
    # unary shape 为（n_labels, height, width）
    probs = probs.transpose((2, 0, 1))
    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)
    d = dcrf.DenseCRF(img.shape[0] * img.shape[1], n_labels) # width, height, n_labels
    d.setUnaryEnergy(unary)

    # This potential penalizes small pieces of segmentation that are
    # spatially isolated -- enforces more spatially consistent segmentations
    # feats = create_pairwise_gaussian(sdims=(10, 10), shape=img.shape[:2])
    # d.addPairwiseEnergy(feats, compat=3,
    #                     kernel=dcrf.DIAG_KERNEL,
    #                     normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This creates the color-dependent features --
    # because the segmentation that we get from CNN are too coarse
    # and we can use local color features to refine them
    feats = create_pairwise_bilateral(sdims=(50, 50), schan=(20, 20, 20),
                                      img=img, chdim=2)

    d.addPairwiseEnergy(feats, compat=10,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(5)
    Q = np.argmax(Q, axis=0).reshape((img.shape[0], img.shape[1]))
    return Q

def my_softmax(x):
    """Compute the softmax in a numerically stable way."""
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

def refine_video_by_crf(args, cluster_model):
    '''
        [处理所有视频帧] 使用crf优化粗糙的语义分割结果
    '''
    patch_size = 64
    cap = cv2.VideoCapture(args.pre_video)  # 读取待标注数据
    fps=cap.get(cv2.CAP_PROP_FPS)
    video_size=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)//3))
    tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tot_patches = 0
    selected_patches =0 
    wait_time = 0
    videoWriter = cv2.VideoWriter(os.path.join(args.result_path.replace("cluster_results", "."),'crf.avi'), cv2.VideoWriter_fourcc('M','J','P','G'), int(30), video_size)
    ################  读入视频帧  #################
    while True:
        ret, full_img = cap.read() # 读入一帧图像
        if not ret: # 读完整段视频，退出
            print('Video end!')
            break
        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        # if (frame_id % 100 != 0): continue
        features_for_save = np.array([])
        # 检查是否有已经计算好的特征文件，如果有就不用重新计算
        if os.path.isfile(os.path.join(args.result_path.replace("cluster_results", "features"), str(frame_id)+".npy")):
            # [注意: features_for_save 共2+128维，前2维是patch对应的坐标，后128维是特征向量]
            features_for_save = np.load(os.path.join(args.result_path.replace("cluster_results", "features"), str(frame_id)+".npy"))
        else:
            continue
        
        time0 = time.time()
        _crf_mask = np.zeros((full_img.shape), dtype=np.uint8)
        _patch_mask = np.zeros((full_img.shape), dtype=np.uint8)
        _prob_mask = np.zeros((full_img.shape[0], full_img.shape[1], args.kmeans), dtype=np.float32)
        for i in range(features_for_save.shape[0]):
            _i, _j = int(features_for_save[i,0]), int(features_for_save[i,1])
            pseudo_rbf_prob = np.zeros(args.kmeans)
            for _k in range(args.kmeans):
                cluster_center_feat = cluster_model.cluster_centers_[_k]
                # 计算到聚类中心的欧几里得距离
                euclidean_dis = np.linalg.norm(features_for_save[i, 2:] - cluster_center_feat)
                # 计算到聚类中心的RBF距离作为概率值
                _sigma = 0.25   # RBF kernel 的参数 length scale
                rbf_dis = np.exp(-euclidean_dis**2/(2*_sigma*_sigma))
                pseudo_rbf_prob[_k] = rbf_dis
            u_color = anchor_color[np.argmax(pseudo_rbf_prob)]
            _patch_mask = cv2.rectangle(_patch_mask, (_j-pred_res//2,_i-pred_res//2), (_j+pred_res//2,_i+pred_res//2), u_color, thickness=-1) #thickness=-1 表示矩形框内颜色填充    
            # 使用Softmax归一化,并计算熵
            pseudo_rbf_prob = my_softmax(pseudo_rbf_prob)
            
            # 将小块的patch上赋予同样的类别预测概率
            for x in range(_i-pred_res//2, _i+pred_res//2):
                for y in range(_j-pred_res//2, _j+pred_res//2):
                    _prob_mask[x][y] = pseudo_rbf_prob
            # print("time cost: {}, {}".format(time1-time0, time2-time1))
        time1 = time.time()

        ''' 使用CRF优化，返回的结果是类别标签 '''
        down_half_patch = dense_crf(probs=_prob_mask[_prob_mask.shape[0]//2:,:,:], img=full_img[full_img.shape[0]//2:,:,:], n_labels=args.kmeans)

        # 将crf后的类别可视化到图片上        
        # 滑动窗处理所有的patch，先不考虑天空，只滑动下半张图片
        for i in range(full_img.shape[0]//2, full_img.shape[0]):
            for j in range(0, full_img.shape[1]):
                # 将crf分类得到的类别标签绘制到图像上
                _crf_mask[i][j] = anchor_color[down_half_patch[i-full_img.shape[0]//2][j]]
        
        time2 = time.time()
        # alpha 为第一张图片的透明度，beta 为第二张图片的透明度 cv2.addWeighted 将原始图片与 mask 融合
        merged_crf_img = cv2.addWeighted(full_img, 0.8, _crf_mask, 0.2, 0)
        merged_lab_img = cv2.addWeighted(full_img, 0.8, _patch_mask, 0.2, 0)
        merged_img = np.concatenate((full_img, merged_crf_img, merged_lab_img), axis=1)
        cv2.putText(merged_img, "frame {}".format(frame_id), (50,50), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255), 3)
        cv2.imwrite(os.path.join(args.result_path.replace("cluster_results", "crf"), str(frame_id)+"_crf.png"), merged_img)
        videoWriter.write(cv2.resize(merged_img, video_size))
        print("frame {}, time cost: {:.3f} + {:.3f}".format(frame_id, time1-time0, time2-time1))
    videoWriter.release()

def main():# 供直接运行本脚本

    # parse argument
    args = parse_option()

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
    
    # 对所有视频帧进行CRF优化(读取已经保存好的features)
    refine_video_by_crf(args, cluster_model)

if __name__ == '__main__':
    main()
