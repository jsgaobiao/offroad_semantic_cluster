'''
Author: your name
Date: 2021-05-24 16:34:27
LastEditTime: 2021-07-01 21:16:20
LastEditors: Please set LastEditors
Description: 读入rbf_response.csv，进行数据可视化
FilePath: \risk_coverage_visualization\vis_rbf_response.py
'''
from heapq import merge
import PIL.Image as Image
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import cv2
import joblib
import os
from sklearn.decomposition import PCA,KernelPCA
from selective_patch_cluster_for_local_PC import parse_option,load_anno

pred_res = 25
anchor_color = [(0,0,255), (0,255,0), (255,0,0), (0,255,255), (255,0,255), (255,128,0), (31,102,156), (220,220,220), (80,127,255), (140,230,240), (127,255,0), (158,168,3), (255,144,30), (214,112,218)]
# 每次聚类结果不同，类别编号会不一样，为了统一不同模型结果的可视化效果，所以对不同的模型使用不同的color_map
#                              植物       土路        路边          others       落叶路      碎石灰/土路  木头/石头
anchor_color_train_mNull = [(255,0,0), (0,255,0), (240,240,0), (31,102,156), (0,240,240), (255,0,255), (0,0,255), (80,127,255), (220,220,220), (140,230,240), (127,255,0), (158,168,3), (255,144,30), (214,112,218)]
#                             土路        植物       路边       建筑/others  碎石/土路/落叶   石头/木头      
anchor_color_train_m678 = [(0,255,0), (255,0,0), (240,240,0), (31,102,156), (255,0,255), (0,0,255), (80,127,255), (0,240,240), (220,220,220), (80,127,255), (140,230,240), (127,255,0), (158,168,3), (255,144,30), (214,112,218)]
#                               土路        植物       路边         others 
anchor_color_train_m145678 = [(0,255,0), (255,0,0), (240,240,0), (0,0,255), (31,102,156), (255,0,255), (80,127,255), (0,240,240), (220,220,220), (80,127,255), (140,230,240), (127,255,0), (158,168,3), (255,144,30), (214,112,218)]
anchor_marker = ['.','.','.','.','.','.','x','x','s','s','s','s','*','*']
#有中文出现的情况，需要u'内容'
anchor_label = [u"0:路",u"1:石头",u"2:植物",u"3:路边",u"4:建筑",u"5:碎石",u"6:水泥堆",u"7:木材",u"8:草泥/落叶","9:"]
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
rbf_sigma = 0.25   # RBF kernel 的参数 length scale

def fig2data(fig):
    '''
        fig = plt.figure()
        image = fig2data(fig)
        @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
        @param fig a matplotlib figure
        @return a numpy 3D array of RGBA values
    '''
    # draw the renderer
    fig.canvas.draw()
    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tobytes())
    image = np.asarray(image)
    return image[:,:,:-1]   # 丢弃Alpha通道

def drawScatter(X, Y, C, isize, xlim=[], ylim=[], anchor_num=0, draw_line=False):
    '''使用opencv绘制散点图'''
    _margin = 25
    if isize[0] <= _margin*2 or isize[1] <= _margin*2:
        _margin = 0
    if xlim == []:
        xlim = [X.min(), X.max()]
    if ylim == []:
        ylim = [Y.min(), Y.max()]
    ret_img = np.zeros(isize, dtype=np.uint8) + 255
    assert X.shape == Y.shape
    # 非黑色的点，加粗作为背景点画出来
    if draw_line:
        for i in range(1, X.shape[0]):
            x_pixel = int(_margin + (X[i] - xlim[0]) / (xlim[1] - xlim[0]) * (isize[1]-2*_margin))
            y_pixel = int(_margin + (Y[i] - ylim[0]) / (ylim[1] - ylim[0]) * (isize[0]-2*_margin))
            assert x_pixel>=0 and y_pixel>=0 and x_pixel<isize[1] and y_pixel<isize[0]
            if C[i-1] != (0,0,0):
                x0_pixel = int(_margin + (X[i-1] - xlim[0]) / (xlim[1] - xlim[0]) * (isize[1]-2*_margin))
                y0_pixel = int(_margin + (Y[i-1] - ylim[0]) / (ylim[1] - ylim[0]) * (isize[0]-2*_margin))
                cv2.line(ret_img, (x0_pixel, y0_pixel), (x_pixel, y_pixel), C[i-1], thickness=4, lineType=cv2.LINE_AA)
    for i in range(X.shape[0]):
        if draw_line and i == 0:
            continue
        x_pixel = int(_margin + (X[i] - xlim[0]) / (xlim[1] - xlim[0]) * (isize[1]-2*_margin))
        y_pixel = int(_margin + (Y[i] - ylim[0]) / (ylim[1] - ylim[0]) * (isize[0]-2*_margin))
        assert x_pixel>=0 and y_pixel>=0 and x_pixel<isize[1] and y_pixel<isize[0]
        # 锚点用x来画
        if not draw_line:
            if i < anchor_num:
                cv2.putText(ret_img, 'x', (x_pixel, y_pixel), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C[i])
            else:
                cv2.circle(ret_img, (x_pixel, y_pixel), 2, C[i], -1)
        else:  # 画折线
            x0_pixel = int(_margin + (X[i-1] - xlim[0]) / (xlim[1] - xlim[0]) * (isize[1]-2*_margin))
            y0_pixel = int(_margin + (Y[i-1] - ylim[0]) / (ylim[1] - ylim[0]) * (isize[0]-2*_margin))
            cv2.line(ret_img, (x0_pixel, y0_pixel), (x_pixel, y_pixel), (0,0,0), thickness=1, lineType=cv2.LINE_AA)
            
    ret_img = cv2.flip(ret_img, 0)
    return ret_img

def getRiskCoverageOfAnchors(args, cluster_model, anchor_feature_list, type_of_anchor_feature_list):
    # 计算训练锚点集合上，coverage和risk的对应关系，可视化热力图的时候需要根据它作颜色的映射
    risk_coverage = []  # [max_rbf_prob, pseudo_rbf_prob(到每个聚类中心的距离)]
    for i in range(anchor_feature_list.shape[0]):
        pseudo_rbf_prob = np.zeros(args.kmeans)

        for _k in range(args.kmeans):
            cluster_center_feat = cluster_model.cluster_centers_[_k]
            # 计算到聚类中心的欧几里得距离
            euclidean_dis = np.linalg.norm(anchor_feature_list[i, :] - cluster_center_feat)
            # 计算到聚类中心的RBF距离作为概率值
            rbf_dis = np.exp(-euclidean_dis**2/(2*rbf_sigma*rbf_sigma))
            pseudo_rbf_prob[_k] = rbf_dis
        # 取1-最大值，这样离聚类中心越近，值越小
        risk_coverage.append(1-pseudo_rbf_prob.max())

    ''' 计算Risk_Coverage曲线，Risk使用到聚类中心的RBF_Prob计算'''
    risk_coverage = sorted(risk_coverage)    # 按照response排序，逐步增加coverage并绘risk_coverage图
    return risk_coverage

def get_projected_risk(risk_coverage_sorted_list, input_risk):
    ''' 将测试集上的patch risk根据训练集上的risk_coverage_curve投影为coverage概率 '''
    return np.searchsorted(risk_coverage_sorted_list, input_risk) / len(risk_coverage_sorted_list)

def risky_data_selection(frm_ids, requested_coverage_curve, select_frame_num = 50):
    ''' 根据requested_coverage_curve'''
    IRANGE = 100  # 前后IRANGE帧内只标注一帧的锚点
    flag_of_frm = np.zeros(len(frm_ids)+1)
    unrank_coverage = np.concatenate((np.expand_dims(requested_coverage_curve,1), np.expand_dims(frm_ids,1)), axis=1)
    selected_frame_list = []
    rank_coverage = unrank_coverage[np.lexsort(-unrank_coverage[:,::-1].T)]
    idx = 0
    delta = rank_coverage[0][0] - 0.85
    ratio = delta / IRANGE
    # 线性下降coverage屏蔽法
    while len(selected_frame_list) < select_frame_num:
        frm_idx = int(rank_coverage[0][1])
        selected_frame_list.append(frm_idx)
        # 以frm_idx为中心，将coverage_curve进行线性下降
        for i in range(max(0,frm_idx-IRANGE), min(frm_idx+IRANGE, unrank_coverage.shape[0]-1)+1):
            unrank_coverage[i] -= (delta - abs(i-frm_idx) * ratio)
        # 线性下降coverage后重新排序
        rank_coverage = unrank_coverage[np.lexsort(-unrank_coverage[:,::-1].T)]
    # 固定区间屏蔽法
    # cur_flag = 1
    # while idx < rank_coverage.shape[0]:
    #     frm_idx = int(rank_coverage[idx][1])
    #     if flag_of_frm[frm_idx] < cur_flag:
    #         selected_frame_list.append(frm_idx)
    #         for i in range(max(0,frm_idx-IRANGE), min(frm_idx+IRANGE, rank_coverage.shape[0]-1)+1):
    #             flag_of_frm[i] = cur_flag
    #         flag_of_frm[frm_idx] = 1e5  # 确保被选过的帧不会再选一次
    #     idx += 1
    #     if len(selected_frame_list) >= select_frame_num:
    #         break
    #     # 如果遍历完还没找到足够的frame，就忽视屏蔽区间，再筛选一轮
    #     if idx == rank_coverage.shape[0]:
    #         cur_flag += 1
    #         idx = 0
    return np.array(selected_frame_list)
################################################################################################################################

# parse argument
args = parse_option()
# 读入标注数据
anchor_dict, anchor_list = load_anno(args)
# 计算所有patch对应的特征向量
anchor_feature_list = np.load(os.path.join(args.result_path, "all_patch_features.npy"), )
type_of_anchor_feature_list = np.load(os.path.join(args.result_path, "all_patch_features_type.npy"))

cluster_method = "kmeans" 
if cluster_method == "kmeans":
    # K-means聚类
    if (os.path.isfile(os.path.join(args.result_path, "kmeans{}.pkl".format(args.kmeans)))):
        cluster_model = joblib.load(os.path.join(args.result_path, "kmeans{}.pkl".format(args.kmeans)))
    else:
        cluster_model = KMeans(n_clusters=args.kmeans).fit(anchor_feature_list)
        joblib.dump(cluster_model, os.path.join(args.result_path, "kmeans{}.pkl".format(args.kmeans)))
    print("K-means {} cluster over!".format(args.kmeans))

# 计算训练锚点集合上，coverage和risk的对应关系，可视化热力图的时候需要根据它作颜色的映射
risk_coverage_sorted_list = getRiskCoverageOfAnchors(args, cluster_model, anchor_feature_list, type_of_anchor_feature_list)

'''读入数据'''
all_frm_rbf_response = []
frm_ids = []
with open(os.path.join(args.result_path, "../rbf_response.csv"), "r") as f:
    # frm_id, rbf...
    for line in f:
        dat = line.strip().split(',')
        dat = list(map(float, dat[:-1]))
        all_frm_rbf_response.append(dat[1:])
        frm_ids.append(dat[0])
all_frm_rbf_response = np.array(all_frm_rbf_response)
frm_ids = np.array(frm_ids)

# all_frm_entropy = []
# with open(os.path.join(args.result_path, "../entropy.csv"), "r") as f:
#     # frm_id, rbf...
#     for line in f:
#         dat = line.strip().split(',')
#         dat = list(map(float, dat[:-1]))
#         all_frm_entropy.append(dat[1:])
# all_frm_entropy = np.array(all_frm_entropy)

all_frm_cluster_label = []
with open(os.path.join(args.result_path, "../cluster_label.csv"), "r") as f:
    # frm_id, rbf...
    for line in f:
        dat = line.strip().split(',')
        dat = list(map(float, dat[:-1]))
        all_frm_cluster_label.append(dat[1:])
all_frm_cluster_label = np.array(all_frm_cluster_label)

''' 视频损坏帧的rbf值是异常值，需要清理 '''
all_frm_mean_rbf = np.mean(all_frm_rbf_response, axis=1)
for i in range(1, all_frm_mean_rbf.shape[0]):
    if abs(all_frm_mean_rbf[i]-all_frm_mean_rbf[i-1]) > 0.1:
        frm_ids[i] = frm_ids[i-1]
        all_frm_mean_rbf[i] = all_frm_mean_rbf[i-1]
        all_frm_rbf_response[i] =  all_frm_rbf_response[i-1]
        # all_frm_entropy[i] = all_frm_entropy[i-1]
        all_frm_cluster_label[i] = all_frm_cluster_label[i-1]
# min_entropy = np.min(all_frm_entropy)
# max_entoropy = np.max(all_frm_entropy)
'''绘制每帧rbf均值的散点图'''
# 训练集coverage为0.9时,rbf的阈值
# 0.7038474623833872   完整数据集，kmeans=7
# 0.7013018535186726   屏蔽 6,7,8 的阈值，kmeans=6
# 0.4874402829851514   屏蔽 1,4,5,6,7,8 的阈值，kmeans=4
train_rbf_threshold = 0.85
if "0429_eval_cluster_by_uncertainty" in args.note:
    rbf_threshold = 0.5735515444085504  #risk0.85         #0.7038474623833872 #risk0.9
    anchor_color = anchor_color_train_mNull
elif "145678" in args.note:
    rbf_threshold = 0.38533537311489996  #risk0.85        #0.4874402829851514 #risk0.9
    anchor_color = anchor_color_train_m145678
elif "678" in args.note:
    rbf_threshold = 0.5771175533001448  #risk0.85         #0.7013018535186726 #risk0.9
    anchor_color = anchor_color_train_m678

rbf_threshold = risk_coverage_sorted_list[int(len(risk_coverage_sorted_list)*0.85)]
x_axis = frm_ids
fig0, ax0 = plt.subplots(figsize=(20,3))
ax0.scatter(x_axis[all_frm_mean_rbf > rbf_threshold], all_frm_mean_rbf[all_frm_mean_rbf > rbf_threshold], c='r', s=1)
ax0.scatter(x_axis[all_frm_mean_rbf <= rbf_threshold], all_frm_mean_rbf[all_frm_mean_rbf <= rbf_threshold], c='k', s=1)
ax0.set_ylim(0.2, 0.8)
fig1, ax1 = plt.subplots(figsize=(20,3))
coverage_of_all_frame = []
coverage_prob_of_all_frame = [] # 每帧的平均rbf_risk，根据训练集的risk_coverage曲线，将risk映射为coverage值（使得不同模型的结果具有可比性）
for i in range(all_frm_mean_rbf.shape[0]):
    coverage_of_all_frame.append(np.count_nonzero(all_frm_rbf_response[i] < rbf_threshold) / all_frm_rbf_response[i].shape[0])
    coverage_prob_of_all_frame.append(get_projected_risk(risk_coverage_sorted_list, all_frm_mean_rbf[i]))
coverage_prob_of_all_frame = np.array(coverage_prob_of_all_frame)
print("每帧平均risk指标下，数据集整体coverage: {}".format(np.count_nonzero(coverage_prob_of_all_frame <= 0.85)/coverage_prob_of_all_frame.shape[0]))
print("patch_risk指标下，数据集整体coverage: {}".format(np.count_nonzero(all_frm_rbf_response <= rbf_threshold)/(all_frm_rbf_response.shape[0]*all_frm_rbf_response.shape[1])))
# print("数据集整体,平均每帧的coverage: {}".format(np.mean(coverage_of_all_frame)))

# 把可以横向比较的coverage_prob_of_all_frame曲线写入文件
with open(os.path.join(args.result_path.replace("cluster_results", "."),'mean_coverage_prob.csv'), 'a+') as curve_f:
    curve_f.write("note={},k={},anchor_lab_mask={}\n".format(args.note, args.kmeans, args.anchor_lab_mask))
    for i in range(coverage_prob_of_all_frame.shape[0]):
        curve_f.write("{},".format(coverage_prob_of_all_frame[i]))
    curve_f.write('\n')
ax1.plot(x_axis, coverage_prob_of_all_frame, c='k')
ax1.scatter(x_axis[coverage_prob_of_all_frame > train_rbf_threshold], coverage_prob_of_all_frame[coverage_prob_of_all_frame > train_rbf_threshold], c='r', s=20)
ax1.set_ylim(0.5, 1.0)
fig0.savefig(os.path.join(args.result_path,'../rbf_of_all_frame.png'))
fig1.savefig(os.path.join(args.result_path,'../mean_coverage_prob.png'))    # requested_coverage_curve

# 在requested_coverage_curve上画出自动选出的待标注帧位置
selected_frm = risky_data_selection(frm_ids, coverage_prob_of_all_frame, select_frame_num = 30)
np.savetxt(os.path.join(args.result_path,'../selected_frame_id.txt'), selected_frm, fmt='%i')
ax1.bar(x_axis[selected_frm-1], [0.8]*selected_frm.shape[0], width=5)
fig1.savefig(os.path.join(args.result_path,'../mean_coverage_prob(with_selected_frames).png'))    # requested_coverage_curve

#### 绘制entropy的曲线
# fig2, ax2 = plt.subplots(figsize=(20,3))
# all_frm_mean_entropy = np.mean(all_frm_entropy, axis=1)
# ax2.plot(x_axis, all_frm_mean_entropy, c='k')
# ax2.scatter(x_axis[coverage_prob_of_all_frame > train_rbf_threshold], all_frm_mean_entropy[coverage_prob_of_all_frame > train_rbf_threshold], c='r', s=20)
# fig2.savefig(os.path.join(args.result_path,'../entropy_of_all_frame.png'))
plt.show()

# '''绘制每个像素位置的平均rbf'''
cap = cv2.VideoCapture(args.pre_video)  # 读取待标注数据
video_size=(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3)
# each_pixel_mean_rbf = np.mean(all_frm_rbf_response, axis=0)
# # 滑动窗处理所有的patch，先不考虑天空，只滑动下半张图片
# _patch_mask = np.zeros(video_size, dtype=np.uint8)
# cnt = 0
# for i in range(video_size[0]//2, video_size[0]-pred_res//2, pred_res):
#     for j in range(pred_res//2, video_size[1]-pred_res//2, pred_res):
#         pixel_v = max(min(255, int(each_pixel_mean_rbf[cnt]* 255)), 0)
#         _patch_mask = cv2.rectangle(_patch_mask, (j-pred_res,i-pred_res), (j+pred_res,i+pred_res), (pixel_v,pixel_v,pixel_v), thickness=-1) #thickness=-1 表示矩形框内颜色填充
#         cnt += 1
# _patch_mask = cv2.applyColorMap(_patch_mask, cv2.COLORMAP_JET)
# cv2.imshow("Average rbf response on image", _patch_mask)
# cv2.imwrite("average_rbf_response_on_image.png", _patch_mask)
# cv2.waitKey(0)

''' 可视化：同时画出视频帧上的rbf_response和聚类的结果，通过手动调整阈值观察（目的是区分类别边缘和没学会的小类别）'''
show_pca_vis = False    #是否在输出的视频上可视化降维之后的图
wait_time = 0
rbf_new_sigma = 0.07
pca = KernelPCA(n_components=2, kernel='rbf')
# pca = PCA(n_components=2)
videoWriter = cv2.VideoWriter(os.path.join(args.result_path,'../risk_scene_detection.avi'), cv2.VideoWriter_fourcc('M','J','P','G'), int(30), (video_size[1], video_size[0]//3*2+150))
frm_idx_of_feature = -1
while True:
    ret, full_img = cap.read() # 读入一帧图像
    if not ret: # 读完整段视频，退出
        print('Video end!')
        break
    frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if not frame_id in frm_ids:
        continue
    frm_idx_of_feature += 1
    if (frame_id % 2 == 0):
        continue
    _patch_mask = np.zeros((full_img.shape), dtype=np.uint8)
    _label_mask = np.zeros((full_img.shape), dtype=np.uint8)
    cnt = 0
    for i in range(video_size[0]//2, video_size[0]-pred_res//2, pred_res):
        for j in range(pred_res//2, video_size[1]-pred_res//2, pred_res):
            rbf_response = np.exp(np.log(all_frm_rbf_response[frm_idx_of_feature][cnt]) * (rbf_sigma ** 2) / (rbf_new_sigma ** 2))
            _patch_label = int(all_frm_cluster_label[frm_idx_of_feature][cnt])
            rbf_response = get_projected_risk(risk_coverage_sorted_list, rbf_response)
            u_color = int(min(255, max(0, (rbf_response) * 255)))
            '''用entropy代替risk可视化'''
            # entropy_val = ((all_frm_entropy[frm_idx_of_feature][cnt] - min_entropy) / (max_entoropy - min_entropy)) ** 6.0
            # u_color = int(min(255, max(0, (entropy_val) * 255)))
            '''用entropy代替risk可视化'''
            _patch_mask = cv2.rectangle(_patch_mask, (j-pred_res//2,i-pred_res//2), (j+pred_res//2,i+pred_res//2), (u_color,u_color,u_color), thickness=-1) #thickness=-1 表示矩形框内颜色填充
            # if (rbf_response < rbf_threshold):
            _label_mask = cv2.rectangle(_label_mask, (j-pred_res//2,i-pred_res//2), (j+pred_res//2,i+pred_res//2), anchor_color[_patch_label], thickness=-1) #thickness=-1 表示矩形框内颜色填充
            cnt += 1                 
    _patch_mask = cv2.GaussianBlur(_patch_mask, (101, 101), 0)
    _patch_mask = cv2.applyColorMap(_patch_mask, cv2.COLORMAP_JET)
    # alpha 为第一张图片的透明度，beta 为第二张图片的透明度 cv2.addWeighted 将原始图片与 mask 融合
    merged_full_img = cv2.addWeighted(full_img, 0.7, _patch_mask, 0.3, 0)
    merged_label_img = cv2.addWeighted(full_img, 0.7, _label_mask, 0.3, 0)
    merged_img = np.concatenate((full_img, merged_full_img, merged_label_img), axis=1)
    merged_img = cv2.resize(merged_img, (video_size[1], video_size[0]//3))
    cv2.putText(merged_img, "frame {}".format(frame_id-1), (30,30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
    # cv2.putText(merged_img, "rbf_sigma: {:.3f}".format(rbf_new_sigma), (30,90), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
    ############ 画出时间轴上rbf_response的曲线 ###########
    # 用opencv画散点图
    _colors = [(lambda x: (0,0,255) if x>rbf_threshold else (0,0,0))(x) for x in all_frm_mean_rbf]
    cvmat_all_frm_mean_rbf = drawScatter(x_axis[:frm_idx_of_feature+1], all_frm_mean_rbf[:frm_idx_of_feature+1], 
                                        _colors[:frm_idx_of_feature+1], isize=(150, 1400, 3), 
                                        xlim=[0, all_frm_mean_rbf.shape[0]+1], ylim=[all_frm_mean_rbf.min(), all_frm_mean_rbf.max()],
                                        draw_line=True)
    cv2.imshow("cvmat_all_frm_mean_rbf", cvmat_all_frm_mean_rbf)
    cvmat_all_frm_mean_rbf = cv2.resize(cvmat_all_frm_mean_rbf, (merged_img.shape[1], 150))
    
    ############ 画出降维后的锚点和所有patch ############
    # 读取已经保存好的features
    features_for_save = np.array([])
    # 检查是否有已经计算好的特征文件，如果有就不用重新计算
    if os.path.isfile(os.path.join(args.result_path.replace("cluster_results", "features"), str(frame_id)+".npy")):
        # [注意: features_for_save 共2+128维，前2维是patch对应的坐标，后128维是特征向量]
        features_for_save = np.load(os.path.join(args.result_path.replace("cluster_results", "features"), str(frame_id)+".npy"))
    feature_list_all = np.concatenate((anchor_feature_list, features_for_save[:,2:]))
    # Kernel PCA降成2维特征后，可视化类别簇
    if (frm_idx_of_feature == 0):
        pca = pca.fit(feature_list_all)
    x_pca = pca.transform(feature_list_all)
    x_pca_anchors = x_pca[:anchor_feature_list.shape[0]]
    x_pca_case_study = x_pca[anchor_feature_list.shape[0]:]
    ######### 用opencv画锚点和case features
    c_rbf = []
    c_lab = []
    for i in range(x_pca_anchors.shape[0]):
        c_lab.append(anchor_color[int(cluster_model.labels_[i])])
        c_rbf.append((128,128,128))
    for i in range(x_pca_case_study.shape[0]):
        c_lab.append(anchor_color[int(all_frm_cluster_label[frm_idx_of_feature][i])])
        _i, _j = list(map(int,features_for_save[i, :2]))
        c_rbf.append((int(_patch_mask[_i,_j,0]), int(_patch_mask[_i,_j,1]), int(_patch_mask[_i,_j,2])))
    cvmat_rbf_img = drawScatter(np.concatenate((x_pca_anchors[:,0], x_pca_case_study[:,0])), 
                                np.concatenate((x_pca_anchors[:,1], x_pca_case_study[:,1])), 
                                c_rbf,
                                isize=(400, 400, 3), 
                                xlim=[-0.12, 0.13], #xlim=[-0.9, 0.95],   
                                ylim=[-0.1, 0.125],
                                anchor_num=x_pca_anchors[:,0].shape[0]) #ylim=[-0.8, 0.95])
    cvmat_lab_img = drawScatter(np.concatenate((x_pca_anchors[:,0], x_pca_case_study[:,0])), 
                                np.concatenate((x_pca_anchors[:,1], x_pca_case_study[:,1])), 
                                c_lab, 
                                isize=(400, 400, 3), 
                                xlim=[-0.12, 0.13], #xlim=[-0.9, 0.95],   
                                ylim=[-0.1, 0.125],
                                anchor_num=x_pca_anchors[:,0].shape[0]) #ylim=[-0.8, 0.95])
    cv2.imshow("rbf_img", cvmat_rbf_img)
    cv2.imshow("lab_img", cvmat_lab_img)

    ## 绘图 ##
    margin_img = np.zeros((400, 80, 3), dtype=np.uint8) + 255 # 空白
    cvmat_pca_img = np.concatenate((margin_img, cvmat_rbf_img, margin_img, cvmat_lab_img, margin_img), axis=1)
    cvmat_pca_img = cv2.resize(cvmat_pca_img, (merged_img.shape[1], merged_img.shape[0]))
    merged_img = np.concatenate((merged_img, cvmat_all_frm_mean_rbf, cvmat_pca_img), axis=0)

    cv2.imshow("image", merged_img)
    print(frame_id)
    if (merged_img.shape != (video_size[0]//3*2+150, video_size[1], 3)):
        print("error merge_img shape: {} vs {}".format(merged_img.shape, (video_size[0]//3*2+200, video_size[1], 3)))
        videoWriter.release()
        break
    videoWriter.write(merged_img)
    _key = cv2.waitKey(wait_time)
    if _key == ord('z'):
        wait_time = 1 - wait_time
    elif _key == ord('q'):
        rbf_new_sigma -= 0.02
    elif _key == ord('e'):
        rbf_new_sigma += 0.02
videoWriter.release()
print("每帧平均risk指标下，数据集整体coverage: {}".format(np.count_nonzero(coverage_prob_of_all_frame <= 0.85)/coverage_prob_of_all_frame.shape[0]))
print("patch_risk指标下，数据集整体coverage: {}".format(np.count_nonzero(all_frm_rbf_response <= rbf_threshold)/(all_frm_rbf_response.shape[0]*all_frm_rbf_response.shape[1])))