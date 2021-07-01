import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import os

anchor_color = [(0,0,255), (0,255,0), (255,0,0), (0,255,255), (255,0,255), (255,255,0), (220,220,220), (31,102,156), (80,127,255), (140,230,240), (127,255,0), (158,168,3), (255,144,30), (214,112,218)]
anchor_label = [u"0:路",u"1:石头",u"2:植物",u"3:路边",u"4:建筑",u"5:碎石",u"6:水泥堆",u"7:木材",u"8:草泥/落叶","9:"]
hex_c = []
for c in anchor_color:
  hex_c.append('#%02x%02x%02x' % (c[2], c[1], c[0]))
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
#有中文出现的情况，需要u'内容'


# startTime = 781
# cap0 = cv2.VideoCapture("/home/gaobiao/Documents/offroad_semantic_cluster/src/results/0324_OOD_train_to_all/train/ood_winter.avi") 
# cap1 = cv2.VideoCapture("/home/gaobiao/Documents/offroad_semantic_cluster/src/results/0324_OOD_train_to_all/train/uncertaintyMapping.avi")
# video_size_0=(cap0.get(cv2.CAP_PROP_FRAME_WIDTH),cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
# tot_frames_0 = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT))
# video_size_1=(cap1.get(cv2.CAP_PROP_FRAME_WIDTH),cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
# tot_frames_1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))

# cap = cv2.VideoCapture("/home/gaobiao/Documents/offroad_semantic_cluster/data/0.avi")  # 读取待标注数据
# video_pred_dir = "/home/gaobiao/Documents/offroad_semantic_cluster/src/results/0324_OOD_train_to_all/train/video_pred"
# fps=cap.get(cv2.CAP_PROP_FPS)
# video_size=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# print(video_size, tot_frames)
# print(video_size_0, tot_frames_0)
# print(video_size_1, tot_frames_1)

# cap0.set(cv2.CAP_PROP_POS_FRAMES, int(startTime))
# cap.set(cv2.CAP_PROP_POS_FRAMES, int(startTime))
# wvideo_size = (int(video_size[0]), int(video_size[1]))    # [1920+1920]//2, [1920+1088+832]//2
# vWriter = cv2.VideoWriter("merge_ood.avi", cv2.VideoWriter_fourcc('M','P','4','2'), 30, wvideo_size)
# ratio = (tot_frames_1) / (tot_frames_0 - startTime)

# frame_id_0 = frame_id_1 = -1
# while True:
#     ret, ood_img = cap0.read()
#     ret_, orgin_img = cap.read() # 读入一帧图像
#     if (not ret) or (not ret_): # 读完整段视频，退出
#         break
#     frame_id_0 += 1
#     if int(frame_id_0 * ratio) > frame_id_1:
#         ret, map_img = cap1.read()
#         if not ret: # 读完整段视频，退出
#             break
#         frame_id_1 += 1
#     # 读入对应的预测结果
#     pred_frame = cv2.imread(os.path.join(video_pred_dir, str(frame_id_0+startTime+1)+'_pred_all.png'))
#     resized_ood_img = cv2.resize(ood_img, (int(video_size_0[0]//2), int(video_size_0[1]//2)))
#     resized_origin_img = cv2.resize(orgin_img, (int(video_size[0]//2), int(video_size[1]//2)))
#     resized_pred_img = cv2.resize(pred_frame, (int(pred_frame.shape[1]//2), int(pred_frame.shape[0]//2)))
#     resized_map_img = cv2.resize(map_img, (int(832//2), int(video_size_0[1]//2)))

#     merge_img_0 = np.concatenate((resized_origin_img, resized_pred_img), axis=1)
#     merge_img_1 = np.concatenate((resized_ood_img, resized_map_img), axis=1)
#     mrege_img_all = np.concatenate((merge_img_0, merge_img_1), axis=0)
#     vWriter.write(mrege_img_all)
#     print(frame_id_0)
    
# vWriter.release()
# cap.release()
# cap0.release()
# cap1.release()






######################### 画出每一帧有多少扩展锚点（直方图） #########################

# f=open("../data/train_extend/extend_anchors_of_frame.txt","r")
# n = int(f.readline().strip())
# dat = np.zeros(5030)
# for i in range(n):
#   frmID, a_num = list(map(int, f.readline().strip().split()))
#   dat[frmID] = a_num
#   print(frmID, a_num)
#   _ = f.readline()

# plt.figure(figsize=(30, 5))
# plt.bar(range(len(dat)-800), dat[800:])
# plt.savefig('anchor_number_at_frames.png', dpi=600)

######################### 画出哪些帧有锚点 #########################

# anchor_dict = np.load("../data/test1_add_1/anchors_annotation.npy", allow_pickle=True).item()
# dat = np.zeros(3250)
# for _f_id in sorted(anchor_dict.keys()):
#     dat[_f_id] = 3
# plt.figure(figsize=(30, 5))
# plt.bar(range(3250), dat, width=8, color='r')
# plt.savefig('anchor_location(test1_add_1).png', dpi=600)
# gb = 1


anchor_lab_mask_1 = [1,4,5,6,7,8]
anchor_lab_mask_2 = [6,7,8]
anchor_dict = np.load("../data/train_fine_anno/anchors_annotation.npy", allow_pickle=True).item()
dat3 = np.zeros(5030)
dat1 = np.zeros(5030)
dat2 = np.zeros(5030)
bar_color = ['g'] * 5030
bar_color1 = ['b'] * 5030
bar_color2 = ['r'] * 5030
for _f_id in sorted(anchor_dict.keys()):
    for _ac_id in range(len(anchor_dict[_f_id])-1, -1, -1):
        if anchor_dict[_f_id][_ac_id][2] not in anchor_lab_mask_1:
            dat1[_f_id] = 1
        elif anchor_dict[_f_id][_ac_id][2] not in anchor_lab_mask_2:
            dat2[_f_id] = 2
        else:
            dat3[_f_id] = 3
plt.figure(figsize=(30, 5))
plt.bar(range(5030), dat1, width=8, color=bar_color1)
plt.savefig('anchor_location(mask145678).png', dpi=600)
plt.bar(range(5030), dat2, width=8, color=bar_color2)
plt.bar(range(5030), dat1, width=8, color=bar_color1)
plt.savefig('anchor_location(mask678).png', dpi=600)
plt.bar(range(5030), dat3, width=8, color=bar_color)
plt.bar(range(5030), dat2, width=8, color=bar_color2)
plt.bar(range(5030), dat1, width=8, color=bar_color1)
plt.savefig('anchor_location(mask_null).png', dpi=600)
print("additional_anchor_num:",np.count_nonzero(dat1), np.count_nonzero(dat2), np.count_nonzero(dat3))
np.savetxt('anchor_location(mask145678).txt', np.nonzero(dat1), fmt='%i')
np.savetxt('anchor_location(mask678).txt', np.nonzero(dat2), fmt='%i')
np.savetxt('anchor_location(mask_null).txt', np.nonzero(dat3), fmt='%i')
exit(0)





######################### 画出扩展锚点前后，各类别锚点的比例 #########################
# f=open("../data/train_extend/extend_anchors_of_frame.txt","r")
# n = int(f.readline().strip())
# cnt_extend = np.zeros(10)
# cnt = np.zeros(10)
# for i in range(n):
#   frmID, a_num = list(map(int, f.readline().strip().split()))
#   print(frmID, a_num)
#   dat = list(map(int, f.readline().strip().split()))
#   for j in range(len(dat)//4):
#     _t = dat[j*4+3]
#     cnt_extend[_t] += 1

# anchor_dict = np.load("../data/train_fine_anno/anchors_annotation.npy", allow_pickle=True).item()
# for i in anchor_dict.keys():
#   for j in range(len(anchor_dict[i])):
#     cnt[anchor_dict[i][j][2]] += 1

# anchor_color = [(0,0,255), (0,255,0), (255,0,0), (0,255,255), (255,0,255), (255,255,0), (220,220,220), (31,102,156), (80,127,255), (140,230,240), (127,255,0), (158,168,3), (255,144,30), (214,112,218)]
# hex_c = []
# for c in anchor_color:
#   hex_c.append('#%02x%02x%02x' % (c[2], c[1], c[0]))

# fig, axs = plt.subplots(1,2,figsize=(10,5))
# plt.subplot(1,2,1)
# plt.bar(range(9), cnt[:9], color=hex_c[:9])
# plt.subplot(1,2,2)
# plt.bar(range(9), cnt_extend[:9], color=hex_c[:9])
# plt.savefig('anchor_type_bar.png', dpi=600)


######################### 画出不确定性在各类别样本（锚点）上的占比 #########################
# 把y轴转化为百分比。
def to_percent(y, position):
    return str(100 * y) + '%'
fig, ax = plt.subplots()
fig1, ax1 = plt.subplots()
# anchor_uncertainty_dict, key=frame_id, value=[uncertainty]
anchor_uncertainty_dict = np.load("/home/gaobiao/Documents/offroad_semantic_cluster/src/results/0324_OOD_train_to_all/train/uncertainty_txt/anchor_uncertainty_dict__entropy.npy", allow_pickle=True).item()
num_bins = 10
min_unc = 1e10
max_unc = -4
for i in anchor_uncertainty_dict.keys():
    if len(anchor_uncertainty_dict[i]) == 0: continue
    min_unc = min(min_unc, np.min(anchor_uncertainty_dict[i]))
    max_unc = max(max_unc, np.max(anchor_uncertainty_dict[i]))
x_axis = np.arange(min_unc, max_unc, (max_unc-min_unc)/num_bins)
# 各类别锚点的不确定性：在不同区间不确定性的集合中的占比
stack_uncertainty = np.zeros((len(anchor_uncertainty_dict.keys()), num_bins))
stack_uncertainty_rate = np.zeros((len(anchor_uncertainty_dict.keys()), num_bins))
# 各类别锚点的不确定性：在本类别中的占比
accumulate_uncertainty_rate = np.zeros((len(anchor_uncertainty_dict.keys()), num_bins))
uncertainty_rate = np.zeros((len(anchor_uncertainty_dict.keys()), num_bins))
for i in anchor_uncertainty_dict.keys():
    if len(anchor_uncertainty_dict[i]) == 0: continue
    # 遍历所有uncertainty的值，统计各区间段的分布
    for j in range(len(anchor_uncertainty_dict[i])):
        _block = int((anchor_uncertainty_dict[i][j] - min_unc) / (max_unc - min_unc + 1e-5) * num_bins)
        stack_uncertainty[i][_block] += 1
    # weight_ = [1./ len(anchor_uncertainty_dict[i])] * len(anchor_uncertainty_dict[i])
    # y_height, bins_limits, patches = ax.hist(anchor_uncertainty_dict[i], num_bins, label=i, alpha = 0.5, weights=weight_)
for i in range(stack_uncertainty.shape[0]-1):
    for j in range(stack_uncertainty.shape[1]):
        accumulate_uncertainty_rate[i][j] = np.sum(stack_uncertainty[i][:j]) / np.sum(stack_uncertainty[i])
        uncertainty_rate[i][j] = stack_uncertainty[i][j] / np.sum(stack_uncertainty[i])
        stack_uncertainty_rate[i][j] = stack_uncertainty[i][j] / np.sum(stack_uncertainty[:,j])
    ax.plot(x_axis, accumulate_uncertainty_rate[i], color=hex_c[i], label=anchor_label[i])
    ax1.plot(x_axis, uncertainty_rate[i], color=hex_c[i], label=anchor_label[i])
# 绘制堆叠图
# ax.stackplot(x_axis, stack_uncertainty_rate[0], 
#                 stack_uncertainty_rate[1], 
#                 stack_uncertainty_rate[2],  
#                 stack_uncertainty_rate[3],
#                 stack_uncertainty_rate[4],
#                 stack_uncertainty_rate[5],
#                 stack_uncertainty_rate[6],
#                 stack_uncertainty_rate[7],
#                 stack_uncertainty_rate[8],
#                 colors=hex_c[:9])
# 添加分布曲线
# for i in range(y_height.shape[0]):
#     ax.plot(bins_limits[1:], y_height[i], '--')
ax.legend()
ax1.legend()
fig.savefig('anchor_uncertainty_dict__entropy.png', dpi=600)
fig1.savefig('anchor_uncertainty_rate__entropy.png', dpi=600)