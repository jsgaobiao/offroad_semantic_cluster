'''
Author: Gao Biao
Date: 2020-12-17 10:49:39
LastEditTime: 2021-01-26 13:20:08
Description: 测试不同epoch model的性能，选择最好的一个
FilePath: /offroad_semantic_cluster/src/run_eval.py
'''
import os

command = "CUDA_VISIBLE_DEVICES=1 python patch_cluster.py --data_folder /home/gaobiao/Documents/offroad_semantic_cluster/data \
        --result_path /home/gaobiao/Documents/offroad_semantic_cluster/src/results \
        --model_path {0} \
        --subset train_fine_anno \
        --note 0124_BG320_test1_to_all/train \
        --kmeans {1} \
        --pre_video /home/gaobiao/Documents/offroad_semantic_cluster/data/1_cut.mp4 \
        --in_channel 6 \
        --background 320"
model_path = "/home/gaobiao/Documents/offroad_semantic_cluster/src/trained_models/0116_RGB_test1_dataAug_BG320__end2end_nce_16_batch_size_32_fdim_128/"
kmeans = 6
# model_name = "ckpt_epoch_400_loss_0.2763.pth"
# 遍历epoch找到最好的model
model_list = os.listdir(model_path)
model_list.sort()
print(model_list)
for m in model_list:
    os.system(command.format(model_path+m, kmeans))

# 测试不同的kmeans
# for k in range(3,8):
#     os.system(command.format(model_path+model_name, k) + "| tee -a results/0123_train_to_all/test2/cluster.log")


