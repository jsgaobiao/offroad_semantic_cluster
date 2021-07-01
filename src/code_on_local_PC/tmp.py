
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5,5))
'''读入数据'''
all_frm_rbf_response = []
with open("results/0429_eval_cluster_by_uncertainty/train2train/rbf_response.csv", "r") as f:
    # frm_id, rbf...
    for line in f:
        dat = line.strip().split(',')
        dat = list(map(float, dat[:-1]))
        all_frm_rbf_response.append(dat[1:])
all_frm_rbf_response = np.array(all_frm_rbf_response)
with open("results/0429_eval_cluster_by_uncertainty/train2train/trainset_risk_coverage.csv", 'r') as f:
    train_risk_coverage = f.readline().strip().split(',')
    train_risk_coverage = list(map(float, train_risk_coverage[:-1]))
coverage_to_coverage_curve = np.zeros(99)
for i in range(99):
    coverage_to_coverage_curve[i] = np.count_nonzero(all_frm_rbf_response[all_frm_rbf_response < train_risk_coverage[i]]) / np.count_nonzero(all_frm_rbf_response) * 100
ax.plot(np.arange(1,100), coverage_to_coverage_curve, c='g', label='subset3')


all_frm_rbf_response = []
with open("results/0528_trainset_BG320_labMask678/train2train/rbf_response.csv", "r") as f:
    # frm_id, rbf...
    for line in f:
        dat = line.strip().split(',')
        dat = list(map(float, dat[:-1]))
        all_frm_rbf_response.append(dat[1:])
all_frm_rbf_response = np.array(all_frm_rbf_response)
with open("results/0528_trainset_BG320_labMask678/train2train/trainset_risk_coverage.csv", 'r') as f:
    train_risk_coverage = f.readline().strip().split(',')
    train_risk_coverage = list(map(float, train_risk_coverage[:-1]))
coverage_to_coverage_curve = np.zeros(99)
for i in range(99):
    coverage_to_coverage_curve[i] = np.count_nonzero(all_frm_rbf_response[all_frm_rbf_response < train_risk_coverage[i]]) / np.count_nonzero(all_frm_rbf_response) * 100
ax.plot(np.arange(1,100), coverage_to_coverage_curve, c='r', label='subset2')


all_frm_rbf_response = []
with open("results/0528_trainset_BG320_labMask145678/train2train/rbf_response.csv", "r") as f:
    # frm_id, rbf...
    for line in f:
        dat = line.strip().split(',')
        dat = list(map(float, dat[:-1]))
        all_frm_rbf_response.append(dat[1:])
all_frm_rbf_response = np.array(all_frm_rbf_response)
with open("results/0528_trainset_BG320_labMask145678/train2train/trainset_risk_coverage.csv", 'r') as f:
    train_risk_coverage = f.readline().strip().split(',')
    train_risk_coverage = list(map(float, train_risk_coverage[:-1]))
coverage_to_coverage_curve = np.zeros(99)
for i in range(99):
    coverage_to_coverage_curve[i] = np.count_nonzero(all_frm_rbf_response[all_frm_rbf_response < train_risk_coverage[i]]) / np.count_nonzero(all_frm_rbf_response) * 100
ax.plot(np.arange(1,100), coverage_to_coverage_curve, c='b', label='subset1')

plt.legend()
plt.show()
fig.savefig('train_coverage_to_test_coverage.png')
