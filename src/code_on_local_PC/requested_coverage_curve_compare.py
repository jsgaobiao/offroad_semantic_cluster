'''
Author: your name
Date: 2021-06-26 16:13:29
LastEditTime: 2021-06-26 17:04:30
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \risk_coverage_visualization\requested_coverage_curve_compare.py
'''

import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(20,3))
'''读入数据'''
all_frm_rbf_response = []
with open(u"mean_coverage_prob(三段对比).csv", "r") as f:
    _ = f.readline()
    line_m145678 = f.readline().strip().split(',')
    line_m145678 = np.array(list(map(float, line_m145678[:-1])))
    _ = f.readline()
    _ = f.readline()
    line_m678 = f.readline().strip().split(',')
    line_m678 = np.array(list(map(float, line_m678[:-1])))
    _ = f.readline()
    _ = f.readline()
    line_mNone = f.readline().strip().split(',')
    line_mNone = np.array(list(map(float, line_mNone[:-1])))

ax.plot(np.arange(0,len(line_m145678)), line_m145678,  label='subset1-A')
ax.plot(np.arange(0,len(line_m678)), line_m678,  label='subset1-B')
ax.plot(np.arange(0,len(line_mNone)), line_mNone,  label='subset1-C')
_step = 23
for i in range(0, len(line_mNone)-_step, _step):
    xbar = i + _step//2
    _c = 'w'
    avg_m145678 = np.mean(line_m145678[i:i+_step])
    avg_m678 = np.mean(line_m678[i:i+_step])
    avg_mNone = np.mean(line_mNone[i:i+_step])
    if (avg_m145678 - min(avg_mNone, avg_m678) < 0.05):
        _c = 'gold'
    ax.bar(xbar, 1, color=_c, width=_step, alpha=0.3, edgecolor=None)
ax.set_ylim(0.65, 1.0)
plt.show()
