import torch
import sys
import torch.nn as nn
import numpy as np
torch.manual_seed(0)

gt_ratio_data = np.load('monitor/gt_ratio_list5_log_exp_train_stage1.npy')
pred_ratio_data = np.load('monitor/pred_ratio_list5_log_exp_train_stage1.npy')
weight_dataset = np.load('feature_tmp5/feature_weight_train.npy')
profit_dataset = np.load('feature_tmp5/feature_profit_train.npy')
label_dataset = np.load('monitor/label_list5_log_exp_train_stage1.npy')

pred_block = []
gt_block = []
weight_block = []
profit_block = []
for i in range(len(gt_ratio_data)):
    for k in range(5):
        gt_block.append(gt_ratio_data[i][k])
        pred_block.append(pred_ratio_data[i][k])
        weight_block.append(weight_dataset[i][k])
        profit_block.append(profit_dataset[i][k])

w = torch.tensor(weight_block, dtype=torch.float32)
p = torch.tensor(profit_block, dtype=torch.float32)
q = torch.tensor(gt_block, dtype = torch.float32)
z = torch.tensor(pred_block, dtype=torch.float32)

# Parameters to learn
a = torch.nn.Parameter(torch.randn(()))
b = torch.nn.Parameter(torch.randn(()))
d = torch.nn.Parameter(torch.randn(()))


optimizer = torch.optim.Adam([a, b, d], lr=1e-1)

for step in range(5000):
    optimizer.zero_grad()
    f_pred = a * torch.log(torch.abs(b * (p / w))) + d 

    loss = torch.mean(torch.abs((f_pred - z)**2))
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f'Step {step}, Loss: {loss.item():.6f}')

a = a.detach().numpy()
b = b.detach().numpy()
d = d.detach().numpy()
print(a, b, d)
