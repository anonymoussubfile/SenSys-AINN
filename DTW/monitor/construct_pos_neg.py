import os
import random
import numpy as np
from math import inf
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from torchaudio.sox_effects import apply_effects_tensor
import torchaudio.functional as AF



mode = 'train'
phi = np.load("{}_phi_int_list_le.npy".format(mode))
c = np.load("{}_c_list_le.npy".format(mode))
dp = np.load("{}_dp_list_le.npy".format(mode))
cs = np.load("{}_cs_list_le.npy".format(mode))
l1 = np.load("{}_l1_list_le.npy".format(mode))
l2 = np.load("{}_l2_list_le.npy".format(mode))
pred = np.load("{}_pred_list_le.npy".format(mode))
gt = np.load("{}_gt_list_le.npy".format(mode))
ref_enc = np.load("{}_ref_enc_list_le.npy".format(mode))

pred_i = pred.astype(np.int64)
gt_i   = gt.astype(np.int64)
mask = (pred_i == gt_i)

pos_phi = phi[mask]
pos_c = c[mask]
pos_dp = dp[mask]
pos_pred = pred[mask]
pos_gt = gt[mask]
pos_cs = cs[mask]
pos_l1 = l1[mask]
pos_l2 = l2[mask]
pos_ref = ref_enc[mask]

neg_phi = phi[~mask]
neg_c = c[~mask]
neg_dp = dp[~mask]
neg_pred = pred[~mask]
neg_gt = gt[~mask]
neg_cs = cs[~mask]
neg_l1 = l1[~mask]
neg_l2 = l2[~mask]
neg_ref = ref_enc[~mask]


np.save('pos_{}_phi_int_list_stack.npy'.format(mode), pos_phi)
np.save('pos_{}_c_list_stack.npy'.format(mode), pos_c)
np.save('pos_{}_dp_list_stack.npy'.format(mode), pos_dp)
np.save('pos_{}_pred_list_stack.npy'.format(mode), pos_pred)
np.save('pos_{}_gt_list_stack.npy'.format(mode), pos_gt)
np.save('pos_{}_l1_list_stack.npy'.format(mode), pos_l1)
np.save('pos_{}_l2_list_stack.npy'.format(mode), pos_l2)
np.save('pos_{}_cs_list_stack.npy'.format(mode), pos_cs)
np.save('pos_{}_ref_enc_list_stack.npy'.format(mode), pos_ref)

np.save('neg_{}_phi_int_list_stack.npy'.format(mode), neg_phi)
np.save('neg_{}_c_list_stack.npy'.format(mode), neg_c)
np.save('neg_{}_dp_list_stack.npy'.format(mode), neg_dp)
np.save('neg_{}_pred_list_stack.npy'.format(mode), neg_pred)
np.save('neg_{}_gt_list_stack.npy'.format(mode), neg_gt)
np.save('neg_{}_l1_list_stack.npy'.format(mode), neg_l1)
np.save('neg_{}_l2_list_stack.npy'.format(mode), neg_l2)
np.save('neg_{}_cs_list_stack.npy'.format(mode), neg_cs)
np.save('neg_{}_ref_enc_list_stack.npy'.format(mode), neg_ref)

