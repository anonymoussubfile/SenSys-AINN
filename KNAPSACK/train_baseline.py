import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle
import numpy as np
import sys
import random
from utils import progress_bar
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
import copy
best_acc = 0
torch.manual_seed(100)
num_sample = 1000

best_acc = 0


class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, mode = 'train'):
        '''
        #data generation process, commenting out and directly using the saved version
        collections = [m for m in range(1, 20)]
        collections = np.array(collections)
        self.weight_dataset = []
        self.profit_dataset = []
        self.value = []
        self.data_str = []
        self.ind_dataset = []
        i = 0
        while len(self.weight_dataset) < 50000:
            i += 1
            #print(i)
            random.seed(i)
            weight_l = random.sample(collections.tolist(), 5)
            random.seed(i + 50000)
            profit_l = random.sample(collections.tolist(), 5)
            key_str = str(weight_l[0]) + '-' + str(weight_l[1]) + '-' + str(weight_l[2]) + '-' + str(weight_l[3]) + '-' + str(weight_l[4]) + str(profit_l[0]) + '-' + str(profit_l[1]) + '-' + str(profit_l[2]) + '-' + str(profit_l[3]) + '-' + str(profit_l[4])
            if key_str in self.data_str:
                continue
            print(i)
            self.weight_dataset.append(np.array(weight_l))
            self.profit_dataset.append(np.array(profit_l))
            self.data_str.append(key_str)


            W = 15
            finalvalue = 0
            fraction = np.array(profit_l) / np.array(weight_l)

            sort_ind = np.argsort(-1 * fraction)
            ind_list = [0, 0, 0, 0, 0]
            for ind in sort_ind:
                if weight_l[ind] <= W:
                    W -= weight_l[ind]
                    finalvalue += profit_l[ind]
                    ind_list[ind] = 1
                else:
                    finalvalue += profit_l[ind] * W / weight_l[ind]
                    ind_list[ind] = W / weight_l[ind]
                    break
            self.ind_dataset.append(ind_list)


            self.value.append(finalvalue)

        if mode == 'train':
            self.weight_dataset = np.array(self.weight_dataset)[:40000]
            self.profit_dataset = np.array(self.profit_dataset)[:40000]
            self.value = np.array(self.value)[:40000]
            self.ind_dataset = np.array(self.ind_dataset)[:40000]
            with open('feature_tmp5/feature_weight_train.npy', 'wb') as f:
                np.save(f, self.weight_dataset)
            with open('feature_tmp5/feature_profit_train.npy', 'wb') as f:
                np.save(f, self.profit_dataset)
            with open('feature_tmp5/label_train.npy', 'wb') as f:
                np.save(f, self.value)
            with open('feature_tmp5/label_ind_train.npy', 'wb') as f:
                np.save(f, self.ind_dataset)

        else:
            self.weight_dataset = np.array(self.weight_dataset)[40000:]
            self.profit_dataset = np.array(self.profit_dataset)[40000:]
            self.value = np.array(self.value)[40000:]
            self.ind_dataset = np.array(self.ind_dataset)[40000:]
            with open('feature_tmp5/feature_weight_test.npy', 'wb') as f:
                np.save(f, self.weight_dataset)
            with open('feature_tmp5/feature_profit_test.npy', 'wb') as f:
                np.save(f, self.profit_dataset)
            with open('feature_tmp5/label_test.npy', 'wb') as f:
                np.save(f, self.value)
            with open('feature_tmp5/label_ind_test.npy', 'wb') as f:
                np.save(f, self.ind_dataset)
        '''


        if mode == 'train':
            self.weight_dataset = np.load('feature/feature_weight_{}.npy'.format(mode))[:num_sample]
            self.profit_dataset = np.load('feature/feature_profit_{}.npy'.format(mode))[:num_sample]
            self.value = np.load('feature/label_{}.npy'.format(mode))[:num_sample]
            self._ind_dataset = np.load('feature/label_ind_{}.npy'.format(mode))[:num_sample]
            
            self.ind_dataset = []
            self.r_dataset = []
            for label_ind in self._ind_dataset:
                new_label_ind = []
                empty_r = [0, 0, 0, 0, 0, 0]
                r_ind = 5
                for j, k in enumerate(label_ind):
                    if 0 < k < 1:
                        r_ind = j
                    if k < 1:
                        new_label_ind.append(0)
                    else:
                        new_label_ind.append(1)
                self.ind_dataset.append(new_label_ind)
                empty_r[r_ind] = 1
                self.r_dataset.append(empty_r)
            self.ind_dataset = np.array(self.ind_dataset)
         
        else:
            self.weight_dataset = np.load('feature/feature_weight_{}.npy'.format(mode))
            self.profit_dataset = np.load('feature/feature_profit_{}.npy'.format(mode))
            self.value = np.load('feature/label_{}.npy'.format(mode))
            self._ind_dataset = np.load('feature/label_ind_{}.npy'.format(mode))
            
            self.ind_dataset = []
            self.r_dataset = []
            for label_ind in self._ind_dataset:
                new_label_ind = []
                r_ind = 5
                empty_r = [0, 0, 0, 0, 0, 0]
                for j, k in enumerate(label_ind):
                    if 0 < k < 1:
                        r_ind = j
                    if k < 1:
                        new_label_ind.append(0)
                    else:
                        new_label_ind.append(1)
                self.ind_dataset.append(new_label_ind)
                empty_r[r_ind] = 1
                self.r_dataset.append(empty_r)
            self.ind_dataset = np.array(self.ind_dataset)
            
    def __len__(self):
        return len(self.weight_dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {'weight': self.weight_dataset[idx] / 15, 'profit': self.profit_dataset[idx], 'label': self.value[idx], 'limit': np.array([15]), 'ind': self.ind_dataset[idx], 'r': np.array(self.r_dataset[idx])}
        return sample

class FractionalKnapsackMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(10, 64*4*2),
            nn.ReLU(),
            nn.Linear(64*4*2, 64*4*2),
            nn.ReLU(),
            nn.Linear(64*4*2, 64*4*2),
            nn.ReLU(),
            nn.Linear(64*4*2, 64*4*2),
            nn.ReLU(),
            nn.Linear(64*4*2, 64*4*2),
            nn.ReLU(),
            nn.Linear(64*4*2, 64*4*2),
            nn.ReLU()
        )
        self.pick_head = nn.Linear(64*4*2, 5)   # k_mask (0 or 1 per item)
        self.partial_head = nn.Linear(64*4*2, 6)  # r (one-hot)

    def forward(self, weights, profits):
        x = torch.cat([weights, profits], dim=1)  # shape (B, 10)
        h = self.hidden(x)

        k_mask_logits = self.pick_head(h)
        r_logits = self.partial_head(h)

        k_mask = torch.sigmoid(k_mask_logits)         # continuous ∈ (0,1)
        r_softmax = F.softmax(r_logits, dim=1)        # one-hot style
        return k_mask, r_softmax




def train(model_ori, criterion, criterion2, optim, dataset):
    model_ori.train()
    train_loss = 0

    cnt = 0
    for batch_idx, input_data in enumerate(dataset):
        weights, profits, limits, labels, inds, r_gt = input_data['weight'].to('cuda', dtype=torch.float), input_data['profit'].to('cuda', dtype=torch.float), input_data['limit'].to('cuda', dtype=torch.float), input_data['label'].to('cuda', dtype=torch.float), input_data['ind'].to('cuda', dtype=torch.float), input_data['r'].to('cuda', dtype=torch.float)
        k_logits, r_logits = model_ori(weights, profits)
        loss = criterion(k_logits, inds) + criterion2(r_logits, r_gt)
        optim.zero_grad()
        loss.backward()
        optim.step()

        train_loss += loss.item()

        progress_bar(batch_idx, len(dataset), 'T2: %.3f'
                     % (train_loss/(batch_idx+1))) 


def test(model_ori, criterion, optim, dataset, mode = 'train'):
    global best_acc
    model_ori.eval()
    correct1, total1 = 0, 0   

    cnt = 0
    for batch_idx, input_data in enumerate(dataset):
        weights, profits, limits, labels, inds, r_gt = input_data['weight'].to('cuda', dtype=torch.float), input_data['profit'].to('cuda', dtype=torch.float), input_data['limit'].to('cuda', dtype=torch.float), input_data['label'].to('cuda', dtype=torch.float), input_data['ind'].to('cuda', dtype=torch.float), input_data['r'].to('cuda', dtype=torch.float)
        k_logits, r_logits = model_ori(weights, profits)

        for i in range(k_logits.size()[0]):
            pred = torch.round(k_logits[i])
            r_pred = torch.round(r_logits[i])
            res = torch.equal(pred.type(torch.int64), inds[i].type(torch.int64))
            res2 = torch.equal(r_gt[i].type(torch.int64), r_pred.type(torch.int64))
            correct1 += res * res2
            total1 += 1

        progress_bar(batch_idx, len(dataset), 'acc: %.3f'
                     % (correct1/ total1))

    if correct1 / total1 > best_acc and mode != 'eval': #train_loss < best_loss:
        print('saving...')
        best_acc = correct1 / total1 #train_loss
        state_dict = {'net': model_ori.state_dict()}
        torch.save(state_dict, 'baseline_fractional_{}.pth'.format(num_sample))


model_ori = FractionalKnapsackMLP()
total_params = sum(p.numel() for p in model_ori.parameters())
print(f"Total parameters: {total_params}")

model_ori = model_ori.to('cuda')

train_dataset = FeatureDataset('train')
train_dataloader = DataLoader(train_dataset, batch_size = 128, shuffle = True)

test_dataset = FeatureDataset('test') #this is actually validation dataset
test_dataloader = DataLoader(test_dataset, batch_size = 128, shuffle = False)

opt = torch.optim.Adam(model_ori.parameters(), lr = 0.001) 

criterion = nn.BCELoss()
criterion2 = nn.CrossEntropyLoss()

for i in range(200):
    print('Epoch: ', i)
    train(model_ori,  criterion, criterion2, opt, train_dataloader)
    test(model_ori,  criterion, opt, test_dataloader, 'test')


eval_dataset = FeatureDataset('eval')
eval_dataloader = DataLoader(eval_dataset, batch_size = 128, shuffle = False)
test(model_ori, criterion, opt, eval_dataloader, 'eval')
