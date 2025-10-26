import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

def keep_leftmost_one(k: torch.Tensor) -> torch.Tensor:
    cumsum = torch.cumsum(k, dim=1)
    mask_first_one = (cumsum == 1) & (k == 1)
    return mask_first_one.to(k.dtype)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    
set_seed(0)

pos_output = np.load("monitor/c_list5_log_exp_train_stage3.npy")
neg_output = np.load("monitor/_c_list5_log_exp_train_stage3.npy")


pos = np.load("monitor/k_list5_log_exp_train_stage3.npy") 
neg = np.load("monitor/_k_list5_log_exp_train_stage3.npy")

pos_extra = np.load("monitor/kl_list5_log_exp_train_stage3.npy")
neg_extra = np.load("monitor/_kl_list5_log_exp_train_stage3.npy")

assert pos.shape[0] == pos_extra.shape[0]
assert neg.shape[0] == neg_extra.shape[0]

pos_flat = pos.reshape(len(pos), -1)     
neg_flat = neg.reshape(len(neg), -1)    

pos_combined = np.concatenate([pos_output > 1, pos, pos_extra], axis=1)  
neg_combined = np.concatenate([neg_output > 1, neg, neg_extra], axis=1)  

X = np.concatenate([pos_combined, neg_combined], axis=0)      
y = np.array([1]*len(pos_combined) + [0]*len(neg_combined))   


class FeatureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = FeatureDataset(X, y)
loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=False)

def shift_one_right_batchwise(k: torch.Tensor) -> torch.Tensor:
    B, N = k.shape
    device = k.device

    indices = torch.argmax(k, dim=1)           
    not_at_end = indices < (N - 1)             
    not_at_start = indices != 0                

    shift_mask = not_at_end & not_at_start     

    k_new = torch.zeros_like(k)

    row_idx = torch.arange(B, device=device)[shift_mask]
    col_idx = indices[shift_mask] + 1
    k_new[row_idx, col_idx] = 1

    row_idx_static = torch.arange(B, device=device)[~shift_mask]
    col_idx_static = indices[~shift_mask]
    k_new[row_idx_static, col_idx_static] = 1

    return k_new



# ----------------- encoder -----------------
class Encoder(nn.Module):
    def __init__(self, in_dim=10*1 + 5, embed_dim=16*4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 64*3),
            nn.LeakyReLU(),
            nn.Linear(64*3, embed_dim)
        )
        self.classifier = nn.Linear(embed_dim, 1)  # 2 classes

    def forward(self, x, y):
        x_half = x[:,:5]
        x_half = keep_leftmost_one(x_half)
        x_half2 = x[:,5:10]
        x_half2 = shift_one_right_batchwise(x_half2)
        new_input = torch.cat((x_half, x_half2, x[:, 10:]), dim = 1)
        z = self.encoder(new_input) 
        logits = self.classifier(z).squeeze(-1) 
        return z, logits



class WeightedContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0, w_pos: float = 1.0, w_neg: float = 5.0):
        super().__init__()
        self.margin = margin
        self.w_pos = w_pos
        self.w_neg = w_neg

    def forward(self, z1, z2, label):
        d = torch.norm(z1 - z2, dim=1)
        pos_loss = self.w_pos * label * d.pow(2)
        neg_loss = self.w_neg * (1 - label) * torch.clamp(self.margin - d, min=0).pow(2)
        return (pos_loss + neg_loss).mean()

# ----------------- training -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = Encoder().to(device)
optimizer = torch.optim.Adam(list(encoder.parameters()), lr=4e-3)
loss_fn = WeightedContrastiveLoss(margin=50.0, w_pos=1.0, w_neg=20.0)
bce_loss_fn = nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

EPOCHS = 25*2
for epoch in range(EPOCHS):
    encoder.train()
    running_loss = 0.0

    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.float().to(device)
        z, logits = encoder(batch_x, batch_y)

        # ----- contrastive loss -----
        pos_idx = (batch_y == 1).nonzero(as_tuple=True)[0].tolist()
        neg_idx = (batch_y == 0).nonzero(as_tuple=True)[0].tolist()
        if len(pos_idx) < 2 or len(neg_idx) == 0:
            continue

        num_pairs = min(len(pos_idx) // 2, len(neg_idx))
        z1, z2, lbl = [], [], []
        
        for _ in range(num_pairs//2):
            i, j = random.sample(pos_idx, 2)
            z1.append(z[i])
            z2.append(z[j])
            lbl.append(1)
        
        for _ in range(num_pairs//2):
            i, j = random.sample(neg_idx, 2)
            z1.append(z[i])
            z2.append(z[j])
            lbl.append(1)
        
        for _ in range(num_pairs * 100):
            i = random.choice(pos_idx)
            j = random.choice(neg_idx)
            z1.append(z[i])
            z2.append(z[j])
            lbl.append(0)
        
        z1 = torch.stack(z1)
        z2 = torch.stack(z2)
        lbl = torch.tensor(lbl, dtype=torch.float32, device=device)

        loss_contrast = loss_fn(z1, z2, lbl)
        
        loss_classification = bce_loss_fn(logits, batch_y)

        total_loss = 1 * loss_contrast + 100 * loss_classification

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()
    scheduler.step()
    print(f"Epoch {epoch+1:02d}/{EPOCHS}  Total Loss: {running_loss:.4f}")

state_dict = {'net': encoder.state_dict()}
torch.save(state_dict, 'encoder_stage3.pth') 
