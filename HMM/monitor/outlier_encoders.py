import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pickle
import random

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


# ========== Load Data ==========
def load_data(prefix='', mode = 'train'):
    if len(prefix) == 0:
        return (
        pickle.load(open('stage1_z1_{}.pkl'.format(mode), 'rb')),
        pickle.load(open('stage1_z_embed_{}.pkl'.format(mode), 'rb')),
        pickle.load(open('stage1_inter_{}.pkl'.format(mode), 'rb')),
        pickle.load(open('stage2_ret_{}.pkl'.format(mode), 'rb')),
        pickle.load(open('stage2_pred_{}.pkl'.format(mode), 'rb')),
        pickle.load(open('stage2_inter_{}.pkl'.format(mode), 'rb')),
        pickle.load(open('stage2_inter2_{}.pkl'.format(mode), 'rb')),
        )
    else:
        return (
        pickle.load(open('{}stage1_z1_{}.pkl'.format(prefix, mode), 'rb')),
        pickle.load(open('{}stage1_z_embed_{}.pkl'.format(prefix, mode), 'rb')),
        pickle.load(open('{}stage1_inter_{}.pkl'.format(prefix, mode), 'rb')),
        pickle.load(open('{}stage2_ret_{}.pkl'.format(prefix, mode), 'rb')),
        pickle.load(open('{}stage2_pred_{}.pkl'.format(prefix, mode), 'rb')),
        pickle.load(open('{}stage2_inter_{}.pkl'.format(prefix, mode), 'rb')),
        pickle.load(open('{}stage2_inter2_{}.pkl'.format(prefix, mode), 'rb')),
        )

# ========== Dataset ==========
class FullDataset(Dataset):
    def __init__(self, z1_list, z_embed_list, inter_list, ret_list, pred_list, inter2_list, inter3_list):
        self.data = list(zip(z1_list, z_embed_list, inter_list, ret_list, pred_list, inter2_list, inter3_list))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        z1, z_embed, inter, ret, pred, inter2, inter3 = self.data[idx]
        return (
            torch.tensor(z1, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(z_embed, dtype=torch.float32),
            torch.tensor(inter, dtype=torch.float32),
            torch.tensor(ret, dtype=torch.float32),
            torch.tensor(pred, dtype=torch.long),
            torch.tensor(inter2, dtype=torch.float32),
            torch.tensor(inter3, dtype=torch.float32)
        )

# ========== Collate Function ==========
def collate_fn(batch):
    z1, z_embed, inter, ret, pred, inter2, inter3 = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in z1])
    z1 = pad_sequence(z1, batch_first=True)
    z_embed = pad_sequence(z_embed, batch_first=True)
    inter = pad_sequence(inter, batch_first=True)
    mask = torch.arange(z1.shape[1]).expand(len(lengths), z1.shape[1]) >= lengths.unsqueeze(1)
    return z1, z_embed, inter, mask, torch.stack(ret), torch.stack(pred), torch.stack(inter2), torch.stack(inter3)

# ========== Models ==========
class Stage1TransformerEncoder(nn.Module):
    def __init__(self, input_dims, d_model=128*2, out_dim=1):
        super().__init__()
        self.proj_z1 = nn.Linear(input_dims[0], d_model)
        self.proj_z_embed = nn.Linear(input_dims[1], d_model)
        self.proj_inter = nn.Linear(input_dims[2], d_model)
        self.fusion = nn.Linear(d_model * 3, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=4, batch_first=True), num_layers=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output_proj = nn.Linear(d_model, out_dim)

    def forward(self, z1, z_embed, inter, mask):
        z1 = self.proj_z1(z1)
        z_embed = self.proj_z_embed(z_embed)
        inter = self.proj_inter(inter)
        x = self.fusion(torch.cat([z1, z_embed, inter], dim=-1))
        x = self.encoder(x, src_key_padding_mask=mask)
        x = self.pool(x.permute(0, 2, 1)).squeeze(-1)
        return self.output_proj(x)

class Stage2MLPEncoder(nn.Module):
    def __init__(self, out_dim=1):
        super().__init__()
        self.pred_embed = nn.Embedding(10, 16)
        self.mlp = nn.Sequential(
            nn.Linear(2 + 16 + 32*2 + 32*2, 64*2), nn.ReLU(), nn.Linear(64*2, out_dim)
        )

    def forward(self, ret, pred, inter1, inter2):
        pred_emb = self.pred_embed(pred)
        x = torch.cat([ret, pred_emb, inter1, inter2], dim=-1)
        return self.mlp(x)

# ========== Contrastive Loss ==========
def contrastive_loss(f1, f2, label, margin=1.0):
    dist = torch.norm(f1 - f2, dim=1)
    loss_pos = label * dist.pow(2)
    loss_neg = (1 - label) * F.relu(margin - dist).pow(2)
    return (loss_pos + loss_neg).mean()


class GaussianClusterLoss(nn.Module):
    def __init__(self, feat_dim=2, sigma=1.0, learn_sigma=False):
        super().__init__()
        self.means = nn.Parameter(torch.randn(2, feat_dim))
        if learn_sigma:
            self.log_sigma = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer('log_sigma', torch.tensor(np.log(sigma)))

    def forward(self, feat, labels):
        mu = self.means[labels]
        dist_sq = ((feat - mu) ** 2).sum(dim=1)
        loss = 0.5 * dist_sq / torch.exp(self.log_sigma * 2) + self.log_sigma
        return loss.mean()

def evaluate(model1, model2, classifier, test_dataset, test_labels, device):
    model1.eval()
    model2.eval()
    classifier.eval()
    with torch.no_grad():
        data = [test_dataset[i] for i in range(len(test_dataset))]
        z1, z_embed, inter, mask, ret, pred, inter2, inter3 = map(lambda x: x.to(device), collate_fn(data))
        f1 = model1(z1, z_embed, inter, mask)
        f2 = model2(ret, pred, inter2, inter3)
        feat = torch.cat([f1, f2], dim=-1)
        logits = classifier(feat).squeeze(1)
        preds = (logits > 0).long().cpu()
        labels = torch.tensor(test_labels).long()
        acc = (preds == labels).float().mean().item()
    print(f"Test Accuracy: {acc:.4f}")

# ========== Training ==========
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pos_data = load_data('', 'train')
    neg_data = load_data('_', 'train')

    N1, N2, N3 = 1, pos_data[1][0].shape[1], pos_data[2][0].shape[1]
    pos_dataset = FullDataset(*pos_data)
    neg_dataset = FullDataset(*neg_data)

    stage1_model = Stage1TransformerEncoder(input_dims=(N1, N2, N3)).to(device)
    stage2_model = Stage2MLPEncoder().to(device)
    classifier = nn.Linear(2, 1).to(device)
    cluster_loss_fn = GaussianClusterLoss().to(device)

    optimizer = torch.optim.AdamW(
        list(stage1_model.parameters()) + list(stage2_model.parameters()) + list(classifier.parameters()) + list(cluster_loss_fn.parameters()),
        lr=1e-4
    )

    for epoch in range(1, 11):
        total_correct = 0
        total_samples = 0

        pairs = []
        labels = []
        pos_idx = list(range(len(pos_dataset)))
        neg_idx = list(range(len(neg_dataset)))
        single_feats = []
        single_label = []
        for _ in range(250):
            i, j = random.sample(pos_idx, 2)
            pairs.append((pos_dataset[i], pos_dataset[j]))
            labels.append(1)
            single_feats.append(pos_dataset[i])
            single_label.append(0)
            single_feats.append(pos_dataset[j])
            single_label.append(0)
        for _ in range(250):
            i, j = random.sample(neg_idx, 2)
            pairs.append((neg_dataset[i], neg_dataset[j]))
            labels.append(1)
            single_feats.append(neg_dataset[i])
            single_label.append(1)
            single_feats.append(neg_dataset[j])
            single_label.append(1)
        for _ in range(5000):
            i = random.choice(pos_idx)
            j = random.choice(neg_idx)
            pairs.append((pos_dataset[i], neg_dataset[j]))
            labels.append(0)
            single_feats.append(pos_dataset[i])
            single_label.append(0)
            single_feats.append(neg_dataset[j])
            single_label.append(1)

        random.shuffle(pairs)

        batch_size = 32
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i+batch_size]
            batch_labels = torch.tensor(labels[i:i+batch_size], dtype=torch.float32).to(device)

            a_list, b_list = zip(*batch_pairs)
            a = collate_fn(a_list)
            b = collate_fn(b_list)

            z1a, z_embed_a, inter_a, mask_a, ret_a, pred_a, inter2_a, inter3_a = map(lambda x: x.to(device), a)
            z1b, z_embed_b, inter_b, mask_b, ret_b, pred_b, inter2_b, inter3_b = map(lambda x: x.to(device), b)

            f1a = stage1_model(z1a, z_embed_a, inter_a, mask_a)
            f2a = stage2_model(ret_a, pred_a, inter2_a, inter3_a)
            f1b = stage1_model(z1b, z_embed_b, inter_b, mask_b)
            f2b = stage2_model(ret_b, pred_b, inter2_b, inter3_b)

            feat_a = torch.cat([f1a, f2a], dim=-1)
            feat_b = torch.cat([f1b, f2b], dim=-1)

            loss_contrast = contrastive_loss(feat_a, feat_b, batch_labels)
            
            single_list = collate_fn(single_feats)
            cls_batch = [x.to(device) for x in single_list]
            cls_labels = torch.tensor(single_label, dtype=torch.float32).to(device)
            z1, z_embed_, inter_, mask_, ret_, pred_, inter2_, inter3_ = map(lambda x: x.to(device), cls_batch)
            f_out = stage1_model(z1, z_embed_, inter_, mask_)
            f_out2 = stage2_model(ret_, pred_, inter2_, inter3_)

            feat_ = torch.cat([f_out, f_out2], dim=-1)
            

            logits_all = classifier(feat_).squeeze(1)
            loss_cls = F.binary_cross_entropy_with_logits(logits_all, cls_labels)


            loss_cluster = cluster_loss_fn(feat_, cls_labels.long())

            total_loss = loss_cluster + 100 * loss_cls + loss_contrast

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            preds = (logits_all > 0).long()
            total_correct += (preds == cls_labels.long()).sum().item()
            total_samples += cls_labels.size(0)

        acc = total_correct / total_samples
        print(f"Epoch {epoch}: Accuracy = {acc:.4f}")

    torch.save({
    'stage1': stage1_model.state_dict(),
    'stage2': stage2_model.state_dict(),
    'classifier': classifier.state_dict(),
    }, 'model_final.pth')

if __name__ == "__main__":
    train()
