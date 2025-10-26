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
import torchsort
from torchaudio.sox_effects import apply_effects_tensor
import torchaudio.functional as AF

class FeatureDataset(torch.utils.data.Dataset):

    def __init__(self, mode, portion = 1.0):
        if mode == 'training':
            self.X = np.load("../{}_X_{}.npy".format(mode, portion))
            self.y = np.load("../{}_y_{}.npy".format(mode, portion))
        else:
            self.X = np.load("../{}_X.npy".format(mode))
            self.y = np.load("../{}_y.npy".format(mode))
        self.class_to_idxs = {}
        for _i in range(len(self.y)):
            if not self.y[_i] in self.class_to_idxs:
                self.class_to_idxs[self.y[_i]] = []
            self.class_to_idxs[self.y[_i]].append(_i)
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {'X': np.asarray(self.X[idx]), 'y': np.asarray(self.y[idx])}
        return sample




# -------------------- Repro --------------------
SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

class FrameEncoder(nn.Module):
    def __init__(self, F, d=64, hidden=96//1, context=7, dropout=0.1):
        super().__init__()
        k = context          
        self.prj = nn.Linear(F, hidden)  

        def dw_pw(cin, cout, dilation=1):
            return nn.Sequential(
                nn.Conv1d(cin, cin, kernel_size=k, padding=dilation*(k//2), dilation=dilation, groups=cin),
                nn.Conv1d(cin, cout, kernel_size=1),
                nn.GELU(),
                nn.Dropout(dropout)
            )

        self.block1 = dw_pw(hidden, hidden, dilation=1)
        self.block2 = dw_pw(hidden, hidden, dilation=2)
        self.block3 = dw_pw(hidden, d, dilation=4)
        self.ln = nn.LayerNorm(d)

    def forward(self, x):          # x: [B, T, F]
        B,T,F = x.shape
        hx = self.prj(x)            # [B,T,hidden]
        h1 = hx.transpose(1, 2)      # [B,hidden,T] for Conv1d over time
        r = h1
        h = self.block1(h1) + r     # residual
        r = h
        h = self.block2(h) + r
        h = self.block3(h)         # [B,d,T]
        h = h.transpose(1, 2)      # [B,T,d]
        return self.ln(h), self.block1(h1)



class NN1_LocalMatcher(nn.Module):
    def __init__(self, d, hidden=128, dropout=0.1, cosine_head=True):
        super().__init__()
        self.cosine_head = cosine_head
        #self.ln_in = nn.LayerNorm(3*d)

        if not cosine_head:
            self.mlp = nn.Sequential(
                nn.Linear(3*d, hidden), nn.GELU(), nn.Dropout(dropout),
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden//2), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden//2, 1)
            )
        else:
            #self.proj_q = nn.Linear(d, d, bias=False)
            self.scale = nn.Parameter(torch.tensor(10.0))  # learnable temperature
            self.W = nn.Parameter(torch.eye(d))

    def forward(self, phi_t, psi_u):  #placeholder
        z = torch.cat([phi_t, psi_u, (phi_t - psi_u).abs()], dim=-1)
        if self.cosine_head:
            q = F.normalize(self.proj_q(phi_t), dim=-1)
            k = F.normalize(self.proj_q(psi_u), dim=-1)
            # cost = -cosine → lower is better match
            return (-self.scale * (q * k).sum(dim=-1)).contiguous()
        else:
            z = self.ln_in(z)
            return self.mlp(z).squeeze(-1)

    @torch.no_grad()
    def _cosine(self, phi_t, psi_u): #placeholder
        # Standard cosine similarity in [-1,1]
        a = F.normalize(phi_t, dim=-1)
        b = F.normalize(psi_u, dim=-1)
        return (a * b).sum(dim=-1)  # [B]

    def reg_loss(self, phi_t, psi_u, c_tu, lambda_reg=0.1): #placeholder
        """Compute MSE between c_tu and ||phi_t - psi_u||."""
        cos_tgt = -1 * self._cosine(phi_t, psi_u)
        pred_cos = torch.tanh(c_tu)
        return lambda_reg * F.l1_loss(c_tu, cos_tgt)


class SoftArgsortTopK(nn.Module):
    def __init__(self, reg_strength: float = 0.01, temperature: float = 1e-1, largest: bool = True):
        super().__init__()
        self.reg_strength = reg_strength
        self.temperature  = temperature
        self.largest      = largest

    def forward(self, scores: torch.Tensor, ref_class: torch.Tensor, DP, psi_all_expand, k: int = KNN_K, K: int = 8):
        B, R = scores.shape
        device, dtype = scores.device, scores.dtype

        if ref_class.dim() == 1:
            ref_class = ref_class.view(1, R).expand(B, R)  # [B,R]
        ref_class_float = ref_class.to(dtype)

        sorted_vals = torchsort.soft_sort(
            scores, regularization_strength=self.reg_strength
        )  # [B, R]
        yk = sorted_vals[:, -k:]      # [B,k] in s_in space

        M = (s_in.unsqueeze(1) - yk.unsqueeze(-1)).abs()   # [B,k,R]

        logits = -M / self.temperature                     # [B,k,R]
        soft_mask = F.softmax(logits, dim=-1)              # [B,k,R], rows sum ~1

        soft_label_expectation = torch.einsum('bkr,br->bk', soft_mask, ref_class_float)  # [B,k]
        
        selected_DP_soft = torch.einsum('bkr,brtu->bktu', soft_mask, DP)
        selected_psi_soft = torch.einsum('bkr,brtd->bktd', soft_mask, psi_all_expand)
        return yk, soft_label_expectation, selected_DP_soft, selected_psi_soft

class KNNLogitHead(nn.Module):
    
    def __init__(self, K: int=8, class_emb_dim: int = 16,
                 phi_hidden: int = 32, rho_hidden: int = 32,
                 use_counts: bool = True, residual_init: float = 0.0):
        super().__init__()
        self.K = K
        self.use_counts = use_counts

        #self.class_emb = nn.Embedding(K, class_emb_dim)

        in_phi = 1 + 0 #class_emb_dim          # [-dist, class_emb]
        self.phi = nn.Sequential(
            nn.Linear(in_phi, phi_hidden), nn.GELU(),
            nn.Linear(phi_hidden, phi_hidden), nn.GELU()
        )

        in_rho = phi_hidden + (1 if use_counts else 0)
        self.rho = nn.Sequential(
            nn.Linear(in_rho, rho_hidden), nn.GELU(),
            nn.Linear(rho_hidden, 1)
        )
        self.alpha = nn.Parameter(torch.ones(K))
        self.bias  = nn.Parameter(torch.zeros(K))

        self.residual_scale = nn.Parameter(torch.tensor(residual_init, dtype=torch.float32))
        self.residual_scale2 = nn.Parameter(torch.tensor(residual_init, dtype=torch.float32))
        # Lightweight init
        for m in list(self.phi) + list(self.rho):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        #nn.init.normal_(self.class_emb.weight, std=0.02)

    def _soft_onehot_from_fractional(self, frac_ids: torch.Tensor) -> torch.Tensor:
        B, k = frac_ids.shape
        device = frac_ids.device
        K = self.K
        cls_idx = torch.arange(K, device=device, dtype=frac_ids.dtype).view(1,1,K)
        dist2 = (frac_ids.unsqueeze(-1) - cls_idx).abs()
        logits = - dist2 / 1e-5
        oh_soft = F.softmax(logits, dim=-1)  # [B,k,K]
        return oh_soft


    def forward(self, topk_vals: torch.Tensor, topk_classes: torch.Tensor) -> torch.Tensor:
        B, k = topk_vals.shape
        K = self.K
        device = topk_vals.device
        dtype  = topk_vals.dtype

        oh = self._soft_onehot_from_fractional(topk_classes)
        class_sum = torch.einsum('bkc,bk->bc', oh, topk_vals)   # [B, K]

        neigh_feat = torch.cat([topk_vals.unsqueeze(-1),], dim=-1)  # [B,k,1+E]
        phi_out = self.phi(neigh_feat)                          # [B,k,H]

        pooled1 = torch.einsum('bkc,bkh->bch', oh, phi_out)      # [B,K,H]

        if self.use_counts:
            counts = oh.sum(dim=1, keepdim=False)               # [B,K]
            pooled = torch.cat([pooled1, counts.unsqueeze(-1)], dim=-1)  # [B,K,H+1]

        delta = self.rho(pooled).squeeze(-1)                    # [B, K]

        logits_like = class_sum * self.residual_scale2 + self.residual_scale * delta   # [B, K]
        return logits_like, class_sum, counts

class NN2_SoftMin(nn.Module):
    def __init__(self, init_tau=0.2, tau_min=0.02, tau_max=2.0, learn_tau=True, add_gap=False, hidden = 16):
        super().__init__()
        self.learn_tau = learn_tau
        self.tau_min, self.tau_max = tau_min, tau_max
        self.net = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, hidden), nn.ReLU(),
            nn.Linear(hidden, 3)
        )
        self.log_tau = nn.Parameter(torch.log(torch.tensor(init_tau))) if learn_tau else nn.Parameter(torch.log(torch.tensor(init_tau)), requires_grad=False)
        self.add_gap = add_gap
        if add_gap:
            self.gap = nn.Parameter(torch.zeros(3))  # bias moves if desired

    def forward(self, neigh3: torch.Tensor):
        # neigh3: [B,3] values = [up,left,diag]
        if self.add_gap:
            neigh3 = neigh3 + self.gap.view(1,3)
        tau = torch.clamp(torch.exp(self.log_tau), self.tau_min, self.tau_max)
        best_vals = -tau * torch.logsumexp(-neigh3 / tau, dim=-1)  # [B]
        best_vals = torch.sum(neigh3 * F.softmax(self.net(-torch.abs(neigh3 - best_vals.unsqueeze(-1))), dim = -1), dim = -1)
        return best_vals


class AINN_SC_Merged(nn.Module):
    def __init__(self, F=13, T=61, num_classes=8, d=64//2, hidden=64//2):
        super().__init__()
        self.F, self.T, self.K = F, T, num_classes
        self.enc = FrameEncoder(F, d)
        self.nn1 = NN1_LocalMatcher(d, hidden)
        self.nn2 = NN2_SoftMin(d, hidden)
        self.selector = SoftArgsortTopK()
        self.agg = KNNLogitHead()
        
    def _neigh3x3(self, DP: torch.Tensor, t: int, u: int) -> torch.Tensor:
        B = DP.size(0)
        rows = torch.tensor([t-1, t], device=DP.device).clamp(min=0)
        cols = torch.tensor([u-1, u], device=DP.device).clamp(min=0)
        neigh, mask = [], []
        for r in rows:
            for c in cols:
                if r == t and c == u:
                    continue
                neigh.append(DP[:, r, c])          
                mask.append((r == 0) | (c == 0))
        neigh = torch.stack(neigh, dim=-1)         
        mask = torch.tensor(mask, device=DP.device, dtype=DP.dtype).unsqueeze(0)
        return neigh
    
    def forward(self, x, refs, knn_k: int = 5):
        B, T, Fk = x.shape
        K, B_ref = refs.shape[0], refs.shape[1]
        #phi = self.enc(x)                  # [B, T, d]
        all_scores = x.new_zeros(B, K, B_ref)
        
        #will optimize later
        K, B_ref, T, Fk = refs.shape
        B = x.size(0)
        R = K * B_ref

        phi, phi_int = self.enc(x)                            # [B, T, d]
        refs_flat = refs.reshape(R, T, Fk)           # [R, T, F]
        psi_all, _ = self.enc(refs_flat)                # [R, T, d]

        q = F.normalize(phi, dim = -1)                 # [B, T, d]
        k = F.normalize(psi_all, dim = -1)           # [R, T, d]
        W = self.nn1.W                              # (d, d)
        W_s = 0.5 * (W + W.t())
        qW = torch.einsum('btd,df->btf', q, W_s)     # [B, T, d]
        cross = torch.einsum('btf,ruf->brtu', qW, k)  # [B, R, T, T]
        C_all = -self.nn1.scale * cross


        phi2 = (F.normalize(phi, dim = -1) ** 2).sum(dim=-1)                 # [B, T]
        psi2 = (F.normalize(psi_all, dim = -1) ** 2).sum(dim=-1)             # [R, T]
        cross = torch.einsum('btd,rud->brtu', q, k)  # [B, R, T, T]


        DP = x.new_full((B, R, T+1, T+1), 0.0)
        pad_bias = 8.0
        DP[:, :, 0, :] = pad_bias
        DP[:, :, :, 0] = pad_bias
        DP[:, :, 0, 0]  = 0.0

        DP_ref = x.new_full((B, R, T+1, T+1), 0.0)
        DP_ref[:, :, 0, :] = pad_bias
        DP_ref[:, :, :, 0] = pad_bias
        DP_ref[:, :, 0, 0]  = 0.0



        for t in range(1, T+1):
            for u in range(1, T+1):
                up   = DP[:, :, t-1, u]      # [B,R]
                left = DP[:, :, t,   u-1]    # [B,R]
                diag = DP[:, :, t-1, u-1]    # [B,R]
                DP[:, :, t, u] = C_all[:, :, t-1, u-1] + self.nn2(torch.stack([up, left, diag], dim = -1))            # [B,R]
                DP_ref[:, :, t, u] = C_all[:, :, t-1, u-1] + torch.min(torch.stack([DP_ref[:, :, t-1, u], DP_ref[:, :, t, u-1], DP_ref[:, :, t-1, u-1]], dim = -1), dim = -1)[0]

        all_scores = DP[:, :, T, T]                 # [B, R]
        all_scores = all_scores.view(B, K, B_ref)   # [B, K, B_ref]



        flat_scores = all_scores.reshape(B, K * B_ref)
        ref_class = torch.arange(K, device=x.device).repeat_interleave(B_ref)  # [K*B_ref]

        d = psi_all.size(-1)
        psi_all_expand = psi_all.unsqueeze(0).expand(B, R, T, d)

        soft_vals, soft_classes, out, topk_ref_enc = self.selector(
            -flat_scores, ref_class, DP, psi_all_expand
        )

        
        logits_like, cs, l2 = self.agg(soft_vals, soft_classes)
        return torch.softmax(logits_like, dim = 1), DP, DP_ref, C_all, phi_int, cs, out, l2, topk_ref_enc  


@torch.no_grad()
def build_knn_refs(dataset, K: int, T: int, F: int, B_ref: int, device=None, seed: int | None = None):
    device = device or "cpu"
    rng = np.random.default_rng(seed)
    refs = torch.zeros(K, B_ref, T, F, device=device, dtype=torch.float32)
    labels = torch.zeros(K, B_ref, dtype=torch.long, device=device)

    for k in range(K):
        pool = dataset.class_to_idxs.get(k, [])
        if len(pool) == 0:
            raise RuntimeError(f"No samples in class {k} for reference building.")
        if len(pool) >= B_ref:
            picks = rng.choice(pool, size=B_ref, replace=False)
        else:
            picks = rng.choice(pool, size=B_ref, replace=True)

        for r, idx in enumerate(picks):
            xk = dataset[idx]['X']               
            refs[k, r] = torch.as_tensor(xk, dtype=torch.float32, device=device)
            labels[k, r] = k
    return refs, labels


# -------------------- Train / Eval --------------------
def run_sc(
    epochs=80,
    batch_size=64*8,
    lr=2e-3,
    device=None
):
    best_acc = -1
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    tr = FeatureDataset('training', 0.7) #for prototype consistency
    tr_w = FeatureDataset('training', 1.0)
    va = FeatureDataset('validation')
    te = FeatureDataset('testing')
    train_loader = DataLoader(tr_w, batch_size=batch_size, shuffle=False,  num_workers=2)
    val_loader   = DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=2)

    model = AINN_SC_Merged(F=13, T=TARGET_T, num_classes=8, d=64, hidden=64).to(device)
    
    #total = sum(p.numel() for p in model.parameters())
    
    def eval_loop(loader, proto_source_ds, mode):
        model.eval()
        correct = total = 0
        with torch.no_grad():
            dp_list = []
            dp_ref_list = []
            c_list = []
            pred_list = []
            gt_list = []
            phi_int_list = []
            ref_enc_list = []
            cs_list = []
            l1_list = []
            l2_list = []
            for _input in loader:
                x, y = _input['X'], _input['y']
                B = x.size(0)
                x, y = x.to(device), y.to(device)
                refs, _ = build_knn_refs(proto_source_ds, K=8, T=61, F=13,
                                     B_ref=KNN_BREF, device=device, seed=1234)
                logits, dp, dp_ref, c_all, phi_int, cs, l1, l2, ref_enc = model(x, refs, knn_k=KNN_K)
                dp_list.extend(dp.detach().cpu().numpy().tolist())
                dp_ref_list.extend(dp_ref.detach().cpu().numpy().tolist())
                c_list.extend(c_all.detach().cpu().numpy().tolist())
                pred = logits.argmax(dim=-1)
                pred_list.extend(pred.detach().cpu().numpy().tolist())
                gt_list.extend(y.detach().cpu().numpy().tolist())
                phi_int_list.extend(phi_int.detach().cpu().numpy().tolist())
                l1_list.extend(l1.detach().cpu().numpy().tolist())
                l2_list.extend(l2.detach().cpu().numpy().tolist())
                cs_list.extend(cs.detach().cpu().numpy().tolist())
                ref_enc_list.extend(ref_enc.detach().cpu().numpy().tolist())
                correct += (pred == y).sum().item()
                total += y.numel()
                
        #we don't need all of them, only certain activations/intermediate outputs are enough. Listing all for completeness.
        np.save('{}_c_list_le.npy'.format(mode), np.array(c_list))
        np.save('{}_dp_list_le.npy'.format(mode), np.array(dp_list))
        np.save('{}_dp_ref_list_le.npy'.format(mode), np.array(dp_ref_list))
        np.save('{}_pred_list_le.npy'.format(mode), np.array(pred_list))
        np.save('{}_gt_list_le.npy'.format(mode), np.array(gt_list))
        np.save('{}_phi_int_list_le.npy'.format(mode), np.array(phi_int_list))
        np.save('{}_l1_list_le.npy'.format(mode), np.array(l1_list))
        np.save('{}_l2_list_le.npy'.format(mode), np.array(l2_list))
        np.save('{}_cs_list_le.npy'.format(mode), np.array(cs_list))
        np.save('{}_ref_enc_list_le.npy'.format(mode), np.array(ref_enc_list))
        return correct / total if total > 0 else 0.0

    
    state_dict = torch.load(
        f'dtw_model_{portion:.1f}_less_iter.pth', map_location='cuda'
    )
    model.load_state_dict(state_dict['net'])
    test_acc = eval_loop(train_loader, tr, 'train')
    test_acc = eval_loop(test_loader, tr, 'test')
    

if __name__ == "__main__":
    run_sc()
