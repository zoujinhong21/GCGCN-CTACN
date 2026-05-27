import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from Pre_CMAPSS import seed, TIME_WINDOW


class ChebConv(nn.Module):
    def __init__(self, in_features, out_features, K, num_nodes):
        super(ChebConv, self).__init__()
        self.K = K
        self.in_features = in_features
        self.out_features = out_features
        self.num_nodes = num_nodes

        self.weight = nn.Parameter(torch.FloatTensor(K, in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, laplacian):
        batch_size = laplacian.size(0)
        total_samples = x.size(0)
        time_window = total_samples // batch_size

        laplacian = laplacian.unsqueeze(1)  # (B, 1, N, N)
        laplacian = laplacian.repeat(1, time_window, 1, 1)  # (B, T, N, N)

        laplacian = laplacian.reshape(total_samples, self.num_nodes, self.num_nodes)

        Tx_0 = x
        Tx = [Tx_0]

        if self.K > 1:
            Tx_1 = torch.einsum('bij,bjk->bik', laplacian, Tx_0)
            Tx.append(Tx_1)

        for k in range(2, self.K):
            term1 = 2 * torch.einsum('bij,bjk->bik', laplacian, Tx[-1])
            Tx_k = term1 - Tx[-2]
            Tx.append(Tx_k)

        out = torch.zeros(total_samples, self.num_nodes, self.out_features,
                          device=x.device, dtype=x.dtype)

        for k in range(self.K):
            weighted = torch.einsum('bni,io->bno', Tx[k], self.weight[k])
            out += weighted

        return out


class ChebGCN(nn.Module):
    def __init__(self, in_features, out_features, K, num_nodes):
        super(ChebGCN, self).__init__()
        self.cheb_gcn = ChebConv(in_features, out_features, K, num_nodes)
        self.relu = nn.ReLU()

    def forward(self, x, laplacian):
        B, T, N, F = x.shape
        x = x.contiguous().view(B*T, N, F)
        x = self.cheb_gcn(x, laplacian)  # GCN -> (B*T, N, F_out)
        x = x.view(B, T, N, -1)          # (B, T, N, F_out)
        x = self.relu(x)
        return x



def build_normalized_laplacian(adj):
    B, N, _ = adj.shape
    I = torch.eye(N, device=adj.device).unsqueeze(0).expand(B, -1, -1)

    D_pos = torch.sum(torch.relu(adj), dim=2)
    D_neg = torch.sum(torch.relu(-adj), dim=2)

    D_pos_safe = torch.where(D_pos < 1e-7, torch.ones_like(D_pos), D_pos)
    D_pos_inv_sqrt = torch.pow(D_pos_safe, -0.5)
    D_pos_inv_sqrt = torch.diag_embed(D_pos_inv_sqrt)


    D_neg_safe = torch.where(D_neg < 1e-7, torch.ones_like(D_neg), D_neg)
    D_neg_inv_sqrt = torch.pow(D_neg_safe, -0.5)
    D_neg_inv_sqrt = torch.diag_embed(D_neg_inv_sqrt)


    L_pos = torch.bmm(torch.bmm(D_pos_inv_sqrt, torch.relu(adj)), D_pos_inv_sqrt)

    L_neg = torch.bmm(torch.bmm(D_neg_inv_sqrt, torch.relu(-adj)), D_neg_inv_sqrt)

    L_sym = I - L_pos + L_neg

    max_val = torch.max(torch.abs(L_sym))
    laplacian = L_sym / torch.clamp(max_val, min=1e-7)

    return laplacian

def build_adaptive_laplacian(X, k=14, threshold=0.81): # 0.81
    B, N, T = X.shape
    X_norm = F.normalize(X, p=2, dim=2)
    adaptive_adj = torch.bmm(X_norm, X_norm.transpose(1, 2))

    abs_adj = torch.abs(adaptive_adj)
    topk_mask = torch.zeros_like(abs_adj)
    topk_values, topk_indices = torch.topk(abs_adj, k, dim=2)
    topk_mask.scatter_(2, topk_indices, 1.0)

    adaptive_adj = torch.where(
        (abs_adj > threshold) & (topk_mask.bool()),
        adaptive_adj,
        torch.zeros_like(adaptive_adj)
    )

    return build_normalized_laplacian(adaptive_adj)



class SELU(nn.Module):
    def __init__(self, alpha=1.67326, scale=1.0507):
        super(SELU, self).__init__()
        self.alpha = alpha
        self.scale = scale

    def forward(self, x):
        return self.scale * torch.where(x > 0, x, self.alpha * (torch.exp(x) - 1))

class TemporalCBAM(nn.Module):
    def __init__(self, feature_dim, reduction_ratio=8, kernel_size=3):
        super().__init__()
        self.feature_dim = feature_dim

        self.channel_att = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // reduction_ratio),
            nn.ReLU(),
            nn.Linear(feature_dim // reduction_ratio, feature_dim),
            nn.Sigmoid()
        )


        self.time_att = nn.Sequential(
            nn.Conv1d(2, 8, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(8, 1, kernel_size, padding=kernel_size // 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        channel_avg = torch.mean(x, dim=1)
        channel_max = torch.max(x, dim=1)[0]

        channel_pool = channel_avg + channel_max

        channel_weights = self.channel_att(channel_pool)
        channel_weights = channel_weights.unsqueeze(1)

        time_stats = torch.cat([
            torch.mean(x, dim=2, keepdim=True),
            torch.max(x, dim=2, keepdim=True)[0]
        ], dim=2)


        time_stats = time_stats.permute(0, 2, 1)
        time_weights = self.time_att(time_stats)
        time_weights = time_weights.permute(0, 2, 1)

        y = x + x * channel_weights * time_weights

        return y



class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()



class Bottle2neck(nn.Module):
    expansion = 1
    groups = 3  # 2,3,4,5
    def __init__(self, num_nodes=14,in_channel=TIME_WINDOW, out_channel=TIME_WINDOW,in_features = 16 ,stride=1, downsample=None,
                 groups=groups, width_per_group=16):  #
        super(Bottle2neck, self).__init__()
        width = int(out_channel * (width_per_group / 60)) * groups
        self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)

        self.conv2 = nn.Conv1d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=2)
        self.chomp1 = Chomp1d(2)
        self.conv22 = nn.Conv1d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=4,dilation=2)
        self.chomp2 = Chomp1d(4)

        self.conv3 = nn.Conv1d(in_channels=width, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channel * self.expansion)
        self.fce = nn.Linear(TIME_WINDOW * num_nodes * in_features, 128)
        self.downsample = downsample
        self.cbam = TemporalCBAM(feature_dim=num_nodes*in_features)
        self.selu = SELU()


    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.cbam(out)
        out = self.selu(out)

        out1 = self.conv2(out)
        out1 = self.chomp1(out1)
        out1 = self.cbam(out1)

        out2 = self.conv22(out)
        out2 = self.chomp2(out2)
        out2 = self.cbam(out2)

        stacked = torch.stack([out1,out2,out2], dim=0)
        out = torch.mean(stacked, dim=0)
        out = self.selu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out += identity
        out = self.selu(out)
        out = out.view(out.size(0), -1)
        out = self.fce(out)
        return out


class CombinedModel(nn.Module):
    def __init__(self, num_nodes, in_features, hidden_features1, hidden_features2, out_features, K):
        super(CombinedModel, self).__init__()
        self.num_nodes = num_nodes
        self.layer1 = ChebGCN(in_features, hidden_features1, K, num_nodes)
        self.layer2 = ChebGCN(hidden_features1+1, hidden_features2, K, num_nodes)
        self.layer3 = ChebGCN(hidden_features1 + hidden_features2 + 1, out_features, K, num_nodes)

        self.cn_model = Bottle2neck(in_features=out_features,num_nodes=num_nodes)

        self.fc = nn.Linear(128, 1)

        self.relu = nn.ReLU()


    def forward(self, x):
        l_in = x.reshape(x.size(0), x.size(1), -1).permute(0, 2, 1)
        laplacian = build_adaptive_laplacian(l_in, k=self.num_nodes)

        out_sn1 = self.layer1(x, laplacian)
        out_sn11 = torch.cat((x, out_sn1), 3)
        out_sn2 = self.layer2(out_sn11, laplacian)
        out_sn22 = torch.cat((x, out_sn1, out_sn2), 3)
        out_sn3 = self.layer3(out_sn22, laplacian)

        in_cn = out_sn3.reshape(out_sn3.size(0), out_sn3.size(1), -1)
        out_cn = self.cn_model(in_cn)
        out_cn = self.relu(out_cn)

        out = self.fc(out_cn)
        return out


