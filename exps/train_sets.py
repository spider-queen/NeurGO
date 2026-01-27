import torch
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def lf1_synthetic_sin(x, b=None, w=None):
    """
    LF1: 原代码中的 cecfun1 (Synthetic Sine)
    f(x) = (sin(x-b) @ w)^2
    """
    batch, n, dim = x.shape
    z = x if b is None else x - b.view(-1)
    sc = torch.sin(z)
    
    # 兼容性处理：如果未生成 w，则使用全1向量
    if w is None:
        w = torch.ones((dim, 1), device=x.device)
        
    sc = sc @ w  # (b,n,d) @ (d,1) = (b,n,1)
    sc = torch.pow(sc, 2).view(batch, n)
    return sc

def lf2_shifted_l1(x, b=None):
    """
    LF2: 原代码中的 cecfun2 (Shifted L1 Norm)
    f(x) = sum(|x-b|)
    """
    if b is not None:
        b = b.view(-1)
        z = x - b
    else:
        z = x
    sc = torch.sum(torch.abs(z), dim=2)
    return sc

def lf3_shifted_correlation(x, b=None):
    """
    LF3: 原代码中的 cecfun3 (Shifted Correlation)
    """
    if b is not None:
        z = x - b
    else:
        z = x
    z1 = z[:, :, :-1]
    z2 = z[:, :, 1:]
    sc = torch.sum(torch.abs(z1 + z2), dim=2) + torch.sum(torch.abs(z), dim=2)
    return sc


# ==========================================
# 训练集配置
# ==========================================

TRAIN_FUNCTIONS = [
    {
        'fid': 'lf1',
        'fun': lf1_synthetic_sin,
        'xlb': -10, 'xub': 10,
        # bias 和 w 将由 gen_train_offset 生成
    },
    {
        'fid': 'lf2',
        'fun': lf2_shifted_l1,
        'xlb': -10, 'xub': 10,
    },
    {
        'fid': 'lf3',
        'fun': lf3_shifted_correlation,
        'xlb': -10, 'xub': 10,
    }
]

def gen_train_offset(dim, fun):
    """为训练函数生成随机偏移和权重"""
    if fun['fid'] == 'lf1':
        fun['bias'] = (torch.rand(dim, device=DEVICE) - 0.5) * (fun['xub'] - fun['xlb'])
        fun['w'] = torch.randn((dim, 1), device=DEVICE)
    elif fun['fid'] in ['lf2', 'lf3']:
        fun['bias'] = (torch.rand(dim, device=DEVICE) - 0.5) * (fun['xub'] - fun['xlb'])

def get_train_fitness(x, fun):
    """计算训练函数的 Fitness"""
    if fun['fid'] == 'lf1':
        return fun['fun'](x, b=fun.get('bias'), w=fun.get('w'))
    else:
        return fun['fun'](x, b=fun.get('bias'))