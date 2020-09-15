# -*- coding: utf-8 -*-
"""
@ project: WDGRL
@ author: lzx
@ file: mmd.py
@ time: 2019/6/18 16:10
"""

import torch

min_var_test = 1e-8


src = torch.randn(64,100)
tgt = torch.randn(64,100)
def guassian_kernel(src,tgt,kernel_mul = 2,kernel_num = 5,fix_sigma = None):
    n_samples = int(src.size()[0])+int(tgt.size()[0])
    total = torch.cat([src,tgt],dim = 0)# 按列合并 (n_samples,feature_dim)
    total0 = total.unsqueeze(0).expand(int(total.size(0)),int(total.size(0)),int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)),int(total.size(0)),int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  # /len(kernel_val)

def rational_quadratic_kernel(x, y, a=1, l=1):
    return torch.pow(1. + torch.pow(x - y, 2) / (2 * a * l * l), -a)

def mmd_rbf_(src,tgt,sigma_list=None):
    batch_size = src.shape[0]
    kernels = guassian_kernel(src, tgt,
                              kernel_mul=2, kernel_num=5, fix_sigma=None)
    XX = kernels[:batch_size,:batch_size]
    YY = kernels[batch_size:,batch_size:]
    XY = kernels[:batch_size,batch_size:]
    YX = kernels[batch_size:,:batch_size]
    loss = torch.mean(XX+YY-XY-YX)
    return loss

def mmd_squared(P,Q):
    """ 
    Use the formula for MMD squared provided in the paper
    to compute the maximum mean discrepancy score
    """
    m       = P.shape[0]
    ratio   = 1.0 / (m*(m-1.0))
    sigma   = 0.0 
 
    for i in range(m):
        for j in range(m):
            if i != j:
                sigma += rational_quadratic_kernel(P[i],P[j]).mean() + rational_quadratic_kernel(Q[i],Q[j]).mean() - rational_quadratic_kernel(P[i],Q[j]).mean() - rational_quadratic_kernel(P[j],Q[i]).mean()
    
    
    return torch.abs(sigma)

# print(mmd_squared(src, tgt, a=0.5))
# print(mmd_squared(src, tgt, a=1))
