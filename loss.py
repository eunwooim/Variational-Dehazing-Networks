import torch 
from math import pi, log, sqrt


log_max = 1e4
log_min = 1e-8

def vlb_loss(inp, d_out, t_out, d_gt, t_gt, A, sigma, eps1, eps2, kl_j, kl_t):
    # Likelihood
    lh = 0.5 * torch.mean((inp - (d_out*t_out) - A*(1-t_out))**2)/sigma
    
    # KL divergence for dehazer
    if kl_j == 'gaussian':
        kl_dehaze = 0.5 * torch.mean((d_out-d_gt)**2)/eps1
    if kl_j == 'laplace':
        kl_dehaze = (torch.exp(-torch.abs(d_out - d_gt) / eps1) * eps1 + torch.abs(d_out - d_gt)) / eps1
    
    # KL divergence for transmission
    if kl_t == 'gaussian':
        kl_transmission = torch.mean((t_out-t_gt)**2)/eps2)
    if kl_t == 'lognormal':
        kl_transmission = torch.div(torch.mean((torch.log(t_out) - torch.log(t_gt))**2),2*eps2)

    total_loss = lh + kl_dehaze + kl_transmission
    return total_loss, lh, kl_dehaze, kl_transmission

def loss_val(d_out, d_gt, eps1, kl_j):
    # KL divergence for dehazer
    if kl_j == 'gaussian':
        kl_dehaze = torch.mean((d_out-d_gt)**2 / eps1)
    elif kl_j == 'laplace':
        kl_dehaze = torch.mean(torch.exp(-torch.abs(d_out-d_gt)/eps1) + torch.abs(d_out-d_gt)) / eps1

    return kl_dehaze
