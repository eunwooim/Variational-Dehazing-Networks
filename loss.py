import torch 
from math import pi, log, sqrt


log_max = log(1e4)
log_min = log(1e-8)

def vlb_loss(inp, d_out, t_out, d_gt, t_gt, A, sigma, eps1, eps2, kl_j, kl_t):
    alpha = d_out[:,:3]
    m2 = torch.exp(d_out[:,3:].clamp_(min=log_min, max=log_max))
    beta = t_out[:,:1]
    n2 = torch.exp(t_out[:,1:].clamp_(min=log_min, max=log_max))

    # Likelihood
    lh = 0.5 * torch.log(torch.tensor(2*pi)) + 0.5* torch.log(torch.tensor(sigma)) + 0.5 * torch.mean(((inp - (alpha*beta) - A*(1-beta))**2)/sigma + 1)

    # KL divergence for dehazer
    if kl_j == 'gaussian':
        m2_div_eps1 = torch.div(m2, eps1)
        kl_dehaze = 0.5 * torch.mean(-torch.log(m2_div_eps1) -1 + m2_div_eps1 + ((alpha-d_gt)**2 / m2_div_eps1))
    elif kl_j == 'laplace':
        m2_div_eps1 = torch.div(m2, eps1)
        kl_dehaze = torch.mean(m2_div_eps1 + torch.exp(-torch.abs(alpha-d_gt)/eps1) + torch.abs(alpha-d_gt)/eps1 - torch.log(m2_div_eps1) -1)

    # KL divergence for transmission
    if kl_t == 'gaussian':
        n2_div_eps2 = torch.div(n2, eps2)
        kl_transmission = 0.5 * torch.mean(-torch.log(n2_div_eps2) -1 + n2_div_eps2 + ((beta-t_gt)**2 / n2_div_eps2))
    elif kl_t == 'laplace':
        n2_div_eps2 = torch.div(n2, eps2)
        kl_transmission = torch.mean(n2_div_eps2 + torch.exp(-torch.abs(beta-t_gt)/eps2) + torch.abs(beta-t_gt)/eps2 - torch.log(n2_div_eps2) -1)
    elif kl_t == 'lognormal':
        torch.log(1+eps/t_gt)
        n2 = torch.div(n2, eps)
        kl_transmission = torch.div(torch.mean((t_gt-beta)**2 + n2), 2*eps) + torch.mean(torch.log(torch.div(1,n2)))
    
    total_loss = lh + kl_dehaze + kl_transmission

    return total_loss, lh, kl_dehaze, kl_transmission

def loss_val(d_out, d_gt, eps1, kl_j):
    alpha = d_out[:,:3]
    m2 = torch.exp(d_out[:,3:].clamp_(min=log_min, max=log_max))

    # KL divergence for dehazer
    if kl_j == 'gaussian':
        m2_div_eps1 = torch.div(m2, eps1)
        kl_dehaze = 0.5 * torch.mean(-torch.log(m2_div_eps1) -1 + m2_div_eps1 + ((alpha-d_gt)**2 / m2_div_eps1))
    elif kl_j == 'laplace':
        m2_div_eps1 = torch.div(m2, eps1)
        kl_dehaze = torch.mean(m2_div_eps1 + torch.exp(-torch.abs(alpha-d_gt)/eps1) + torch.abs(alpha-d_gt)/eps1 - torch.log(m2_div_eps1) -1)

    return kl_dehaze
