import torch 
from math import pi, log 


log_max = log(1e4)
log_min = log(1e-8)

def loss(inp, d_out, t_out, d_gt, t_gt, A, sigma, eps, kl_j, kl_t):
    alpha = d_out[:,:3]
    m2 = torch.exp(d_out[:,3:].clamp_(min=log_min, max=log_max))
    beta = t_out[:,:1]
    n2 = torch.exp(t_out[:,1:].clamp_(min=log_min, max=log_max))

    # Likelihood
    lh = 0.5 * torch.log(torch.tensor(2*pi)) + 0.5* torch.log(torch.tensor(sigma)) + 0.5 * torch.mean(((inp - (alpha*beta) - A*(1-beta))**2)/sigma + 1)

    # KL divergence for dehazer
    if kl_j == 'gaussian':
        m2 = torch.div(m2, eps)
        kl_dehaze = 0.5 * torch.mean(-torch.log(m2) -1 + m2 + ((alpha-d_gt)**2 / eps))
    elif kl_j == 'laplace':
        eps = torch.sqrt(eps/2)
        m2 = torch.div(m2, eps)
        kl_dehaze = torch.mean(m2 + torch.exp(-torch.abs(alpha-d_gt)/eps) + torch.abs(alpha-d_gt)/eps - torch.log(m2) -1)

    # KL divergence for transmission
    if kl_t == 'gaussian':
        n2 = torch.div(n2, eps)
        kl_transmission = 0.5 * torch.mean(-torch.log(n2) -1 + n2 + ((beta-t_gt)**2 / eps))
    elif kl_t == 'laplace':
        eps = torch.sqrt(eps/2)
        n2 = torch.div(n2, eps)
        kl_transmission = torch.mean(n2 + torch.exp(-torch.abs(beta-t_gt)/eps) + torch.abs(beta-t_gt)/eps - torch.log(n2) -1)
    elif kl_t == 'lognormal':
        torch.log(1+eps/t_gt)
        n2 = torch.div(n2, eps)
        kl_transmission = torch.div(torch.mean((t_gt-beta)**2 + n2), 2*eps) + torch.mean(torch.log(torch.div(1,n2))
    
    total_loss = lh + kl_dehaze + kl_transmission

    return total_loss, lh, kl_dehaze, kl_transmission
