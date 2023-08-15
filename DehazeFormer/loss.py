import torch 
from math import pi, log, sqrt


MAX, MIN = 1e4, 1e-10

def vlb_loss(inp, d_out, t_out, d_gt, t_gt, A, sigma, eps1, eps2, kl_j, kl_t):
    alpha = d_out[:,:3]
    m2 = torch.clamp(d_out[:,3:], min=1e-10)
    beta = t_out[:,:1]
    n2 = torch.clamp(t_out[:,1:], min=1e-10)

    # Likelihood
    lh = 0.5 * torch.log(torch.tensor(2*pi)) + 0.5 * torch.log(torch.tensor(sigma)) + 0.5 * torch.mean(((inp - (alpha*beta) - A*(1-beta))**2)/sigma + 1)

    # KL divergence for dehazer
    if kl_j == 'gaussian':
        m2_div_eps1 = torch.div(m2, eps1)
        kl_dehaze = 0.5 * torch.mean(-torch.log(m2_div_eps1) -1 + m2_div_eps1 + ((alpha-d_gt)**2 / eps1))
    elif kl_j == 'laplace':
        m2_div_eps1 = torch.div(m2, eps1)
        kl_dehaze = torch.mean(m2_div_eps1 + torch.exp(-torch.abs(alpha-d_gt)/eps1) + torch.abs(alpha-d_gt)/eps1 - torch.log(m2_div_eps1) -1)

    # KL divergence for transmission
    if kl_t == 'gaussian':
        n2_div_eps2 = torch.div(n2, eps2)
        kl_transmission = 0.5 * torch.mean(-torch.log(n2_div_eps2) -1 + n2_div_eps2 + ((beta-t_gt)**2 / eps2))
    elif kl_t == 'laplace':
        n2_div_eps2 = torch.div(n2, eps2)
        kl_transmission = torch.mean(n2_div_eps2 + torch.exp(-torch.abs(beta-t_gt)/2*eps2) + torch.abs(beta-t_gt)/eps2 - torch.log(n2_div_eps2) -1)
    elif kl_t == 'lognormal':
        kl_transmission = torch.div(torch.mean((torch.log(beta)-torch.log(t_gt))**2 + n2),2*eps2) - 0.5 + torch.log(torch.mean(torch.div(eps2,n2)))

    print('lh', lh)
    print('kl_dehaze', kl_dehaze)
    print('kl_trans', kl_transmission)
    total_loss = lh + kl_dehaze + kl_transmission
    return total_loss, lh, kl_dehaze, kl_transmission

def loss_val(d_out, d_gt, eps1, kl_j):
    alpha = d_out[:,:3]
    m2 = torch.exp(d_out[:,3:])#.clamp_(min=log_min, max=log_max)

    # KL divergence for dehazer
    m2_div_eps1 = torch.div(m2, eps1)
    if kl_j == 'gaussian':
        kl_dehaze = 0.5 * torch.mean(-torch.log(m2_div_eps1) -1 + m2_div_eps1 + ((alpha-d_gt)**2 / eps1))
    elif kl_j == 'laplace':
        kl_dehaze = torch.mean(m2_div_eps1 + torch.exp(-torch.abs(alpha-d_gt)/eps1) + torch.abs(alpha-d_gt)/eps1 - torch.log(m2_div_eps1) -1)

    return kl_dehaze

def vlb_loss_value(inp, d_out, t_out, d_gt, t_gt, A, sigma, eps1, eps2, kl_j, kl_t):

    lh = 0.5 * torch.mean((inp - (d_out*t_out) - A*(1-t_out))**2)/sigma

    if kl_j == 'gaussian':
        kl_dehaze = 0.5 * torch.mean((d_out-d_gt)**2)/eps1
    if kl_j == 'laplace':
        kl_dehaze = torch.mean(torch.exp(-torch.abs(d_out - d_gt)/eps1) + torch.abs(d_out - d_gt)/eps1)

    if kl_t == 'gaussian':
        kl_transmission = 0.5 * torch.mean((t_out - t_gt)**2)/eps2
    if kl_t == 'lognormal':
        kl_transmission = 0.5 * torch.mean((torch.log(t_out) - torch.log(t_gt))**2)/eps2

    total_loss = lh + kl_dehaze + kl_transmission
    return total_loss, lh, kl_dehaze, kl_transmission

def loss_val_value(d_out, d_gt, eps1, kl_j):
    # KL divergence for dehazer
    if kl_j == 'gaussian':
        kl_dehaze = torch.mean((d_out-d_gt)**2 / eps1)
    elif kl_j == 'laplace':
        kl_dehaze = torch.mean(torch.exp(-torch.abs(d_out-d_gt)/eps1) + torch.abs(d_out-d_gt)/eps1)

    return kl_dehaze