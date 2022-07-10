import torch 
import torch.nn as nn
from math import pi, log 

def loss_fn(input, out_dehaze, out_transmission, gt_dehaze, gt_transmission, A, sigma, eps1, eps2):
    '''
    Input: 
        eps1 : variance for prior of Z (epsilon^2를 기본으로 줄 것임)
        eps2 : variance for prior of T 
        sigma : variance of y 
    '''

    # parameters predicted of dehaze 
    alpha = out_dehaze[:, :3]
    m2 = out_dehaze[:,3:]

    # parameters predicted of transmission
    beta = out_transmission[:, :3]
    n2 = out_transmission[:, :3]

    # Likelihood
    lh = 0.5 * torch.mean(-log(2*pi) - log(sigma) + (1/sigma)*((input - alpha*beta - A*(1-beta)**2) + sigma))

    # KL divergence for dehaze 
    m2_div_eps1 = torch.div(m2, eps1)
    kl_dehaze = 0.5 * torch.mean(-log(m2_div_eps1) -1 + m2_div_eps1 + (alpha-gt_dehaze)**2 / eps1)

    # KL divergence for transmission 
    n2_div_eps2 = torch.div(n2, eps2)
    kl_transmission = 0.5 * torch.mean(-log(n2_div_eps2) -1 + n2_div_eps2 + (beta-gt_transmission)**2 / eps2)    

    total_loss = kl_transmission + kl_dehaze + lh

    return total_loss