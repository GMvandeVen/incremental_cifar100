import torch


##-------------------------------------------------------------------------------------------------------------------##

################################################
## Sample tensors from specified distribution ##
################################################

def sample_Logistic(mean, logvar, bins=256.):
    '''Given tensors of [mean] and [logvar] of logistic distribution, sample tensors from specified distributions.'''
    u = torch.rand(mean.size()).to(mean.device)                        # -> sample from U[0,1)
    sample = mean + torch.exp(logvar) * (torch.log(u)-torch.log(1.-u)) # -> convert to Logistic(means, exp(logvar))
    # round to nearest discrete pixel-value
    rounded = torch.floor(sample * bins) / bins
    # clamp outlying values to 0 or 255 and return
    return torch.clamp(rounded, min=1./(bins*2), max=1.-1./(bins*2))

##-------------------------------------------------------------------------------------------------------------------##