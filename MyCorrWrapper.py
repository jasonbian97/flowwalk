from einops import rearrange
import torch.nn as nn
from spatial_correlation_sampler import SpatialCorrelationSampler

class MyCorrWrapper(nn.Module):
    """wrap spatial_correlation_sampler's output to match the Nvidia's corr package
    input: search_range
    """
    def __init__(self,radius):
        super(MyCorrWrapper,self).__init__()
        self.sampler = SpatialCorrelationSampler(
            kernel_size=1,
            patch_size=2*radius+1,
            stride=1,
            padding=0,
            dilation_patch=1
        )

    def forward(self, input1, input2):
        B,C,H,W = input1.shape
        input1 = input1.contiguous()
        input2 = input2.contiguous()

        corr = self.sampler(input1, input2)
        corr = rearrange(corr, "B h w H W -> B (h w) H W")
        corr = corr/C
        return corr