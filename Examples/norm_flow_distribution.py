from torch.distributions.transforms import SigmoidTransform
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.utils import _sum_rightmost
import torch

class NormFlowDis(TransformedDistribution):

    def __init__(self, prior, transforms, sample_shape, device):

        self.base_dist = prior
        self.sample_shape = sample_shape
        super().__init__(self.base_dist, transforms)

    def rsample(self):
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched. Samples first from base distribution and applies
        `transform()` for every transform in the list.
        """
        x = self.base_dist.rsample(self.sample_shape)
        for transform in self.transforms:
            x = transform(x)
        return x

