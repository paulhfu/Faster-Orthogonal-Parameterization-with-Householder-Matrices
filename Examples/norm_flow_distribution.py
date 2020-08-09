from torch.distributions.transforms import SigmoidTransform
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.utils import _sum_rightmost
import torch

class NormFlowDis(TransformedDistribution):

    def __init__(self, prior, transforms, sample_shape):
        """
        A transformed distribution with fixed sample shape
        :param prior: Base distribution
        :param transforms: list of serial transformations
        :param sample_shape: fixed sample shape
        """
        self.base_dist = prior
        self.sample_shape = sample_shape
        super().__init__(self.base_dist, transforms)

    def rsample(self):
        """
        generates an reparameterized sample of the fixed sample shape
        """
        x = self.base_dist.rsample(self.sample_shape)
        for transform in self.transforms:
            x = transform(x)
        return x

