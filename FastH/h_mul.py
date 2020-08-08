from FastH.fast_h import fast_hmm
from torch.distributions.transforms import Transform, constraints
import torch


class HMulTransform(Transform):

    domain = constraints.real
    codomain = constraints.real
    bijective = True

    def __init__(self, n_vecs, sz_vec, device, requires_grad=True):
        super(HMulTransform, self).__init__()
        self.n_vecs = n_vecs
        self.v = torch.randn((n_vecs, sz_vec), requires_grad=requires_grad, device=device)
        self.H = fast_hmm(self.v, stride=n_vecs//2)
        self.backward_happened = False

        def bw_hook(grad):
            self.backward_happened = True
        self.v.register_hook(bw_hook)

    def __eq__(self, other):
        return isinstance(other, HMulTransform)

    def _call(self, x):
        if self.backward_happened:
            self.H = fast_hmm(self.v, stride=self.n_vecs//2)
            self.backward_happened = False
        return torch.matmul(self.H, x)

    def _inverse(self, y):
        if self.backward_happened:
            self.H = fast_hmm(self.v, stride=self.n_vecs//2)
            self.backward_happened = False
        return torch.matmul(self.H.T, y)

    def log_abs_det_jacobian(self, x, y):
        return torch.tensor(0.0)