import torch
from FastH.fast_h import fast_hmm

class HouseholderLayer(torch.nn.Module):
    """
    concatenations of Householder matrices
    """

    def __init__(self, n_matrices, sample_size, n_in_features, n_out_features, n_hidden, device, requires_grad=True):
        super(HouseholderLayer, self).__init__()
        self.n_matrices = n_matrices
        self.n_hidden = n_hidden
        self.sample_size = sample_size
        self.n_out_features = n_out_features
        self.n_in_features = n_in_features

        v = [torch.randn((self.n_in_features, self.n_out_features, n_matrices, sample_size), requires_grad=requires_grad, device=device)]
        for i in range(self.n_hidden):
            v.append(torch.randn((self.n_out_features, self.n_out_features, n_matrices, sample_size), requires_grad=requires_grad, device=device))

        self.v = torch.nn.ParameterList(v)
        self.H = [fast_hmm(layer) for layer in self.v]

        self.backward_happened = False
        def bw_hook(module, grad_in, grad_out):
            module.backward_happened = True
        self.register_backward_hook(bw_hook)

    def forward(self, x):
        if self.backward_happened:
            self.H = [fast_hmm(layer) for layer in self.v]
            self.backward_happened = False

        return torch.matmul(self.H, x.transpose(-1, -2).unsqueeze(-1)).squeeze(-1).transpose(-1, -2)

    def inverse_forward(self, y):
        if self.backward_happened:
            self.H = [fast_hmm(layer) for layer in self.v]
            self.backward_happened = False
        return torch.matmul(self.H.transpose(-1, -2), y.transpose(-1, -2).unsqueeze(-1)).squeeze(-1).transpose(-1, -2)

    def log_abs_det_jacobian(self, x, y):
        return torch.tensor(0.0)