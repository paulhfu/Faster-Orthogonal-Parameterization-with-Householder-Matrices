import torch

def fast_hmm(v, stride=None, use_pre_q=False):
    """
    Fast product of a series of Householder matrices. This implementation is oriented to the one introducesd in:
    https://invertibleworkshop.github.io/accepted_papers/pdfs/10.pdf

    :param v: Batched series of Householder matrices. The last dim is the dim of one vector and the second last is the
    number of elements in one product. This is the min amount of dims that need to be present.
    All further ones are considered batch dimensions.
    :param stride: Controls the number of parallel operations by the WY representation (see paper)
    should not be larger than half the number of matrices in one product.
    :param use_pre_q: Use precomputed instances for the final product (different to the paper)
    :return: The batched product of Householder matrices defined by v
    """
    assert v.ndim > 1
    if stride is None:
        stride = v.shape[-2] // 2
    assert stride <= v.shape[-2]

    d, m = v.shape[-2], v.shape[-1]
    k = d // stride
    assert k != 0
    last = k * stride
    v = v / torch.norm(v, dim=-1, p=2, keepdim=True)
    v = v.unsqueeze(-1)
    u = 2 * v
    ID = torch.eye(m, device=u.device)
    for dim in range(v.ndim-3):
        ID = ID.unsqueeze(0)

    # step 1 (compute intermediate groupings P_i)
    W = u[..., 0:last:stride, :, :]
    Y = v[..., 0:last:stride, :, :]

    Q = ID - torch.matmul(W, Y.squeeze(-1).unsqueeze(-2))
    for idx in range(1, stride):
       W = torch.cat([W, torch.matmul(Q, u[..., idx:last:stride, :, :])], dim=-1)
       Y = torch.cat([Y, v[..., idx:last:stride, :, :]], dim=-1)
       Q = ID - torch.matmul(W, Y.transpose(-1, -2))

    # step 2 (multiply the WY reps)
    if use_pre_q:
        H = Q[..., 0, :, :]
        for idx in range(1, k):
            H = torch.matmul(H, Q[..., idx, :, :])
    else:
        H = ID - torch.matmul(W[..., k-1, :, :], Y[..., k-1, :, :].transpose(-1, -2))
        for idx in reversed(range(0, k-1)):
            H = H - torch.matmul(W[..., idx, :, :], torch.matmul(Y[..., idx, :, :].transpose(-1, -2), H))

    # get the WY representation of the residual
    if d > last:
        W_resi = u[..., last, :, :]
        Y_resi = v[..., last, :, :]
        Q_resi = ID - torch.matmul(W_resi, Y_resi.squeeze(-1).unsqueeze(-2))
        for idx in range(last+1, d):
            W_resi = torch.cat([W_resi, torch.matmul(Q_resi, u[..., idx, :, :])], dim=-1)
            Y_resi = torch.cat([Y_resi, v[..., idx, :, :]], dim=-1)
            Q_resi = ID - torch.matmul(W_resi, Y_resi.transpose(-1, -2))

        # and finally the residual factor
        H = torch.matmul(H, Q_resi)

    return H


