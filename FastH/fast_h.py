import torch

def fast_hmm(v, stride, use_pre_q=False):
    assert v.ndim == 2
    assert v.shape[0]//stride != 0

    d, m = v.shape[0], v.shape[1]
    k = d // stride
    assert k != 0
    last = k * stride
    v = v / torch.norm(v, dim=-1, p=2, keepdim=True)
    u = 2 * v

    # step 1 (computer intermediate groupings P_i)
    W = u[0:last:stride].unsqueeze(-1)
    Y = v[0:last:stride].unsqueeze(-1)
    ID = torch.eye(m, device=u.device)
    Q = ID - torch.matmul(W, Y.squeeze().unsqueeze(-2))
    for idx in range(1, stride):
       W = torch.cat([W, torch.matmul(Q, u[idx:last:stride].unsqueeze(-1))], dim=-1)
       Y = torch.cat([Y, v[idx:last:stride].unsqueeze(-1)], dim=-1)
       Q = ID - torch.matmul(W, Y.transpose(1, 2))

    # step 2 (multiply the WY reps)
    if use_pre_q:
        H = Q[0]
        for idx in range(1, k):
            H = torch.matmul(H, Q[idx])
    else:
        H = ID - torch.matmul(W[k-1], Y[k-1].transpose(0, 1))
        for idx in reversed(range(0, k-1)):
            H = H - torch.matmul(W[idx], torch.mm(Y[idx].transpose(0, 1), H))

    # get the WY representation of the residual
    if d > last:
        W_resi = u[last].unsqueeze(-1)
        Y_resi = v[last].unsqueeze(-1)
        Q_resi = ID - torch.matmul(W_resi, Y_resi.squeeze().unsqueeze(0))
        for idx in range(last+1, d):
            W_resi = torch.cat([W_resi, torch.matmul(Q_resi, u[idx].unsqueeze(-1))], dim=-1)
            Y_resi = torch.cat([Y_resi, v[idx].unsqueeze(-1)], dim=-1)
            Q_resi = ID - torch.matmul(W_resi, Y_resi.transpose(0, 1))

        # and finally the residual factor
        H = torch.matmul(H, Q_resi)

    return H


