import torch
import matplotlib.pyplot as plt
import numpy as np
from Examples.h_transforms import HMulTransform2D
from torch.distributions import Normal
from norm_flow_distribution import NormFlowDis
from utils import ind_flat_2_spat
from scipy.ndimage import gaussian_filter
from tensorboardX import SummaryWriter
from skimage.draw import circle
from torch.distributions import SigmoidTransform


if torch.cuda.device_count() > 0:
    device = "cuda:0"
else:
    device = 'cpu'

sz = 64
sample_size = 1024

flow_net = [HMulTransform2D(2, sample_size, device),
            # SigmoidTransform(),
            HMulTransform2D(2, sample_size, device),
            # SigmoidTransform(),
            HMulTransform2D(2, sample_size, device),
            # SigmoidTransform(),
            HMulTransform2D(2, sample_size, device),
            # SigmoidTransform(),
            HMulTransform2D(2, sample_size, device),
            SigmoidTransform(),
            ]

params = []
for module in flow_net:
    if isinstance(module, HMulTransform2D):
        params.append(module.v)

optim = torch.optim.Adam(params, lr=1e-5)

# prior = MultivariateNormal(torch.zeros(2, device=device), torch.eye(2, device=device))
# prior = TransformedDistribution(Uniform(torch.zeros(1, device=device), torch.ones(1, device=device)), SigmoidTransform())
prior = Normal(torch.zeros(2, device=device), torch.ones(2, device=device))

norm_flow_distribution = NormFlowDis(prior, flow_net, (sample_size, ))

# get a 2d spiral as target distribution
tgt_dis = torch.ones((sz, sz), dtype=torch.float) * 0.1
radii = torch.linspace(0, sz//2-1, 1000)
phis = torch.linspace(0, 10*np.pi, 1000)
# for rad, phi in zip(radii, phis):
#     tgt_dis[sz//2 + int(np.sin(phi)*rad), sz//2 + int(np.cos(phi)*rad)] = 8
circle = circle(sz//2, sz//2, sz//20)
tgt_dis[circle[0], circle[1]] = 10

# plt.imshow(tgt_dis.numpy());plt.show()
tgt_dis = torch.from_numpy(gaussian_filter(tgt_dis, 4.0))

flat_tgt_dis = torch.softmax(tgt_dis.view(-1), dim=0).to(device)
tgt_dis = flat_tgt_dis.view(sz, sz)
loss_func = torch.nn.KLDivLoss(reduction='batchmean')
writer = SummaryWriter(logdir="./logs")
loss_counter = 0
plt.imshow(tgt_dis.cpu().numpy())
plt.title("target distribution")
plt.show()

# learn by sampling from the target distribution

for epoch in range(10):
    for it0 in range(15):
        sample = torch.multinomial(flat_tgt_dis, sample_size).float()
        sample = ind_flat_2_spat(sample, (sz, sz))
        sample = torch.from_numpy((sample.astype(np.float) + 1e-6) / (sz - 1e-6)).float().to(device)  # prevent 0 and 1 since only in the limit of sigmoid

        # sample_collection = torch.zeros((sz, sz,), dtype=torch.float)
        # sample_collection[(sample[:, 0]*sz).long(), (sample[:, 1]*sz).long()] = 1
        # plt.imshow(sample_collection.detach().cpu().numpy());plt.show()

        log_probs = norm_flow_distribution.log_prob(sample)
        loss = -log_probs.mean()

        optim.zero_grad()
        loss.backward()
        optim.step()
        loss_counter += 1
        print(loss.item())
        writer.add_scalar("m1/loss", loss, loss_counter)

    sample_collection = torch.zeros((sz, sz), dtype=torch.float)
    for it1 in range(4):
        sample = norm_flow_distribution.rsample()
        sample = (sample * sz).long()
        # valid = ((sample >= 0) & (sample < sz)).sum(-1) == 2
        sample_collection[sample[:, 0], sample[:, 1]] += 1

    plt.clf()
    fig = plt.figure(frameon=False)
    plt.imshow(sample_collection.reshape((sz, sz)).cpu().numpy())
    plt.title("predicted samples")
    plt.show()
    writer.add_figure("m1/samples", fig, loss_counter % 500)
    for i, module in enumerate(flow_net):
        if isinstance(module, HMulTransform2D):
            writer.add_histogram(f"m1/module_{i}_parameter_dis", module.v.view(-1).detach().cpu().numpy(), loss_counter % 500)


