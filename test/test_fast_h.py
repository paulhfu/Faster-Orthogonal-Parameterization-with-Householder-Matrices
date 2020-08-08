from FastH.fast_h import fast_hmm
import unittest
import torch


class TestFastH(unittest.TestCase):

    def test_all(self):
        def get_householder_mat(v):
            v = v / torch.norm(v, dim=-1, p=2, keepdim=True)
            u = 2 * v
            ID = torch.eye(v.shape[0], device=u.device)
            return ID - torch.matmul(u.unsqueeze(-1), v.unsqueeze(0))

        def naive_method(v):
            P = []
            for single_v in v:
                P.append(get_householder_mat(single_v))
            H = torch.matmul(P[0], P[1])
            for idx in range(2, len(P)):
                H = torch.matmul(H, P[idx])
            return H
        # set some params
        torch.set_default_tensor_type(torch.DoubleTensor)
        if torch.cuda.device_count() > 0:
            device = "cuda:0"
        else:
            device = 'cpu'

        n_hm = torch.randint(50, 200, (1,)).item()
        n_param = torch.randint(100, 400, (1,)).item()
        stride = torch.randint(2, 200, (1,)).item()

        v = torch.randn((n_hm, n_param)).to(device)
        # get a random input matrix X
        input = torch.randn((n_param, n_param)).to(device)


        H_naive = naive_method(v)
        out = torch.matmul(H_naive, input)
        inv_out = torch.matmul(H_naive.T, out)

        err1 = (torch.eye(n_param, device=device) - torch.matmul(H_naive.T, H_naive)).abs().sum().item()
        err2 = (inv_out - input).abs().sum().item()
        assert err1 < 1e-8
        assert err2 < 1e-8
        print("Naive method: ")
        print("Identity-test. Error: ", err1)
        print("Inverse-test with an input. Error : ", err2)
        print("\n++++++++++++++++++++++++++++++++++++\n")


        H_paper = fast_hmm(v, stride, use_pre_q=False)
        out = torch.matmul(H_paper, input)
        inv_out = torch.matmul(H_paper.T, out)

        err1 = (torch.eye(n_param, device=device) - torch.matmul(H_paper.T, H_paper)).abs().sum().item()
        err2 = (H_paper - H_naive).abs().sum().item()
        err3 = (inv_out - input).abs().sum().item()
        assert err1 < 1e-8
        assert err2 < 1e-8
        assert err3 < 1e-8
        print("Proposed method: ")
        print("Identity-test. Error: ", err1)
        print("Error to naive method: ", err2)
        print("Inverse-test with an input. Error : ", err3)
        print("\n++++++++++++++++++++++++++++++++++++\n")


        H_new = fast_hmm(v, stride, use_pre_q=True)
        out = torch.matmul(H_new, input)
        inv_out = torch.matmul(H_new.T, out)

        err1 = (torch.eye(n_param, device=device) - torch.matmul(H_new.T, H_new)).abs().sum().item()
        err2 = (H_new - H_naive).abs().sum().item()
        err3 = (inv_out - input).abs().sum().item()
        assert err1 < 1e-8
        assert err2 < 1e-8
        assert err3 < 1e-8
        print("leveraging precomputed Qs method: ")
        print("Identity-test. Error: ", err1)
        print("Error to naive method: ", err2)
        print("Inverse-test with an input. Error : ", err3)


if __name__ == '__main__':
    unittest.main()
