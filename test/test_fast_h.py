from FastH.fast_h import fast_hmm
import unittest
import torch
import numpy as np
import random


class TestFastH(unittest.TestCase):
    seed = torch.randint(0, 2 ** 32, (1,)).item()
    print("Run Tests with seed: ", seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    def _get_householder_mat(self, v):
        v = v / torch.norm(v, dim=-1, p=2, keepdim=True)
        u = 2 * v
        ID = torch.eye(v.shape[0], device=u.device)
        for dim in range(v.ndim-2):
            ID = ID.unsqueeze(0)
        return ID - torch.matmul(u.unsqueeze(-1), v.unsqueeze(-2))

    def _naive_method_1d(self, v):
        P = []
        for single_v in v:
            P.append(self._get_householder_mat(single_v))
        H = torch.matmul(P[0], P[1])
        for idx in range(2, len(P)):
            H = torch.matmul(H, P[idx])
        return H

    def _naive_method_nd_1(self, v):
        P = []
        for sub_v in v:
            P_sub = []
            for single_v in sub_v:
                P_sub.append(self._get_householder_mat(single_v))
            P.append(P_sub)
        H = []
        for P_sub in P:
            H_sub = torch.matmul(P_sub[0], P_sub[1])
            for idx in range(2, len(P_sub)):
                H_sub = torch.matmul(H_sub, P_sub[idx])
            H.append(H_sub)
        return torch.stack(H, 0)

    def _naive_method_nd_2(self, v):
        P = []
        for sub1_v in v:
            P_sub1 = []
            for sub2_v in sub1_v:
                P_sub2 = []
                for single_v in sub2_v:
                    P_sub2.append(self._get_householder_mat(single_v))
                P_sub1.append(P_sub2)
            P.append(P_sub1)

        H = []
        for P_sub1 in P:
            H_sub1 = []
            for P_sub2 in P_sub1:
                H_sub2 = torch.matmul(P_sub2[0], P_sub2[1])
                for idx in range(2, len(P_sub2)):
                    H_sub2 = torch.matmul(H_sub2, P_sub2[idx])
                H_sub1.append(H_sub2)
            H.append(torch.stack(H_sub1, 0))
        return torch.stack(H, 0)

    def test_1d(self):
        print("\n" + "#" * 30)
        print("Test 1d")
        # set some params
        torch.set_default_tensor_type(torch.DoubleTensor)
        if torch.cuda.device_count() > 0:
            device = "cuda:0"
        else:
            device = 'cpu'

        n_hm = torch.randint(50, 200, (1,)).item()
        n_param = torch.randint(100, 400, (1,)).item()
        stride = torch.randint(1, n_hm, (1,)).item()

        v = torch.randn((n_hm, n_param)).to(device)
        # get a random input matrix X
        input = torch.randn((n_param, n_param)).to(device)

        H_naive = self._naive_method_1d(v)
        out = torch.matmul(H_naive, input)
        inv_out = torch.matmul(H_naive.T, out)

        err1 = (torch.eye(n_param, device=device) - torch.matmul(H_naive.T, H_naive)).abs().sum().item()
        err2 = (inv_out - input).abs().sum().item()
        assert err1 < 1e-8
        assert err2 < 1e-8
        print("Naive method: ")
        print("Identity-test. Error: ", err1)
        print("Inverse-test with an input. Error : ", err2)
        print("\n" + "+" * 30 + "\n")


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
        print("\n" + "+" * 30 + "\n")


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

    def test_nd_1(self):
        print("\n" + "#" * 30)
        print("Test nd with one axis")
        # set some params
        torch.set_default_tensor_type(torch.DoubleTensor)
        if torch.cuda.device_count() > 0:
            device = "cuda:0"
        else:
            device = 'cpu'

        n_d = torch.randint(3, 10, (1,)).item()
        n_hm = torch.randint(50, 100, (1,)).item()
        n_param = torch.randint(100, 200, (1,)).item()
        stride = torch.randint(1, n_hm, (1,)).item()

        v = torch.randn((n_d, n_hm, n_param)).to(device)
        # get a random input matrix X
        input = torch.randn((n_d, n_param, n_param)).to(device)
        ID = torch.eye(n_param, device=device).unsqueeze(0)


        H_naive = self._naive_method_nd_1(v)
        out = torch.matmul(H_naive, input)
        inv_out = torch.matmul(H_naive.transpose(-1, -2), out)

        err1 = (ID - torch.matmul(H_naive.transpose(-1, -2), H_naive)).abs().sum().item()
        err2 = (inv_out - input).abs().sum().item()
        assert err1 < 1e-8
        assert err2 < 1e-8
        print("Naive method: ")
        print("Identity-test. Error: ", err1)
        print("Inverse-test with an input. Error : ", err2)
        print("\n" + "+" * 30 + "\n")


        H_paper = fast_hmm(v, stride, use_pre_q=False)
        out = torch.matmul(H_paper, input)
        inv_out = torch.matmul(H_paper.transpose(-1, -2), out)

        err1 = (ID - torch.matmul(H_paper.transpose(-1, -2), H_paper)).abs().sum().item()
        err2 = (H_paper - H_naive).abs().sum().item()
        err3 = (inv_out - input).abs().sum().item()
        assert err1 < 1e-8
        assert err2 < 1e-8
        assert err3 < 1e-8
        print("Proposed method: ")
        print("Identity-test. Error: ", err1)
        print("Error to naive method: ", err2)
        print("Inverse-test with an input. Error : ", err3)
        print("\n" + "+" * 30 + "\n")


        H_new = fast_hmm(v, stride, use_pre_q=True)
        out = torch.matmul(H_new, input)
        inv_out = torch.matmul(H_new.transpose(-1, -2), out)

        err1 = (ID - torch.matmul(H_new.transpose(-1, -2), H_new)).abs().sum().item()
        err2 = (H_new - H_naive).abs().sum().item()
        err3 = (inv_out - input).abs().sum().item()
        assert err1 < 1e-8
        assert err2 < 1e-8
        assert err3 < 1e-8
        print("leveraging precomputed Qs method: ")
        print("Identity-test. Error: ", err1)
        print("Error to naive method: ", err2)
        print("Inverse-test with an input. Error : ", err3)

    def test_nd_2(self):
        print("\n" + "#" * 30)
        print("Test nd with two axis")
        # set some params
        torch.set_default_tensor_type(torch.DoubleTensor)
        if torch.cuda.device_count() > 0:
            device = "cuda:0"
        else:
            device = 'cpu'

        n_d1 = torch.randint(1, 4, (1,)).item()
        n_d2 = torch.randint(1, 4, (1,)).item()
        n_hm = torch.randint(50, 100, (1,)).item()
        n_param = torch.randint(100, 200, (1,)).item()
        stride = torch.randint(1, n_hm, (1,)).item()

        v = torch.randn((n_d1, n_d2, n_hm, n_param)).to(device)
        # get a random input matrix X
        input = torch.randn((n_d1, n_d2, n_param, n_param)).to(device)
        ID = torch.eye(n_param, device=device).unsqueeze(0)

        H_naive = self._naive_method_nd_2(v)
        out = torch.matmul(H_naive, input)
        inv_out = torch.matmul(H_naive.transpose(-1, -2), out)

        err1 = (ID - torch.matmul(H_naive.transpose(-1, -2), H_naive)).abs().sum().item()
        err2 = (inv_out - input).abs().sum().item()
        assert err1 < 1e-8
        assert err2 < 1e-8
        print("Naive method: ")
        print("Identity-test. Error: ", err1)
        print("Inverse-test with an input. Error : ", err2)
        print("\n" + "+" * 30 + "\n")

        H_paper = fast_hmm(v, stride, use_pre_q=False)
        out = torch.matmul(H_paper, input)
        inv_out = torch.matmul(H_paper.transpose(-1, -2), out)

        err1 = (ID - torch.matmul(H_paper.transpose(-1, -2), H_paper)).abs().sum().item()
        err2 = (H_paper - H_naive).abs().sum().item()
        err3 = (inv_out - input).abs().sum().item()
        assert err1 < 1e-8
        assert err2 < 1e-8
        assert err3 < 1e-8
        print("Proposed method: ")
        print("Identity-test. Error: ", err1)
        print("Error to naive method: ", err2)
        print("Inverse-test with an input. Error : ", err3)
        print("\n" + "+" * 30 + "\n")

        H_new = fast_hmm(v, stride, use_pre_q=True)
        out = torch.matmul(H_new, input)
        inv_out = torch.matmul(H_new.transpose(-1, -2), out)

        err1 = (ID - torch.matmul(H_new.transpose(-1, -2), H_new)).abs().sum().item()
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
