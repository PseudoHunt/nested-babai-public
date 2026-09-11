"""Synthetic test of the per-tensor / per-channel (group_size=-1) paths: coarse-only endpoint == native, one scale per tensor, dequant identity.
run from turboboa/: python ../scripts/test_per_tensor.py"""
import os, sys, torch, copy
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'turboboa'))
from quantizers.turboboa import TurboBoA
from quantizers.nested import NestedRounder
from quantizers.minmax import MinMaxQuantizer
torch.manual_seed(0)

def make(W0, H_in, dXXT, H_out, two_sided, nested, bits, per_tensor):
    lay = torch.nn.Linear(W0.shape[1], W0.shape[0], bias=False).cuda(); lay.weight.data = W0.clone()
    opts = dict(qparam_comput='Hessian', adaptive_qparam=True, refine_qparam=False, n_iters=1, consider_dX=True, alpha=.25,
                n_quant_rows=16, act_order_col=False, act_order_row=False, nested=nested, per_tensor=per_tensor)
    t = TurboBoA(lay, opts, {'replace': 2**-11})
    t.quantizer = MinMaxQuantizer(); t.quantizer.configure(bits, per_channel=True, group_size=-1, sym=False, mse=False)
    t.quantizer.find_params(lay.weight.data)
    t.H_in = H_in.clone(); t.dXXT = dXXT.clone(); t.H_out = H_out.clone() if two_sided else None
    return t

for per_tensor in (True, False):
    for two_sided in (False, True):
        d_out, d_in, bits = 64, 256, 3
        W0 = torch.randn(d_out, d_in).cuda() * 0.02; W0[3, 7] = 0.3   # an outlier row/col
        X = torch.randn(4096, d_in).cuda(); H_in = X.T @ X / 4096; dXXT = torch.randn_like(H_in) * 1e-3; dXXT = dXXT @ dXXT.T
        Y = torch.randn(2, 4096, d_out // 2).cuda(); H_out = torch.einsum('hni,hnj->hij', Y, Y) / 4096
        res = {}
        for arm in (None, 'E3', 'E4', 'B', 'C'):
            nested = None if arm is None else NestedRounder(arm, p=0.01, lam=0.05)
            t = make(W0, H_in, dXXT, H_out, two_sided, nested, bits, per_tensor); t.quant()
            res[arm] = dict(W=t.layer.weight.data.float().clone(), dump=copy.deepcopy(t.dump))
        s = res[None]['dump']['scale']; z = res[None]['dump']['zero']
        assert s.shape == (d_out, 1), s.shape
        if per_tensor:
            assert (s == s[0, 0]).all() and (z == z[0, 0]).all(), 'per-tensor: scale/zero not constant across rows'
            assert res[None]['dump'].get('per_tensor') is True
        else:
            assert (s != s[0, 0]).any(), 'per-channel: expected row-dependent scales'
        assert (res['E3']['dump']['q4'].int() == 2 * res[None]['dump']['q'].int()).all() and (res['E3']['W'] == res[None]['W']).all() \
            and (res['E3']['dump']['scale'] == s).all() and (res['E3']['dump']['zero'] == z).all(), 'coarse-only endpoint != native'
        for arm in ('E3', 'E4', 'B', 'C'):
            d = res[arm]['dump']; q4 = d['q4'].int()
            deq = ((d['scale'] / 2) * (q4.float() - 2 * d['zero'].float())).cuda()
            assert torch.allclose(deq, res[arm]['W'], atol=1e-6), f'{arm}: dequant mismatch'
        err = {a: ((res[a]['W'] - W0) @ torch.linalg.cholesky(H_in)).pow(2).sum().item() for a in res}
        print(f'per_tensor={per_tensor} two_sided={two_sided}: OK  loss native {err[None]:.4g} E4 {err["E4"]:.4g} B {err["B"]:.4g} C {err["C"]:.4g}  '
              f'scale {s[0,0].item():.4g} zero {z[0,0].item():.0f}  refined B {res["B"]["dump"]["refine"].float().mean():.4f}')
print('ALL OK')
