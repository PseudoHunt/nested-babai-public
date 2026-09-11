"""Synthetic unit test of the generalised nested path for coarse b in {2,3}: one-sided layer and two-sided row-chunked layer."""
# run from turboboa/: python ../scripts/test_nested_bits.py
import os, sys, torch, copy
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'turboboa'))
from quantizers.turboboa import TurboBoA
from quantizers.nested import NestedRounder
from quantizers.minmax import MinMaxQuantizer
torch.manual_seed(0)

def make(d_out, d_in, two_sided, nested, bits):
    lay = torch.nn.Linear(d_in, d_out, bias=False).cuda()
    lay.weight.data = W0.clone()
    opts = dict(qparam_comput='Hessian', adaptive_qparam=True, refine_qparam=True, n_iters=1, consider_dX=True, alpha=.25,
                n_quant_rows=16, act_order_col=False, act_order_row=False, nested=nested)
    t = TurboBoA(lay, opts, {'replace': 2**-11})
    t.quantizer = MinMaxQuantizer(); t.quantizer.configure(bits, per_channel=True, group_size=128, sym=False, mse=False)
    t.quantizer.find_params(lay.weight.data)
    t.H_in = H_in.clone(); t.dXXT = dXXT.clone(); t.H_out = H_out.clone() if two_sided else None
    return t

for bits in (2, 3):
    for two_sided in (False, True):
        d_out, d_in = 64, 256
        W0 = torch.randn(d_out, d_in).cuda() * 0.02
        X = torch.randn(4096, d_in).cuda(); H_in = X.T @ X / 4096; dXXT = torch.randn_like(H_in) * 1e-3; dXXT = dXXT @ dXXT.T
        nh = 2; Y = torch.randn(nh, 4096, d_out // nh).cuda(); H_out = torch.einsum('hni,hnj->hij', Y, Y) / 4096
        res = {}
        for arm in (None, 'E3', 'E4', 'B', 'C'):
            nested = None if arm is None else NestedRounder(arm, p=0.01, lam=0.05)
            t = make(d_out, d_in, two_sided, nested, bits); t.quant()
            res[arm] = dict(W=t.layer.weight.data.float().clone(), dump=copy.deepcopy(t.dump))
        maxq_c = 2**bits - 1
        s, z = res['E3']['dump']['scale'], res['E3']['dump']['zero']
        codes_native = res[None]['dump']['q'].int(); codes_e3 = res['E3']['dump']['q4'].int()
        assert (codes_e3 == 2 * codes_native).all(), 'E3 codes != 2*native'
        assert (res['E3']['W'] == res[None]['W']).all() and (s == res[None]['dump']['scale']).all() and (z == res[None]['dump']['zero']).all(), 'E3 not bit-identical'
        for arm in ('E3', 'E4', 'B', 'C'):
            d = res[arm]['dump']; q4 = d['q4'].int(); assert q4.min() >= 0 and q4.max() <= 2 * maxq_c + 1 and int(d['bits']) == bits + 1 and int(d['zero_bits']) == bits
            sf = (d['scale'] / 2).repeat_interleave(128, 1); zf = 2 * d['zero'].float().repeat_interleave(128, 1)
            deq = (sf * (q4.float() - zf)).cuda()
            assert torch.allclose(deq, res[arm]['W'], atol=1e-6), f'{arm}: s_f (q4 - z_f) != stored weights'
        err = {a: ((res[a]['W'] - W0) @ torch.linalg.cholesky(H_in)).pow(2).sum().item() for a in res}
        print(f'bits={bits} two_sided={two_sided}: OK  loss native {err[None]:.4g} E3 {err["E3"]:.4g} E4 {err["E4"]:.4g} B {err["B"]:.4g} C {err["C"]:.4g}  '
              f'refined B {res["B"]["dump"]["refine"].float().mean():.4f} C {res["C"]["dump"]["refine"].float().mean():.4f} free {res["E3"]["dump"]["free"].float().mean():.3f}')
print('ALL OK')
