"""
Nested-lattice refinement rule for the TurboBoA column loop.

Coarse grid (b-bit, maxq_c = 2^b - 1): q_c = clamp(round(w/s_c) + z_c, 0, maxq_c),  deq_c = s_c (q_c - z_c)
Fine grid ((b+1)-bit): s_f = s_c/2, z_f = 2 z_c, q_f = clamp(round(w/s_f) + z_f, 0, 2 maxq_c + 1), deq_f = s_f (q_f - z_f)
Stored code: q4 = q_f if refined else 2 q_c.  Dequant is always s_f (q4 - z_f).
The main experiment is b = 3 (codes 0..15); the 2-bit experiment is b = 2 (coarse 0..3, fine 0..7).

The scale/zero search stays on the 3-bit grid (done by TurboBoA's own find_params_H); this module only
replaces the rounding call inside the column loop and decides, per element, whether to refine.
The error propagated by the solver is that of the single stored value.
"""
import math
import torch

ARMS = ("E3", "E4", "A", "B", "C", "D")


class NestedRounder:
    def __init__(self, arm, p=0.01, lam=None, log_percol=False):
        assert arm in ARMS, arm
        self.arm = arm
        self.p = float(p)
        self.lam = float(lam) if lam is not None else None
        if arm in ("C", "D"):
            assert self.lam is not None, "arm C/D need --nested_lambda"
        self.log_percol = log_percol
        self.maxq_c = 7        # coarse-grid max code (2^b - 1); set per layer from the solver's quantizer
        self.reset_layer()

    # ------------------------------------------------------------------ per-layer state
    def reset_layer(self):
        self.N = None          # layer normalizer N_l (solver loss units per element)
        self.mask = None       # arm A: bool tensor shaped like W [nh, rows, ng, gs]
        self.fisher = None     # arm D: float tensor shaped like W, already divided by mean(F_l)
        self.n_cols = None

    def begin_layer(self, W, H_in, H_out, mask=None, fisher=None, maxq_c=7):
        """W: [nh, rows, ng, gs] (post-preprocess, post-reorder). H_in: [nH, d, d]. H_out: [nh, rows_h, rows_h] or None."""
        from utils.quant_utils import compute_loss_degradation
        self.reset_layer()
        self.maxq_c = int(maxq_c)
        # N_l = (solver's own loss functional evaluated on W itself) / numel.
        # one-sided: sum_h tr(W_h H_in,h W_h^T); two-sided (q/k): sum_h tr(H_out,h W_h H_in W_h^T)
        self.N = (compute_loss_degradation(W, torch.zeros_like(W), H_in, H_out, None) / W.numel()).item()
        self.n_cols = W.shape[-2] * W.shape[-1]
        if self.arm == "A":
            assert mask is not None
            self.mask = mask
        if self.arm == "D":
            assert fisher is not None
            self.fisher = fisher / fisher.float().mean().clamp_min(1e-30)

    # ------------------------------------------------------------------ the rounding rule
    def round_column(self, w, s_c, z_c, d_j, r_i, col, sub=None):
        """
        w    : compensated weight column, [nh, rows, 1]
        s_c,z_c : 3-bit group qparams, [nh, rows, 1]
        d_j  : solver column weight 1/U_in[j,j]^2, broadcastable [nH,1,1]
        r_i  : per-row weight (two-sided layers) [nh, rows, 1] or None
        col  : global column index (after reordering) -- used for arm-B accumulator
        sub  : (row_slice, group_idx, c1) for looking up mask / fisher aligned with W
        returns deq [nh,rows,1], q4 [nh,rows,1] (float, integer-valued), gain [nh,rows,1], refine bool, free bool
        """
        s_f = s_c / 2
        z_f = 2 * z_c
        q_c = torch.clamp(torch.round(w / s_c) + z_c, 0, self.maxq_c)
        q_f = torch.clamp(torch.round(w / s_f) + z_f, 0, 2 * self.maxq_c + 1)
        deq_c = s_f * (2 * q_c - z_f)   # == s_c * (q_c - z_c) bit-exactly
        deq_f = s_f * (q_f - z_f)
        e_c = w - deq_c
        e_f = w - deq_f
        gain = (e_c * e_c - e_f * e_f) * d_j
        if r_i is not None:
            gain = gain * r_i
        gain = gain.clamp_min_(0)
        free = (q_f == 2 * q_c)

        arm = self.arm
        if arm == "E3":
            refine = torch.zeros_like(free)
        elif arm == "E4":
            refine = torch.ones_like(free)
        elif arm == "A":
            rs, g, c1 = sub
            refine = self.mask[:, rs, g, c1].unsqueeze(-1)
        elif arm == "B":
            n = gain.numel()
            pn = self.p * n
            k = int(math.floor(pn * (col + 1)) - math.floor(pn * col))
            refine = torch.zeros_like(free)
            if k > 0:
                gflat = gain.reshape(-1)
                idx = torch.topk(gflat, k).indices
                rflat = refine.reshape(-1)
                rflat[idx] = True
                refine = rflat.reshape(gain.shape) & (gain > 0)
        elif arm == "C":
            refine = (gain / self.N) > self.lam
        elif arm == "D":
            rs, g, c1 = sub
            refine = (gain * self.fisher[:, rs, g, c1].unsqueeze(-1) / self.N) > self.lam
        else:
            raise NotImplementedError(arm)

        q4 = torch.where(refine, q_f, 2 * q_c)
        deq = s_f * (q4 - z_f)
        return deq, q4, gain, refine, free
