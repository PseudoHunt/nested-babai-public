import torch

from quantizers.utils import get_cholesky_of_inverse, reorder_col, reorder_row, reverse_reorder_col, reverse_reorder_row
from utils.quant_utils import (
    compute_loss_degradation,
    damping,
    fake_quantize,
    filter_dead_neuron,
    optimize_group_qparams,
    quantize,
)
from utils.utils import cleanup_memory

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

class TurboBoA:
    def __init__(self, layer, opts, hyperparams):
        self.layer = layer
        W = self.layer.weight.data
        self.org_shape, self.org_dtype = W.shape, W.dtype

        self.quantizer = None
        self.H_in = None
        self.H_out = None
        self.dXXT = None

        self.qparam_comput = opts['qparam_comput']
        self.adaptive_qparam = opts['adaptive_qparam']
        self.refine_qparam = opts['refine_qparam']
        self.n_iters = opts['n_iters']
        self.consider_dX = opts['consider_dX']
        self.alpha = opts['alpha']
        self.n_quant_rows = opts['n_quant_rows']
        self.act_order_col = opts['act_order_col']
        self.act_order_row = opts['act_order_row']
        self.hyperparams = hyperparams

        # --- nested-lattice refinement (flag-guarded; None => stock TurboBoA path) ---
        self.nested = opts.get('nested', None)
        self.nested_mask = None     # arm A: bool [d_out, d_in]
        self.nested_fisher = None   # arm D: float [d_out, d_in]
        self.dump = {}              # integer codes / qparams / logs collected for the rate script


    def quant(self, print_memory_usage=False):
        assert self.quantizer is not None, "Quantizer should be defined first."
        assert self.H_in is not None, "Hessian should be computed first."

        W, H_in, H_out, dXXT = self.preprocess()
        
        # Hessian-based re-ordering for columns
        if self.act_order_col:
            W, H_in, dXXT, invperm_col = reorder_col(W, H_in, dXXT)

        U_in = get_cholesky_of_inverse(H_in)
        P = (dXXT @ U_in.transpose(-1, -2)).triu(diagonal=1) @ U_in if dXXT is not None else None 
        
        if self.nested is not None:
            assert not self.act_order_col and not self.act_order_row, "nested path: act-order permutation of the mask not implemented"
            assert self.quantizer.maxq.item() in (3, 7), "nested path: coarse grid must be --w_bits 2 or 3"
            mask = self.nested_mask.reshape(W.shape).to(W.device) if self.nested_mask is not None else None
            fisher = self.nested_fisher.reshape(W.shape).to(W.device).float() if self.nested_fisher is not None else None
            self.nested.begin_layer(W, H_in, H_out, mask=mask, fisher=fisher, maxq_c=int(self.quantizer.maxq.item()))
            self.dump['N'] = self.nested.N
        self._ex = None

        if H_out is None:
            Q, scale, zero = self.gptq(W, H_in, U_in, P)
            if self.refine_qparam:
                Q, scale = self.optimize_qparam_gptq(W, Q, H_in, dXXT, scale, zero, self.quantizer.maxq)
        else: 
            # Hessian-based re-ordering for rows
            if self.act_order_row:
                W, H_out, invperm_row  = reorder_row(W, H_out)
            
            Q, scale, zero = self.turboboa(W, H_in, U_in, H_out, self.n_quant_rows, dXXT, P)
            if self.refine_qparam:
                Q, scale = self.optimize_qparam_turboboa(W, Q, H_in, H_out, dXXT, scale, zero, self.quantizer.maxq)
            
            # reverse re-ordering for columns
            if self.act_order_row:
                Q, scale, zero = reverse_reorder_row(Q, scale, zero, invperm_row)
        
        # reverse re-ordering for columns
        if self.act_order_col:
            Q = reverse_reorder_col(Q, invperm_col)
        
        # --- dump integer codes + qparams (exact, fp32) for the rate script; nested extras if present ---
        d_out, d_in = self.org_shape
        if self.nested is not None:
            ex = self._ex
            self.dump['q4'] = ex['q4'].reshape(d_out, d_in).to(torch.uint8).cpu()
            self.dump['refine'] = ex['refine'].reshape(d_out, d_in).cpu()
            self.dump['free'] = ex['free'].reshape(d_out, d_in).cpu()
            self.dump['gain_over_N'] = (ex['gain'].reshape(d_out, d_in) / max(self.nested.N, 1e-30)).cpu()
            self.dump['bits'] = int(self.quantizer.n_bits) + 1      # stored codes are (b+1)-bit
            self.dump['zero_bits'] = int(self.quantizer.n_bits)     # but the stored zero is the coarse b-bit one
        else:
            self.dump['q'] = quantize(Q, scale, zero, self.quantizer.maxq).reshape(d_out, d_in).to(torch.uint8).cpu()
            self.dump['bits'] = int(self.quantizer.n_bits)
            self.dump['zero_bits'] = int(self.quantizer.n_bits)
        self.dump['scale'] = scale.reshape(d_out, -1).cpu()   # [d_out, n_groups]  (coarse scale s_c on the nested path)
        self.dump['zero'] = zero.reshape(d_out, -1).to(torch.uint8).cpu()
        self._ex = None
        
        self.free()
        if print_memory_usage:
            print(f'\t |GPU memory: {torch.cuda.max_memory_allocated("cuda") / 1024**3:.3f}|')

        # assign quantized (fake-quant) weights
        self.layer.weight.data = Q.reshape(self.org_shape).to(dtype=self.org_dtype, device="cuda:0")
        self.quantizer.scale = scale.reshape(self.quantizer.scale.shape).to(device="cuda:0")
        self.quantizer.zero = zero.reshape(self.quantizer.zero.shape).to(device="cuda:0")


    def gptq(self, W, H_in, U_in, P, row_w=None, row_slice=slice(None)):
        org_shape = W.shape
        n_groups, group_size = org_shape[-2], org_shape[-1]

        Q = torch.zeros_like(W)
        W_update = W.clone()
        nested = self.nested
        if nested is not None:
            Q4 = torch.zeros_like(W)
            G = torch.zeros_like(W)
            R = torch.zeros(W.shape, dtype=torch.bool, device=W.device)
            F = torch.zeros(W.shape, dtype=torch.bool, device=W.device)
            Uinv2 = (1.0 / U_in.diagonal(dim1=-2, dim2=-1) ** 2).unsqueeze(-1).unsqueeze(-1)  # d_j = 1/U_in[j,j]^2, [nH,1,1] per column below
        scale, zero = torch.ones((*W.shape[:-1], 1), device=W.device), torch.zeros((*W.shape[:-1], 1), device=W.device)
        for idx_group in range(n_groups):
            c_start = idx_group * group_size
            if self.qparam_comput == "MinMax":
                scale_group, zero_group = self.quantizer.find_params_H(W[..., idx_group, :], None, search=False)
            elif self.qparam_comput == "MMSE":
                scale_group, zero_group = self.quantizer.find_params_H(W[..., idx_group, :], None, search=True)
            elif self.qparam_comput == "Hessian":
                scale_group, zero_group = self.quantizer.find_params_H(W[..., idx_group, :], H_in[..., c_start:c_start+group_size, c_start:c_start+group_size], search=True)
            else:
                raise NotImplementedError()
            scale[..., idx_group, :] = scale_group
            zero[..., idx_group, :] = zero_group

            for c1 in range(group_size):
                w = W_update[..., idx_group, c1].unsqueeze(-1)
                if nested is None:
                    q = fake_quantize(w, scale_group, zero_group, self.quantizer.maxq)
                else:
                    q, q4, gain, refine, free = nested.round_column(
                        w, scale_group, zero_group, Uinv2[..., c_start+c1, :, :], row_w, c_start+c1, sub=(row_slice, idx_group, c1))
                    Q4[..., idx_group, c1] = q4.squeeze(-1)
                    G[..., idx_group, c1] = gain.squeeze(-1)
                    R[..., idx_group, c1] = refine.squeeze(-1)
                    F[..., idx_group, c1] = free.squeeze(-1)
                Q[..., idx_group, c1] = q.squeeze(-1)

                err = (w - q) / U_in[..., c_start+c1, c_start+c1].unsqueeze(-1).unsqueeze(-1)

                if P is not None:
                    update = err @ U_in[..., c_start+c1, c_start+c1:].unsqueeze(-2) - w @ P[..., c_start+c1, c_start+c1:].unsqueeze(-2)
                else:
                    update = err @ U_in[..., c_start+c1, c_start+c1:].unsqueeze(-2)
                W_update[..., idx_group, c1:] -= update[..., :group_size-c1]
                W_update[..., idx_group+1:, :] -= update[..., group_size-c1:].reshape(*org_shape[:-2], n_groups-idx_group-1, group_size)

        if nested is not None:
            ex = dict(q4=Q4, gain=G, refine=R, free=F)
            if self._ex is None:
                self._ex = ex
            else:  # row-chunked call from turboboa(): append along the row axis
                for k in ex:
                    self._ex[k] = torch.cat([self._ex[k], ex[k]], dim=1)
        return Q, scale, zero    


    def turboboa(self, W, H_in, U_in, H_out, N_rows, dXXT, P):
        org_shape = W.shape
        n_heads, head_dim, n_groups, group_size = org_shape
        U_out = get_cholesky_of_inverse(H_out)
        dev = W.device

        RHinv = dXXT @ U_in.transpose(-1, -2) @ U_in if dXXT is not None else None
        if not self.adaptive_qparam:
            if self.qparam_comput == "MinMax":
                scale, zero = self.quantizer.find_params_H_group(W, None, search=False)
            elif self.qparam_comput == "MMSE":
                scale, zero = self.quantizer.find_params_H_group(W, None, search=True)
            elif self.qparam_comput == "Hessian":
                scale, zero = self.quantizer.find_params_H_group(W, H_in, search=True)
            else:
                raise NotImplementedError()
        else:
            scale, zero = self.quantizer.scale.reshape(*org_shape[:-2], n_groups, 1), self.quantizer.zero.reshape(*org_shape[:-2], n_groups, 1)
        
        Q = torch.zeros_like(W)
        W_update = W.clone()
        for r1 in range(0, head_dim, N_rows):
            r2 = min(r1 + N_rows, head_dim)

            if N_rows != 1:
                U_out_sub_inv = torch.linalg.solve_triangular(U_out[:, r1:r2, r1:r2], torch.eye(r2-r1).repeat(n_heads, 1, 1).to(dev), upper=True)
            else:
                U_out_sub_inv = 1 / U_out[:, r1:r2, r1:r2]
            P2 = U_out.transpose(-1, -2)[:, r2:, r1:r2] @ U_out_sub_inv.transpose(-1, -2)

            # per-row OBS weight of this chunk's rows in the solver's own loss: [(U_sub^T U_sub)^-1]_ii
            row_w = (U_out_sub_inv ** 2).sum(-1).unsqueeze(-1) if self.nested is not None else None
            Q_sub, scale_sub, zero_sub = self.gptq(W_update[:, r1:r2, :, :].clone(), H_in, U_in, P, row_w=row_w, row_slice=slice(r1, r2))
            Q[:, r1:r2, :, :] = Q_sub
            scale[:, r1:r2, :, :] = scale_sub
            zero[:, r1:r2, :, :] = zero_sub

            if RHinv is not None:
                update_rows = torch.einsum('hrn, hngd -> hrgd', P2, (W_update[:, r1:r2, :, :].reshape(n_heads, r2-r1, -1) @ RHinv).reshape(n_heads, r2-r1, n_groups, group_size) + (W_update[:, r1:r2, :, :] - Q_sub))
            else:
                update_rows = torch.einsum('hrn, hngd -> hrgd', P2, (W_update[:, r1:r2, :, :] - Q_sub))
            W_update[:, r2:, :, :] -= update_rows
        
        return Q, scale, zero


    def optimize_qparam_gptq(self, W, Q, H_in, dXXT, scale, zero, maxq):
        Q_int = quantize(Q, scale, zero, maxq)
        Q_int_shifted = Q_int - zero
        
        n_heads, head_dim, n_groups, group_size = Q.shape
        hidden_size = n_groups * group_size
        loss_orig = compute_loss_degradation(W, Q, H_in, None, dXXT)
        if n_groups == 1:
            W = W.reshape(n_heads, head_dim, hidden_size)
            Q = Q.reshape(n_heads, head_dim, hidden_size)
            Q_int_shifted = Q_int_shifted.reshape(n_heads, head_dim, hidden_size)
            QH = Q_int_shifted @ H_in
            scale = scale.reshape(n_heads, head_dim)
            if dXXT is not None:
                WdXXTQ = W @ dXXT @ Q_int_shifted.transpose(-1, -2)
            denominator = torch.diagonal(QH @ Q_int_shifted.transpose(-1, -2), dim1=-2, dim2=-1)
            numerator = torch.diagonal(QH @ (W - Q).transpose(-1, -2), dim1=-2, dim2=-1)
            if dXXT is not None:
                numerator += torch.diagonal(WdXXTQ, dim1=-2, dim2=-1)
            scale += numerator / denominator
            Q = scale.unsqueeze(-1) * Q_int_shifted
            
            loss_new = compute_loss_degradation(W, Q, H_in, None, dXXT)
            print(f"Loss: {loss_orig:.2f} -> {loss_new:.2f}")
            Q = Q.reshape(n_heads, head_dim, n_groups, group_size)

            return Q, scale
        
        else:
            print('NOT supported for group-wise quantization yet')
            return Q, scale


    def optimize_qparam_turboboa(self, W, Q, H_in, H_out, dXXT, scale, zero, maxq):
        Q_int = quantize(Q, scale, zero, maxq)
        Q_int_shifted = Q_int - zero
        
        n_heads, head_dim, n_groups, group_size = Q.shape
        hidden_size = n_groups * group_size
        loss_orig = compute_loss_degradation(W, Q, H_in, H_out, dXXT)
        if n_groups == 1:
            W = W.reshape(n_heads, head_dim, hidden_size)
            Q = Q.reshape(n_heads, head_dim, hidden_size)
            Q_int_shifted = Q_int_shifted.reshape(n_heads, head_dim, hidden_size)
            QH = Q_int_shifted @ H_in
            diag_QHQ = torch.diagonal(QH @ Q_int_shifted.transpose(-1, -2), dim1=-2, dim2=-1)
            scale = scale.reshape(n_heads, head_dim)
            if dXXT is not None:
                diag_HWdXXTQ = torch.diagonal(H_out @ W @ dXXT @ Q_int_shifted.transpose(-1, -2), dim1=-2, dim2=-1)
            for _ in range(self.n_iters):
                for idx_row in range(head_dim):
                    denominator = H_out[:, idx_row, idx_row] * diag_QHQ[:, idx_row]
                    numerator = (QH[:, idx_row, :].unsqueeze(-2) @ (W - Q).transpose(-1, -2) @ H_out[:, :, idx_row].unsqueeze(-1)).squeeze(dim=(1, 2))
                    if dXXT is not None:
                        numerator += diag_HWdXXTQ[:, idx_row]
                    scale[:, idx_row] += numerator / denominator
                    Q = scale.unsqueeze(-1) * Q_int_shifted

                loss_new = compute_loss_degradation(W, Q, H_in, H_out, dXXT)    
                print(f"Loss: {loss_orig:.2f} -> {loss_new:.2f}")

            # loss_new = compute_loss_degradation(W, Q, H_in, H_out, dXXT)
            # print(f"Loss: {loss_orig:.2f} -> {loss_new:.2f}")
            Q = Q.reshape(n_heads, head_dim, n_groups, group_size)

            return Q, scale
        
        else:
            print('NOT supported for group-wise quantization yet')
            return Q, scale


    def preprocess(self):
        W = self.layer.weight.data.clone().to(device=self.H_in.device)
        W = W.float()

        H_in, H_out = self.H_in.clone(), self.H_out.clone() if self.H_out is not None else None
        dXXT = self.alpha * self.dXXT.clone() if self.dXXT is not None else None 
        
        W, H_in, dXXT = filter_dead_neuron(W, H_in, dXXT, replace=self.hyperparams['replace'], apply_damping=True)
        if H_out is not None:
            H_out = damping(H_out)
        
        if len(H_in.shape) == 2:  # common Hessian for all heads
            H_in = H_in.unsqueeze(0)
            if dXXT is not None:
                dXXT = dXXT.unsqueeze(0)
        
        n_heads = H_out.shape[0] if H_out is not None else H_in.shape[0]
        hidden_size = W.shape[-1]
        head_dim = W.shape[0] // n_heads

        if self.quantizer.group_size == -1:
            W = W.view(n_heads, head_dim, 1, hidden_size)
        else:
            n_groups = hidden_size // self.quantizer.group_size
            W = W.view(n_heads, head_dim, hidden_size).reshape(n_heads, head_dim, n_groups, self.quantizer.group_size)

        self.H_in = None
        self.H_out = None
        self.dXXT = None

        return W, H_in, H_out, dXXT


    def free(self):
        self.H_in = None
        self.H_out = None
        self.dXXT = None

        cleanup_memory(verbose=False)
