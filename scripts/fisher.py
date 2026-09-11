#!/usr/bin/env python
"""
SqueezeLLM-style diagonal empirical Fisher for arms A / D.

F_ij = sum_n (dL_n / dw_ij)^2 over the 128 calibration sequences of the given seed (same cache TurboBoA uses),
L_n = mean next-token CE of sequence n (labels = inputs). Model forward in its native bf16, gradients in bf16,
accumulation in fp32. Only the transformer-block linear weights are accumulated (embeddings / lm_head are not
quantized). Output: dict {"model.layers.{i}.{name}.weight": fp32 tensor} saved with torch.save.
"""
import argparse, os, time
import torch
from transformers import LlamaForCausalLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm_path", default=os.environ.get("LLAMA_PATH"))
    ap.add_argument("--calib_cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nsamples", type=int, default=None)
    args = ap.parse_args()

    dev = "cuda"
    model = LlamaForCausalLM.from_pretrained(args.llm_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev)
    model.eval()
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    targets = {}
    for i, blk in enumerate(model.model.layers):
        for name, mod in blk.named_modules():
            if isinstance(mod, torch.nn.Linear):
                key = f"model.layers.{i}.{name}.weight"
                mod.weight.requires_grad_(True)
                targets[key] = mod.weight
    F = {k: torch.zeros(v.shape, dtype=torch.float32, device=dev) for k, v in targets.items()}
    print(f"{len(targets)} target tensors, {sum(v.numel() for v in targets.values())/1e6:.1f}M params")

    calib = torch.load(args.calib_cache)
    if args.nsamples:
        calib = calib[: args.nsamples]
    t0 = time.time()
    for n, (inp, _) in enumerate(calib):
        inp = inp.to(dev)
        out = model(input_ids=inp, labels=inp)
        out.loss.backward()
        with torch.no_grad():
            for k, w in targets.items():
                F[k].add_(w.grad.float() ** 2)
                w.grad = None
        if n % 16 == 0:
            print(f"  sample {n}/{len(calib)}  loss {out.loss.item():.3f}  {time.time()-t0:.0f}s  mem {torch.cuda.max_memory_allocated()/2**30:.1f}GB", flush=True)
    F = {k: v.cpu() for k, v in F.items()}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(F, args.out)
    tot = sum(v.sum().item() for v in F.values())
    print(f"saved {args.out}  total Fisher mass {tot:.4g}  time {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
