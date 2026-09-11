import io
import json
import os
import sys
import time

import torch
from contextlib import redirect_stdout

from quantize import turboboa_fwrd
from utils.data_utils import get_calib_data
from utils.eval_utils import evaluate
from utils.model_utils import get_model
from utils.process_args import get_turboboa_arguments, get_turboboa_weight_quant_infos

if __name__ == '__main__':
    args = get_turboboa_arguments()
    
    # load model
    with redirect_stdout(io.StringIO()) as f:
        llm = get_model(args.llm_path)
    llm.seqlen = args.seqlen
    llm.eval()

    # evaluate the fp model performance
    if args.eval_fp:
        results = evaluate(llm, args)
        print(results)
        exit(0)

    # load calib. data
    calib_data = get_calib_data(args)

    # nested arms A/D: static diagonal Fisher
    if args.nested_arm in ('A', 'D'):
        assert args.fisher_path, "--fisher_path required for arms A/D"
        args.fisher_dict = torch.load(args.fisher_path, map_location='cpu')
    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)
        jl = os.path.join(args.dump_dir, 'layers.jsonl')
        if os.path.exists(jl):
            os.remove(jl)

    # quantize
    qconfigs, turboboa_opts, hyperparams = get_turboboa_weight_quant_infos(args)
    print("Start quantization")
    tick = time.time()
    turboboa_fwrd(llm, calib_data, qconfigs, turboboa_opts, hyperparams, args)
    process_time = round(time.time() - tick, 3)
    print(f"Quantization processing time: {process_time}")
    
    # evaluate
    print(args)
    results = evaluate(llm, args)
    results['time'] = process_time
    print(results)
    if args.dump_dir:
        cfg = {k: v for k, v in vars(args).items() if k not in ('fisher_dict', 'tokenizer_cls')}
        with open(os.path.join(args.dump_dir, 'result.json'), 'w') as f:
            json.dump(dict(config=cfg, results=results, argv=sys.argv), f, indent=1)
