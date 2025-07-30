import os
import sys
import numpy as np
import torch
import inspect
import json
import copy
import argparse
import random
import wandb

import config
import models
from data.utils import get_dataset, prepare_dataset
# from optim.train_linear_p import train_linear_probe, eval_with_linear_probe
from optim.train_linear_p_skip import train_linear_probe, eval_with_linear_probe
import distributed

def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config_format', default='base', choices=config.registered_formats())
    parser.add_argument('--resume_ckpt', type=str, default=None, help="Path to the checkpoint to resume training from")
    parser.add_argument('--block_id', type=int, default=7, help="The block ID to train the linear probe on")

    args, rem_args = parser.parse_known_args()

    if args.resume_ckpt is not None:
        if os.path.isfile(args.resume_ckpt):
            args.resume_ckpt, args.resume_ckpt_filename = os.path.split(args.resume_ckpt)
        else:
            args.resume_ckpt_filename = "ckpt.pt"

        # with open(os.path.join(args.resume_ckpt, "summary.json")) as f:
        #     summary = json.load(f)

        # for k, v in summary['args'].items():
        #     if k == "config_format" and args.config_format is not None:
        #         continue
        #     if k not in ["device", "dtype", "resume_ckpt"]:  # 排除 resume_ckpt
        #        setattr(args, k, v)

    return config.parse_args_with_format(format=args.config_format, base_parser=parser, args=rem_args, namespace=args)


def main(args): 
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    distributed_backend = distributed.make_backend_from_args(args)
    args = distributed_backend.get_adjusted_args_for_process(args)

    args.device = torch.device(args.device)
    torch.cuda.set_device(args.device)
    device_type = 'cuda' if 'cuda' in str(args.device) else 'cpu'
    
    print(f"Loading dataset from local files")
    train_data = np.memmap('/data2/ztwu/openwebtext/train.bin', dtype=np.uint16, mode='r')
    val_data = np.memmap('/data2/ztwu/openwebtext/val.bin', dtype=np.uint16, mode='r')
    data = {'train': train_data, 'val': val_data}
    
    print(f"Num training tokens: {len(data['train'])}")
    print(f"Num validation tokens: {len(data['val'])}")
    
    model = models.make_model_from_args(args).to(args.device)

    model = distributed_backend.transform_model(model)
    
        # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Initialize LayerNorm and linear head for the specified block
    layer_norm = torch.nn.LayerNorm(args.n_embd).to(args.device)
    linear_head = torch.nn.Linear(args.n_embd, args.vocab_size).to(args.device)

    # Optimizer
    optimizer = torch.optim.Adam(list(layer_norm.parameters()) + list(linear_head.parameters()), lr=args.lr)

    # Scheduler
    if args.scheduler != 'none':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=args.lr,
            total_steps=args.iterations,
            pct_start=args.warmup_percent,
            anneal_strategy=args.scheduler,
            cycle_momentum=False,
            div_factor=1e2,
            final_div_factor=.1
        )
    else:
        scheduler = None

    # WandB initialization
    if distributed_backend.is_master_process() and args.wandb:
        params_copy = copy.deepcopy(vars(args))
        del params_copy['device']
        wandb.init(project=args.wandb_project, name=args.exp_name, config=params_copy)
    
    ckpt_path = f"{args.results_base_folder}/{args.dataset}/{args.model}/{args.exp_name}"
    if not os.path.exists(ckpt_path):
        if distributed_backend.is_master_process():
            os.makedirs(ckpt_path)
    
    # Load checkpoint
    start_itr = 0
    if args.resume_ckpt is not None:
        checkpoint = torch.load(os.path.join(args.resume_ckpt, args.resume_ckpt_filename), map_location=args.device)
        model.load_state_dict({x: y for x, y in checkpoint['model'].items() if "attn.bias" not in x and "wpe" not in x}, strict=False)
        print(f"Resumed training from checkpoint {args.resume_ckpt}")

    print(f"\nTraining linear probe on block {args.block_id} for model={args.model} \n{vars(args)}\n")

    # Call training function
    stats = train_linear_probe(
        model=model,
        data=data,
        config=args,
        iterations=args.iterations,
        acc_steps=args.acc_steps,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        eval_freq=args.eval_freq,
        block_id=args.block_id,
        extra_args=args,
        lr=args.lr
    )
    
    args.device = None
    args.dtype = None
    stats['args'] = vars(args)
    if distributed_backend.is_master_process():
        with open(f"{ckpt_path}/summary.json", "w") as fs:
            json.dump(stats, fs)
    distributed_backend.finalize()


if __name__ == "__main__":
    args = get_args()
    print(f"Using seed: {args.seed}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print(f"Random seed set to: {args.seed}")
    main(args)